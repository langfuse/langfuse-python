import inspect
from typing import Optional, Any, Tuple, Dict, Generator, AsyncGenerator
import uuid

from langfuse.client import (
    Langfuse,
    StatefulSpanClient,
    StatefulGenerationClient,
    StateType,
    StatefulClient,
)

from logging import getLogger
from ._context import InstrumentorContext

logger = getLogger(__name__)

try:
    from llama_index.core.instrumentation.span_handlers import BaseSpanHandler
    from llama_index.core.instrumentation.span import BaseSpan
    from llama_index.core.base.embeddings.base import BaseEmbedding
    from llama_index.core.base.base_query_engine import BaseQueryEngine
    from llama_index.core.llms import LLM, ChatResponse
    from llama_index.core.base.response.schema import (
        StreamingResponse,
        AsyncStreamingResponse,
    )

except ImportError:
    raise ModuleNotFoundError(
        "Please install llama-index to use the Langfuse llama-index integration: 'pip install llama-index'"
    )


class LangfuseSpan(BaseSpan):
    """Langfuse Span."""

    client: StatefulClient


class LlamaIndexSpanHandler(BaseSpanHandler[LangfuseSpan], extra="allow"):
    """[BETA] Span Handler for exporting LlamaIndex instrumentation module spans to Langfuse.

    This beta integration is currently under active development and subject to change. Please provide feedback to [the Langfuse team](https://github.com/langfuse/langfuse/issues/1931).

    For production setups, please use the existing callback-based integration (LlamaIndexCallbackHandler).

    Usage:

    ```python
    import llama_index.core.instrumentation as instrument
    from langfuse.llama_index import LlamaIndexSpanHandler

    langfuse_span_handler = LlamaIndexSpanHandler()
    instrument.get_dispatcher().add_span_handler(langfuse_span_handler)
    ```
    """

    def __init__(
        self,
        *,
        langfuse_client: Langfuse,
        observation_updates: Dict[str, Dict[str, Any]],
    ):
        super().__init__()

        self._langfuse_client = langfuse_client
        self._observation_updates = observation_updates
        self._context = InstrumentorContext()

    def new_span(
        self,
        id_: str,
        bound_args: inspect.BoundArguments,
        instance: Optional[Any] = None,
        parent_span_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Optional[LangfuseSpan]:
        logger.debug(
            f"Creating new span {instance.__class__.__name__} with ID {id_} and parent ID {parent_span_id}"
        )
        trace_id = self._context.trace_id
        instance_name = type(instance).__name__
        qual_name = self._parse_qualname(id_)  # qualname is the first part of the id_

        if not parent_span_id:
            self._context.update(root_llama_index_span_id=id_)

            if not self._context.parent_observation_id:
                trace_id = self._context.trace_id or str(uuid.uuid4())
                self._context.update(trace_id=trace_id)

                if self._context.update_parent:
                    self._langfuse_client.trace(
                        **self._context.trace_data,
                        id=trace_id,
                        name=self._context.trace_name or instance_name,
                        input=bound_args.arguments,
                    )

        if not trace_id:
            logger.warning(
                f"Span ID {id_} is being dropped without a trace ID. This span will not be recorded."
            )
            return

        if self._is_generation(id_, instance):
            self._langfuse_client.generation(
                id=id_,
                trace_id=trace_id,
                parent_observation_id=parent_span_id
                or self._context.parent_observation_id,
                name=qual_name or instance_name,
                input=self._parse_generation_input(bound_args, instance),
                metadata=kwargs,
            )

        else:
            self._langfuse_client.span(
                id=id_,
                trace_id=trace_id,
                parent_observation_id=parent_span_id
                or self._context.parent_observation_id,
                name=qual_name or instance_name,
                input=bound_args.arguments,
                metadata=kwargs,
            )

        # Initialize observation update for the span to be populated by event handler
        self._observation_updates[id_] = {}

    def prepare_to_exit_span(
        self,
        id_: str,
        bound_args: inspect.BoundArguments,
        instance: Optional[Any] = None,
        result: Optional[Any] = None,
        **kwargs: Any,
    ) -> Optional[LangfuseSpan]:
        logger.debug(f"Exiting span {instance.__class__.__name__} with ID {id_}")

        observation_updates = self._observation_updates.pop(id_, {})
        output, metadata = self._parse_output_metadata(instance, result)

        # Reset the context root if the span is the root span
        if id_ == self._context.root_llama_index_span_id:
            if self._context.update_parent:
                self._langfuse_client.trace(
                    id=self._context.trace_id, output=output, metadata=metadata
                )

            if not self._context.is_user_managed_trace:
                self._context.reset_trace_id()

        if self._is_generation(id_, instance):
            generationClient = self._get_generation_client(id_)
            generationClient.end(
                **observation_updates,
                output=output,
                metadata=metadata,
            )

        else:
            spanClient = self._get_span_client(id_)
            spanClient.end(
                **observation_updates,
                output=output,
                metadata=metadata,
            )

    def prepare_to_drop_span(
        self,
        id_: str,
        bound_args: inspect.BoundArguments,
        instance: Optional[Any] = None,
        err: Optional[BaseException] = None,
        **kwargs: Any,
    ) -> Optional[LangfuseSpan]:
        logger.debug(f"Dropping span {instance.__class__.__name__} with ID {id_}")

        observation_updates = self._observation_updates.pop(id_, {})

        # Reset the context root if the span is the root span
        if id_ == self._context.root_llama_index_span_id:
            if self._context.update_parent:
                self._langfuse_client.trace(
                    id=self._context.trace_id,
                    output=str(err),
                )

            if not self._context.is_user_managed_trace:
                self._context.reset_trace_id()

        if self._is_generation(id_, instance):
            generationClient = self._get_generation_client(id_)
            generationClient.end(
                **observation_updates,
                level="ERROR",
                status_message=str(err),
            )

        else:
            spanClient = self._get_span_client(id_)
            spanClient.end(
                **observation_updates,
                level="ERROR",
                status_message=str(err),
            )

    def _is_generation(self, id_: str, instance: Optional[Any] = None) -> bool:
        """Check if the instance is a generation (embedding or LLM).

        Verifies if the instance is a subclass of BaseEmbedding or LLM,
        but not these base classes themselves.

        Args:
            id_ (str): ID for parsing qualified name.
            instance (Optional[Any]): Instance to check.

        Returns:
            bool: True if instance is a valid generation, False otherwise.
        """
        qual_name = self._parse_qualname(id_)

        return (
            qual_name is not None
            and isinstance(instance, (BaseEmbedding, LLM))
            and not (
                any(
                    base_class_name in qual_name
                    for base_class_name in ("BaseEmbedding", "LLM")
                )
                and qual_name not in ("BaseEmbedding.get_text_embedding_batch")
            )
        )

    def _get_generation_client(self, id: str) -> StatefulGenerationClient:
        trace_id = self._context.trace_id
        if trace_id is None:
            raise ValueError("Trace ID is not set")

        return StatefulGenerationClient(
            client=self._langfuse_client.client,
            id=id,
            trace_id=trace_id,
            task_manager=self._langfuse_client.task_manager,
            state_type=StateType.OBSERVATION,
        )

    def _get_span_client(self, id: str) -> StatefulSpanClient:
        trace_id = self._context.trace_id
        if trace_id is None:
            raise ValueError("Trace ID is not set")

        return StatefulSpanClient(
            client=self._langfuse_client.client,
            id=id,
            trace_id=trace_id,
            task_manager=self._langfuse_client.task_manager,
            state_type=StateType.OBSERVATION,
        )

    def _parse_generation_input(
        self,
        bound_args: inspect.BoundArguments,
        instance: Optional[Any] = None,
    ):
        if isinstance(instance, BaseEmbedding) and "texts" in bound_args.arguments:
            return {"num_texts": len(bound_args.arguments["texts"])}

        return bound_args.arguments

    def _parse_output_metadata(
        self, instance: Optional[Any], result: Optional[Any]
    ) -> Tuple[Optional[Any], Optional[Any]]:
        if not result or isinstance(
            result,
            (Generator, AsyncGenerator, StreamingResponse, AsyncStreamingResponse),
        ):
            return None, None

        if isinstance(result, ChatResponse):
            return result.message, None

        if isinstance(instance, BaseEmbedding) and isinstance(result, list):
            return {
                "num_embeddings": 1
                if len(result) > 0 and not isinstance(result[0], list)
                else len(result)
            }, None

        if isinstance(instance, BaseQueryEngine) and "response" in result.__dict__:
            metadata_dict = {
                key: val
                for key, val in result.__dict__.items()
                if key != "response"
                and not isinstance(val, (Generator, AsyncGenerator))
            }

            return result.response, metadata_dict

        return result, None

    def _parse_qualname(self, id_: str) -> Optional[str]:
        return id_.split("-")[0] if "-" in id_ else None
