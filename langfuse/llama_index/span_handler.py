import httpx
import inspect
from contextvars import ContextVar
from types import GeneratorType
from typing import Optional, Callable, Any, List, Union, Tuple
import uuid

from langfuse.client import (
    StatefulSpanClient,
    StatefulGenerationClient,
    StateType,
    StatefulClient,
)
from langfuse.utils.langfuse_singleton import LangfuseSingleton
from langfuse.utils import _get_timestamp

from logging import getLogger


logger = getLogger("Langfuse_LlamaIndexSpanHandler")

try:
    from llama_index.core.instrumentation.span_handlers import BaseSpanHandler
    from llama_index.core.instrumentation.span import BaseSpan
    from llama_index.core.utilities.token_counting import TokenCounter
    from llama_index.core.base.embeddings.base import BaseEmbedding
    from llama_index.core.base.base_query_engine import BaseQueryEngine
    from llama_index.core.chat_engine.types import StreamingAgentChatResponse
    from llama_index.core.llms import LLM


except ImportError:
    raise ModuleNotFoundError(
        "Please install llama-index to use the Langfuse llama-index integration: 'pip install llama-index'"
    )

context_root: ContextVar[Union[Tuple[str, str], Tuple[None, None]]] = ContextVar(
    "context_root", default=(None, None)
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
        public_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        host: Optional[str] = None,
        release: Optional[str] = None,
        debug: Optional[bool] = None,
        threads: Optional[int] = None,
        flush_at: Optional[int] = None,
        flush_interval: Optional[int] = None,
        max_retries: Optional[int] = None,
        timeout: Optional[int] = None,
        httpx_client: Optional[httpx.Client] = None,
        enabled: Optional[bool] = None,
        sample_rate: Optional[float] = None,
        tokenizer: Optional[Callable[[str], list]] = None,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        trace_name: Optional[str] = None,
        version: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Any] = None,
    ):
        super().__init__()

        self.session_id: Optional[str] = session_id
        self.user_id: Optional[str] = user_id
        self.trace_name: Optional[str] = trace_name
        self.version: Optional[str] = version
        self.tags: Optional[List[str]] = tags
        self.metadata: Optional[Any] = metadata
        self.trace_name: Optional[str] = trace_name

        self.current_trace_id: Optional[str] = None

        self._langfuse = LangfuseSingleton().get(
            public_key=public_key,
            secret_key=secret_key,
            host=host,
            release=release,
            debug=debug,
            threads=threads,
            flush_at=flush_at,
            flush_interval=flush_interval,
            max_retries=max_retries,
            timeout=timeout,
            httpx_client=httpx_client,
            enabled=enabled,
            sample_rate=sample_rate,
            sdk_integration="llama-index_instrumentation",
        )

        self._token_counter = TokenCounter(tokenizer)

    def new_span(
        self,
        id_: str,
        bound_args: inspect.BoundArguments,
        instance: Optional[Any] = None,
        parent_span_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Optional[LangfuseSpan]:
        logger.debug(
            f"Creating new span {instance.__class__.__name__} with ID {id_} and parend ID {parent_span_id}"
        )
        trace_id, root_span_id = context_root.get()
        name = instance.__class__.__name__

        # Create wrapper trace for the first span
        if not parent_span_id:
            if trace_id is not None:
                logger.warning(
                    f"Trace ID {trace_id} already exists, but no parent span ID was provided. This span will be treated as a new trace."
                )

            trace_id = str(uuid.uuid4())
            root_span_id = id_
            self.current_trace_id = trace_id
            context_root.set((trace_id, root_span_id))

            self._langfuse.trace(
                id=trace_id,
                name=self.trace_name or name,
                input=bound_args.arguments,
                metadata=self.metadata,
                version=self.version,
                tags=self.tags,
                user_id=self.user_id,
                session_id=self.session_id,
                release=self._langfuse.release,
                timestamp=_get_timestamp(),
            )

        if not trace_id:
            logger.warning(
                f"Span ID {id_} is being dropped without a trace ID. This span will not be recorded."
            )
            return

        if self._is_generation(instance):
            self._langfuse.generation(
                id=id_,
                trace_id=trace_id,
                start_time=_get_timestamp(),
                parent_observation_id=parent_span_id,
                name=name,
                input=self._parse_generation_input(bound_args, instance),
                metadata=kwargs,
            )

        else:
            self._langfuse.span(
                id=id_,
                trace_id=trace_id,
                start_time=_get_timestamp(),
                parent_observation_id=parent_span_id,
                name=name,
                input=bound_args.arguments,
                metadata=kwargs,
            )

    def prepare_to_exit_span(
        self,
        id_: str,
        bound_args: inspect.BoundArguments,
        instance: Optional[Any] = None,
        result: Optional[Any] = None,
        **kwargs: Any,
    ) -> Optional[LangfuseSpan]:
        logger.debug(f"Exiting span {instance.__class__.__name__} with ID {id_}")
        trace_id, root_span_id = context_root.get()

        if not trace_id:
            logger.warning(
                f"Span ID {id_} is being dropped without a trace ID. This span will not be recorded."
            )
            return

        output, metadata = self._parse_output_metadata(instance, result)

        # Reset the context root if the span is the root span
        if id_ == root_span_id:
            self._langfuse.trace(
                id=self.current_trace_id, output=output, metadata=metadata
            )
            context_root.set((None, None))

        if self._is_generation(instance):
            generationClient = self._get_generation_client(id_, trace_id)
            generationClient.end(output=output, metadata=metadata)

        else:
            spanClient = self._get_span_client(id_, trace_id)
            spanClient.end(output=output, metadata=metadata)

    def prepare_to_drop_span(
        self,
        id_: str,
        bound_args: inspect.BoundArguments,
        instance: Optional[Any] = None,
        err: Optional[BaseException] = None,
        **kwargs: Any,
    ) -> Optional[LangfuseSpan]:
        logger.debug(f"Dropping span {instance.__class__.__name__} with ID {id_}")
        trace_id, root_span_id = context_root.get()

        if not trace_id:
            logger.warning(
                f"Span ID {id_} is being dropped without a trace ID. This span will not be recorded."
            )
            return

        # Reset the context root if the span is the root span
        if id_ == root_span_id:
            self._langfuse.trace(id=trace_id, output=str(err))
            context_root.set((None, None))

        if self._is_generation(instance):
            generationClient = self._get_generation_client(id_, trace_id)
            generationClient.end(
                level="ERROR",
                status_message=str(err),
            )

        else:
            spanClient = self._get_span_client(id_, trace_id)
            spanClient.end(
                level="ERROR",
                status_message=str(err),
            )

    def flush(self) -> None:
        """Flushes the Langfuse client."""
        self._langfuse.flush()

    def _is_generation(self, instance: Optional[Any]) -> bool:
        return isinstance(instance, (BaseEmbedding, LLM))

    def get_current_trace_id(self) -> Optional[str]:
        """Get the latest trace id. This can be the the ID of an ongoing or completed trace."""
        return self.current_trace_id

    def _get_generation_client(
        self, id: str, trace_id: str
    ) -> StatefulGenerationClient:
        return StatefulGenerationClient(
            client=self._langfuse.client,
            id=id,
            trace_id=trace_id,
            task_manager=self._langfuse.task_manager,
            state_type=StateType.OBSERVATION,
        )

    def _get_span_client(self, id: str, trace_id: str) -> StatefulSpanClient:
        return StatefulSpanClient(
            client=self._langfuse.client,
            id=id,
            trace_id=trace_id,
            task_manager=self._langfuse.task_manager,
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
        if not result or isinstance(result, StreamingAgentChatResponse):
            # Todo: use event handler to populate IO
            return None, None

        if isinstance(instance, BaseEmbedding) and isinstance(result, list):
            return {
                "num_embeddings": 1
                if len(result) > 0 and not isinstance(result[0], list)
                else len(result)
            }, None

        if isinstance(result, GeneratorType):
            return "".join(list(result)), None

        if isinstance(instance, BaseQueryEngine) and "response" in result.__dict__:
            metadata_dict = {
                key: val for key, val in result.__dict__.items() if key != "response"
            }

            return result.response, metadata_dict

        return result, None
