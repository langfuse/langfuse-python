from collections import defaultdict
from contextvars import ContextVar
from typing import Any, Dict, List, Optional, Union, Tuple, Callable, Generator
from uuid import uuid4
import logging

from langfuse.client import (
    StatefulSpanClient,
    StatefulTraceClient,
    StatefulGenerationClient,
    StateType,
)
from langfuse.utils.error_logging import (
    auto_decorate_methods_with,
    catch_and_log_errors,
)
from langfuse.types import TraceMetadata
from langfuse.utils.base_callback_handler import LangfuseBaseCallbackHandler
from .utils import CallbackEvent, ParsedLLMEndPayload

try:
    from llama_index.core.callbacks.base_handler import (
        BaseCallbackHandler as LlamaIndexBaseCallbackHandler,
    )
    from llama_index.core.callbacks.schema import (
        CBEventType,
        BASE_TRACE_EVENT,
        EventPayload,
    )
    from llama_index.core.utilities.token_counting import TokenCounter
except ImportError:
    raise ModuleNotFoundError(
        "Please install llama-index to use the Langfuse llama-index integration: 'pip install llama-index'"
    )

context_root: ContextVar[Optional[Union[StatefulTraceClient, StatefulSpanClient]]] = (
    ContextVar("root", default=None)
)
context_trace_metadata: ContextVar[TraceMetadata] = ContextVar(
    "trace_metadata",
    default={
        "name": None,
        "user_id": None,
        "session_id": None,
        "version": None,
        "release": None,
        "metadata": None,
        "tags": None,
        "public": None,
    },
)


@auto_decorate_methods_with(catch_and_log_errors, exclude=["__init__"])
class LlamaIndexCallbackHandler(
    LlamaIndexBaseCallbackHandler, LangfuseBaseCallbackHandler
):
    """LlamaIndex callback handler for Langfuse. This version is in alpha and may change in the future."""

    log = logging.getLogger("langfuse")

    def __init__(
        self,
        *,
        public_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        host: Optional[str] = None,
        debug: bool = False,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        trace_name: Optional[str] = None,
        release: Optional[str] = None,
        version: Optional[str] = None,
        tags: Optional[List[str]] = None,
        threads: Optional[int] = None,
        flush_at: Optional[int] = None,
        flush_interval: Optional[int] = None,
        max_retries: Optional[int] = None,
        timeout: Optional[int] = None,
        event_starts_to_ignore: Optional[List[CBEventType]] = None,
        event_ends_to_ignore: Optional[List[CBEventType]] = None,
        tokenizer: Optional[Callable[[str], list]] = None,
        sdk_integration: Optional[str] = None,
    ) -> None:
        LlamaIndexBaseCallbackHandler.__init__(
            self,
            event_starts_to_ignore=event_starts_to_ignore or [],
            event_ends_to_ignore=event_ends_to_ignore or [],
        )
        LangfuseBaseCallbackHandler.__init__(
            self,
            public_key=public_key,
            secret_key=secret_key,
            host=host,
            debug=debug,
            session_id=session_id,
            user_id=user_id,
            trace_name=trace_name,
            release=release,
            version=version,
            threads=threads,
            flush_at=flush_at,
            flush_interval=flush_interval,
            max_retries=max_retries,
            timeout=timeout,
            sdk_integration=sdk_integration or "llama-index_callback",
        )

        self.event_map: Dict[str, List[CallbackEvent]] = defaultdict(list)
        self._llama_index_trace_name: Optional[str] = None
        self._token_counter = TokenCounter(tokenizer)
        self.tags = tags

        # For stream-chat, the last LLM end_event arrives after the trace has ended
        # Keep track of these orphans to upsert them with the correct trace_id after the trace has ended
        self._orphaned_LLM_generations: Dict[
            str, Tuple[StatefulGenerationClient, StatefulTraceClient]
        ] = {}

    def set_root(
        self, root: Optional[Union[StatefulTraceClient, StatefulSpanClient]]
    ) -> None:
        context_root.set(root)

        if root is None:
            self.trace = None
            self.root_span = None
            self._task_manager = self.langfuse.task_manager if self.langfuse else None

            return

        if isinstance(root, StatefulTraceClient):
            self.trace = root

        elif isinstance(root, StatefulSpanClient):
            self.root_span = root
            self.trace = StatefulTraceClient(
                root.client,
                root.trace_id,
                StateType.TRACE,
                root.trace_id,
                root.task_manager,
            )

        self._task_manager = root.task_manager

    def set_trace_params(
        self,
        name: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        version: Optional[str] = None,
        release: Optional[str] = None,
        metadata: Optional[Any] = None,
        tags: Optional[List[str]] = None,
        public: Optional[bool] = None,
    ):
        """Set the trace params that will be used for all following operations.

        Allows setting params of subsequent traces at any point in the code.
        Overwrites the default params set in the callback constructor.

        Attention: If a root trace or span is set on the callback handler, those trace params will be used and NOT those set through this method.

        Parameters:
        - name (Optional[str]): Identifier of the trace. Useful for sorting/filtering in the UI.
        - user_id (Optional[str]): The id of the user that triggered the execution. Used to provide user-level analytics.
        - session_id (Optional[str]): Used to group multiple traces into a session in Langfuse. Use your own session/thread identifier.
        - version (Optional[str]): The version of the trace type. Used to understand how changes to the trace type affect metrics. Useful in debugging.
        - metadata (Optional[Any]): Additional metadata of the trace. Can be any JSON object. Metadata is merged when being updated via the API.
        - tags (Optional[List[str]]): Tags are used to categorize or label traces. Traces can be filtered by tags in the Langfuse UI and GET API.
        - public (Optional[bool]): You can make a trace public to share it via a public link. This allows others to view the trace without needing to log in or be members of your Langfuse project.


        Returns:
        None
        """
        context_trace_metadata.set(
            {
                "name": name,
                "user_id": user_id,
                "session_id": session_id,
                "version": version,
                "release": release,
                "metadata": metadata,
                "tags": tags,
                "public": public,
            }
        )

    def start_trace(self, trace_id: Optional[str] = None) -> None:
        """Run when an overall trace is launched."""
        self._llama_index_trace_name = trace_id

    def end_trace(
        self,
        trace_id: Optional[str] = None,
        trace_map: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        """Run when an overall trace is exited."""
        if not trace_map:
            self.log.debug("No events in trace map to create the observation tree.")
            return

        # Generate Langfuse observations after trace has ended and full trace_map is available.
        # For long-running traces this leads to events only being sent to Langfuse after the trace has ended.
        # Timestamps remain accurate as they are set at the time of the event.
        self._create_observations_from_trace_map(
            event_id=BASE_TRACE_EVENT, trace_map=trace_map
        )
        self._update_trace_data(trace_map=trace_map)

    def on_event_start(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        parent_id: str = "",
        **kwargs: Any,
    ) -> str:
        """Run when an event starts and return id of event."""
        start_event = CallbackEvent(
            event_id=event_id, event_type=event_type, payload=payload
        )
        self.event_map[event_id].append(start_event)

        return event_id

    def on_event_end(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        **kwargs: Any,
    ) -> None:
        """Run when an event ends."""
        end_event = CallbackEvent(
            event_id=event_id, event_type=event_type, payload=payload
        )
        self.event_map[event_id].append(end_event)

        if event_type == CBEventType.LLM and event_id in self._orphaned_LLM_generations:
            generation, trace = self._orphaned_LLM_generations[event_id]
            self._handle_orphaned_LLM_end_event(
                end_event, generation=generation, trace=trace
            )
            del self._orphaned_LLM_generations[event_id]

    def _create_observations_from_trace_map(
        self,
        event_id: str,
        trace_map: Dict[str, List[str]],
        parent: Optional[
            Union[StatefulTraceClient, StatefulSpanClient, StatefulGenerationClient]
        ] = None,
    ) -> None:
        """Recursively create langfuse observations based on the trace_map."""
        if event_id != BASE_TRACE_EVENT and not self.event_map.get(event_id):
            return

        if event_id == BASE_TRACE_EVENT:
            observation = self._get_root_observation()
        else:
            observation = self._create_observation(
                event_id=event_id, parent=parent, trace_id=self.trace.id
            )

        for child_event_id in trace_map.get(event_id, []):
            self._create_observations_from_trace_map(
                event_id=child_event_id, parent=observation, trace_map=trace_map
            )

    def _get_root_observation(self) -> Union[StatefulTraceClient, StatefulSpanClient]:
        user_provided_root = context_root.get()

        # Make sure that if a user-provided root is set, it has been set in the same trace
        # and it's not a root from a different trace
        if (
            user_provided_root is not None
            and self.trace
            and self.trace.id == user_provided_root.trace_id
        ):
            return user_provided_root

        else:
            trace_metadata = context_trace_metadata.get()
            name = (
                trace_metadata["name"]
                or self.trace_name
                or f"LlamaIndex_{self._llama_index_trace_name}"
            )
            version = trace_metadata["version"] or self.version
            release = trace_metadata["release"] or self.release
            session_id = trace_metadata["session_id"] or self.session_id
            user_id = trace_metadata["user_id"] or self.user_id
            metadata = trace_metadata["metadata"]
            tags = trace_metadata["tags"] or self.tags
            public = trace_metadata["public"] or None

            self.trace = self.langfuse.trace(
                id=str(uuid4()),
                name=name,
                version=version,
                session_id=session_id,
                user_id=user_id,
                metadata=metadata,
                tags=tags,
                release=release,
                public=public,
            )

            return self.trace

    def _create_observation(
        self,
        event_id: str,
        parent: Union[
            StatefulTraceClient, StatefulSpanClient, StatefulGenerationClient
        ],
        trace_id: str,
    ) -> Union[StatefulSpanClient, StatefulGenerationClient]:
        event_type = self.event_map[event_id][0].event_type

        if event_type == CBEventType.LLM:
            return self._handle_LLM_events(event_id, parent, trace_id)
        elif event_type == CBEventType.EMBEDDING:
            return self._handle_embedding_events(event_id, parent, trace_id)
        else:
            return self._handle_span_events(event_id, parent, trace_id)

    def _handle_LLM_events(
        self,
        event_id: str,
        parent: Union[
            StatefulTraceClient, StatefulSpanClient, StatefulGenerationClient
        ],
        trace_id: str,
    ) -> StatefulGenerationClient:
        events = self.event_map[event_id]
        start_event, end_event = events[0], events[-1]

        if start_event.payload and EventPayload.SERIALIZED in start_event.payload:
            serialized = start_event.payload.get(EventPayload.SERIALIZED, {})
            name = serialized.get("class_name", "LLM")
            temperature = serialized.get("temperature", None)
            max_tokens = serialized.get("max_tokens", None)
            timeout = serialized.get("timeout", None)

        parsed_end_payload = self._parse_LLM_end_event_payload(end_event)
        parsed_metadata = self._parse_metadata_from_event(end_event)

        generation = parent.generation(
            id=event_id,
            trace_id=trace_id,
            version=self.version,
            name=name,
            start_time=start_event.time,
            metadata=parsed_metadata,
            model_parameters={
                "temperature": temperature,
                "max_tokens": max_tokens,
                "request_timeout": timeout,
            },
            **parsed_end_payload,
        )

        # Register orphaned LLM event (only start event, no end event) to be later upserted with the correct trace_id
        if len(events) == 1:
            self._orphaned_LLM_generations[event_id] = (generation, self.trace)

        return generation

    def _handle_orphaned_LLM_end_event(
        self,
        end_event: CallbackEvent,
        generation: StatefulGenerationClient,
        trace: StatefulTraceClient,
    ) -> None:
        parsed_end_payload = self._parse_LLM_end_event_payload(end_event)

        generation.update(
            **parsed_end_payload,
        )

        if generation.trace_id != trace.id:
            raise ValueError(
                f"Generation trace_id {generation.trace_id} does not match trace.id {trace.id}"
            )

        trace.update(output=parsed_end_payload["output"])

    def _parse_LLM_end_event_payload(
        self, end_event: CallbackEvent
    ) -> ParsedLLMEndPayload:
        result: ParsedLLMEndPayload = {
            "input": None,
            "output": None,
            "usage": None,
            "model": None,
            "end_time": end_event.time,
        }

        if not end_event.payload:
            return result

        result["input"] = self._parse_input_from_event(end_event)
        result["output"] = self._parse_output_from_event(end_event)
        result["model"], result["usage"] = self._parse_usage_from_event_payload(
            end_event.payload
        )

        return result

    def _parse_usage_from_event_payload(self, event_payload: Dict):
        model = usage = None

        if not (
            EventPayload.MESSAGES in event_payload
            and EventPayload.RESPONSE in event_payload
        ):
            return model, usage

        response = event_payload.get(EventPayload.RESPONSE)

        if hasattr(response, "raw"):
            model = response.raw.get("model")
            token_usage = response.raw.get("usage", {})

            if token_usage:
                usage = {
                    "input": getattr(token_usage, "prompt_tokens", None),
                    "output": getattr(token_usage, "completion_tokens", None),
                    "total": getattr(token_usage, "total_tokens", None),
                }

        return model, usage

    def _handle_embedding_events(
        self,
        event_id: str,
        parent: Union[
            StatefulTraceClient, StatefulSpanClient, StatefulGenerationClient
        ],
        trace_id: str,
    ) -> StatefulGenerationClient:
        events = self.event_map[event_id]
        start_event, end_event = events[0], events[-1]

        if start_event.payload and EventPayload.SERIALIZED in start_event.payload:
            serialized = start_event.payload.get(EventPayload.SERIALIZED, {})
            name = serialized.get("class_name", "Embedding")
            model = serialized.get("model_name", None)
            timeout = serialized.get("timeout", None)

        if end_event.payload:
            chunks = end_event.payload.get(EventPayload.CHUNKS, [])
            input = {"num_chunks": len(chunks)}
            embeddings = end_event.payload.get(EventPayload.EMBEDDINGS, [])
            output = {"num_embeddings": len(embeddings)}

            token_count = sum(
                self._token_counter.get_string_tokens(chunk) for chunk in chunks
            )

            usage = {
                "input": 0,
                "output": 0,
                "total": token_count or None,
            }

        generation = parent.generation(
            id=event_id,
            trace_id=trace_id,
            name=name,
            start_time=start_event.time,
            end_time=end_event.time,
            version=self.version,
            model=model,
            input=input,
            output=output,
            usage=usage or None,
            model_parameters={
                "request_timeout": timeout,
            },
        )

        return generation

    def _handle_span_events(
        self,
        event_id: str,
        parent: Union[
            StatefulTraceClient, StatefulSpanClient, StatefulGenerationClient
        ],
        trace_id: str,
    ) -> StatefulSpanClient:
        start_event, end_event = self.event_map[event_id]

        extracted_input = self._parse_input_from_event(start_event)
        extracted_output = self._parse_output_from_event(end_event)
        extracted_metadata = self._parse_metadata_from_event(end_event)

        metadata = (
            extracted_metadata if extracted_output != extracted_metadata else None
        )

        span = parent.span(
            id=event_id,
            trace_id=trace_id,
            start_time=start_event.time,
            name=start_event.event_type.value,
            version=self.version,
            session_id=self.session_id,
            input=extracted_input,
            output=extracted_output,
            metadata=metadata,
        )

        if end_event:
            span.end(end_time=end_event.time)

        return span

    def _update_trace_data(self, trace_map):
        if context_root.get():  # Exit early if root is user-provided.
            return

        child_event_ids = trace_map.get(BASE_TRACE_EVENT, [])
        if not child_event_ids:
            return

        event_pair = self.event_map.get(child_event_ids[0])
        if not event_pair or len(event_pair) < 2:
            return

        start_event, end_event = event_pair
        input = self._parse_input_from_event(start_event)
        output = self._parse_output_from_event(end_event)

        if input or output:
            self.trace.update(input=input, output=output)

    def _parse_input_from_event(self, event: CallbackEvent):
        if event.payload is None:
            return

        if (
            event.event_type == CBEventType.NODE_PARSING
            and EventPayload.DOCUMENTS in event.payload
        ):
            documents = event.payload.pop(EventPayload.DOCUMENTS)
            event.payload["documents"] = [doc.metadata for doc in documents]
            return event.payload

        for key in [EventPayload.MESSAGES, EventPayload.QUERY_STR, EventPayload.PROMPT]:
            if key in event.payload:
                return event.payload.get(key)

        return event.payload

    def _parse_output_from_event(self, event: CallbackEvent):
        if event.payload is None:
            return

        if (
            event.event_type == CBEventType.NODE_PARSING
            and EventPayload.NODES in event.payload
        ):
            nodes = event.payload.pop(EventPayload.NODES)
            event.payload["num_nodes"] = len(nodes)
            return event.payload

        if (
            event.event_type == CBEventType.CHUNKING
            and EventPayload.CHUNKS in event.payload
        ):
            chunks = event.payload.pop(EventPayload.CHUNKS)
            event.payload["num_chunks"] = len(chunks)

        if EventPayload.COMPLETION in event.payload:
            return event.payload.get(EventPayload.COMPLETION)

        if EventPayload.RESPONSE in event.payload:
            response = event.payload.get(EventPayload.RESPONSE)

            # Skip streaming responses as consuming them would block the user's execution path
            if "Streaming" in type(response).__name__:
                return None

            if hasattr(response, "response"):
                return response.response

            if hasattr(response, "message"):
                output = dict(response.message)
                if "additional_kwargs" in output:
                    if "tool_calls" in output["additional_kwargs"]:
                        output["tool_calls"] = output["additional_kwargs"]["tool_calls"]

                    del output["additional_kwargs"]

                return output

        return event.payload

    def _parse_metadata_from_event(self, event: CallbackEvent):
        if event.payload is None:
            return

        metadata = {}

        for key in event.payload.keys():
            if key not in [
                EventPayload.MESSAGES,
                EventPayload.QUERY_STR,
                EventPayload.PROMPT,
                EventPayload.COMPLETION,
                EventPayload.SERIALIZED,
                "additional_kwargs",
            ]:
                if key != EventPayload.RESPONSE:
                    metadata[key] = event.payload[key]
                else:
                    response = event.payload.get(EventPayload.RESPONSE)

                    for res_key, value in vars(response).items():
                        if (
                            not res_key.startswith("_")
                            and res_key
                            not in [
                                "response",
                                "response_txt",
                                "message",
                                "additional_kwargs",
                                "delta",
                                "raw",
                            ]
                            and not isinstance(value, Generator)
                        ):
                            metadata[res_key] = value

        return metadata or None
