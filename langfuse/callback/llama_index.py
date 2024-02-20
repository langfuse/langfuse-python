from collections import defaultdict
from typing import Any, Dict, List, Optional, Union, Tuple, Callable
from uuid import uuid4
import logging

from langfuse.client import (
    StatefulSpanClient,
    StatefulTraceClient,
    StatefulGenerationClient,
)
from langfuse.decorators.error_logging import (
    auto_decorate_methods_with,
    catch_and_log_errors,
)
from langfuse.callback.base import BaseCallbackHandler as LangfuseBaseCallbackHandler
from langfuse.callback.utils import CallbackEvent

try:
    from llama_index.core.callbacks.base_handler import (
        BaseCallbackHandler as LLamaIndexBaseCallbackHandler,
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


@auto_decorate_methods_with(catch_and_log_errors, exclude=["__init__"])
class LLamaIndexCallbackHandler(
    LLamaIndexBaseCallbackHandler, LangfuseBaseCallbackHandler
):
    log = logging.getLogger("langfuse")

    def __init__(
        self,
        *,
        public_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        host: Optional[str] = None,
        debug: bool = False,
        stateful_client: Optional[
            Union[StatefulTraceClient, StatefulSpanClient]
        ] = None,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        trace_name: Optional[str] = None,
        release: Optional[str] = None,
        version: Optional[str] = None,
        threads: Optional[int] = None,
        flush_at: Optional[int] = None,
        flush_interval: Optional[int] = None,
        max_retries: Optional[int] = None,
        timeout: Optional[int] = None,
        event_starts_to_ignore: Optional[List[CBEventType]] = None,
        event_ends_to_ignore: Optional[List[CBEventType]] = None,
        tokenizer: Optional[Callable[[str], list]] = None,
    ) -> None:
        LLamaIndexBaseCallbackHandler.__init__(
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
            stateful_client=stateful_client,
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
            sdk_integration="llama-index",
        )

        self.root = stateful_client
        self.event_map: Dict[str, List[CallbackEvent]] = defaultdict(list)
        self._llama_index_trace_name: Optional[str] = None
        self._token_counter = TokenCounter(tokenizer)

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
        if self.root is not None:
            return self.root  # return user-provided root trace or span

        else:
            self.trace = self.langfuse.trace(
                id=str(uuid4()),
                name=self.trace_name or f"LlamaIndex_{self._llama_index_trace_name}",
                version=self.version,
                session_id=self.session_id,
                user_id=self.user_id,
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
        start_event, end_event = self.event_map[event_id]

        if start_event.payload and EventPayload.SERIALIZED in start_event.payload:
            serialized = start_event.payload.get(EventPayload.SERIALIZED, {})
            name = serialized.get("class_name", "LLM")
            temperature = serialized.get("temperature", None)
            max_tokens = serialized.get("max_tokens", None)
            timeout = serialized.get("timeout", None)

        if end_event.payload:
            if EventPayload.PROMPT in end_event.payload:
                input = end_event.payload.get(EventPayload.PROMPT)
                output = end_event.payload.get(EventPayload.COMPLETION)

            elif EventPayload.MESSAGES in end_event.payload:
                input = end_event.payload.get(EventPayload.MESSAGES)
                response = end_event.payload.get(EventPayload.RESPONSE, {})
                output = response.message.copy()
                if hasattr(output, "additional_kwargs"):
                    delattr(output, "additional_kwargs")
                model = response.raw.get("model", None)
                token_usage = dict(response.raw.get("usage", {}))
                usage = None
                if token_usage:
                    usage = {
                        "input": token_usage.get("prompt_tokens"),
                        "output": token_usage.get("completion_tokens"),
                        "total": token_usage.get("total_tokens"),
                    }

        generation = parent.generation(
            id=event_id,
            trace_id=trace_id,
            version=self.version,
            name=name,
            start_time=start_event.time,
            end_time=end_event.time,
            usage=usage or None,
            model=model,
            input=input,
            output=output,
            metadata=end_event.payload,
            model_parameters={
                "temperature": temperature,
                "max_tokens": max_tokens,
                "request_timeout": timeout,
            },
        )

        return generation

    def _handle_embedding_events(
        self,
        event_id: str,
        parent: Union[
            StatefulTraceClient, StatefulSpanClient, StatefulGenerationClient
        ],
        trace_id: str,
    ) -> StatefulGenerationClient:
        start_event, end_event = self.event_map[event_id]

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

            # usage = None
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
        input = start_event.payload
        output = end_event.payload

        if start_event.event_type == CBEventType.NODE_PARSING:
            input, output = self._handle_node_parsing_payload(self.event_map[event_id])

        elif start_event.event_type == CBEventType.CHUNKING:
            input, output = self.handle_chunking_payload(self.event_map[event_id])

        span = parent.span(
            id=event_id,
            trace_id=trace_id,
            start_time=start_event.time,
            name=start_event.event_type.value,
            version=self.version,
            session_id=self.session_id,
            input=input,
            output=output,
        )

        if end_event:
            span.end(end_time=end_event.time)

        return span

    def _handle_node_parsing_payload(
        self, events: List[CallbackEvent]
    ) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """Handle the payload of a NODE_PARSING event."""
        inputs = events[0].payload
        outputs = events[-1].payload

        if inputs and EventPayload.DOCUMENTS in inputs:
            documents = inputs.pop(EventPayload.DOCUMENTS)
            inputs["documents"] = [doc.metadata for doc in documents]

        if outputs and EventPayload.NODES in outputs:
            nodes = outputs.pop(EventPayload.NODES)
            outputs["num_nodes"] = len(nodes)

        return inputs, outputs

    def handle_chunking_payload(
        self, events: List[CallbackEvent]
    ) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """Handle the payload of a NODE_PARSING event."""
        inputs = None
        outputs = events[-1].payload

        if outputs and EventPayload.CHUNKS in outputs:
            chunks = outputs.pop(EventPayload.CHUNKS)
            outputs["num_chunks"] = len(chunks)

        return inputs, outputs
