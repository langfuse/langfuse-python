import os
import logging
from abc import ABC
from uuid import UUID
from datetime import datetime
from collections import defaultdict
from contextvars import ContextVar
from contextlib import contextmanager
from typing import Any, Dict, Generator, List, Optional, Union, Tuple, Callable

from llama_index.callbacks.base_handler import BaseCallbackHandler
from llama_index.callbacks.token_counting import get_llm_token_counts
from llama_index.callbacks.schema import (
    CBEvent,
    BASE_TRACE_EVENT,
    LEAF_EVENTS,
    CBEventType,
    EventPayload,
    TIMESTAMP_FORMAT,
)
from llama_index.utils import globals_helper

from langfuse.api.resources.commons.types.observation_level import ObservationLevel
from langfuse.client import Langfuse, StateType, StatefulSpanClient, StatefulTraceClient
from langfuse.model import CreateGeneration, CreateSpan, CreateTrace, UpdateGeneration, UpdateSpan


global_stack_trace = ContextVar("trace", default=[BASE_TRACE_EVENT])
empty_trace_ids: List[str] = []
global_stack_trace_ids = ContextVar("trace_ids", default=empty_trace_ids)


class CallbackHandler(BaseCallbackHandler, ABC):
    def __init__(
        self,
        tokenizer: Optional[Callable[[str], List]] = None,
        public_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        host: str = None,
        debug: bool = False,
        statefulClient: Optional[Union[StatefulTraceClient, StatefulSpanClient]] = None,
        release: Optional[str] = None,
    ) -> None:
        # If we're provided a stateful trace client directly
        prioritized_public_key = public_key if public_key else os.environ.get("LANGFUSE_PUBLIC_KEY")
        prioritized_secret_key = secret_key if secret_key else os.environ.get("LANGFUSE_SECRET_KEY")
        prioritized_host = host if host else os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com")

        if debug:
            # Ensures that debug level messages are logged when debug mode is on.
            # Otherwise, defaults to WARNING level.
            # See https://docs.python.org/3/howto/logging.html#what-happens-if-no-configuration-is-provided

            logging.basicConfig()
            self.log.setLevel(logging.DEBUG)

            self.log.debug("Debug mode is on. Logging debug level messages.")
        else:
            self.log.setLevel(logging.WARNING)

        if statefulClient and isinstance(statefulClient, StatefulTraceClient):
            self.trace = statefulClient
            self.runs = {}
            self.rootSpan = None

        elif statefulClient and isinstance(statefulClient, StatefulSpanClient):
            self.runs = {}
            self.rootSpan = statefulClient
            self.trace = StatefulTraceClient(
                statefulClient.client,
                statefulClient.trace_id,
                StateType.TRACE,
                statefulClient.trace_id,
                statefulClient.task_manager,
            )
            self.runs[statefulClient.id] = statefulClient

        # Otherwise, initialize stateless using the provided keys
        elif prioritized_public_key and prioritized_secret_key:
            self.langfuse = Langfuse(
                public_key=prioritized_public_key,
                secret_key=prioritized_secret_key,
                host=prioritized_host,
                debug=debug,
                release=release,
            )
            self.trace = None
            self.rootSpan = None
            self.runs = {}

        else:
            self.log.error("Either provide a stateful langfuse object or both public_key and secret_key.")
            raise ValueError("Either provide a stateful langfuse object or both public_key and secret_key.")

        self.tokenizer = tokenizer or globals_helper.tokenizer
        self._event_pairs_by_id: Dict[str, List[CBEvent]] = defaultdict(list)
        self._cur_trace_id: Optional[str] = None
        self._trace_map: Dict[str, List[str]] = defaultdict(list)

    def start_trace(self, trace_id: Optional[str] = None) -> None:
        """Run when an overall trace is launched."""

        current_trace_stack_ids = global_stack_trace_ids.get().copy()
        if trace_id is not None:
            if len(current_trace_stack_ids) == 0:
                self._reset_trace_events()

                self._trace_map = defaultdict(list)
                self._cur_trace_id = trace_id
                self._start_time = datetime.now()

                current_trace_stack_ids = [trace_id]
            else:
                current_trace_stack_ids.append(trace_id)

        global_stack_trace_ids.set(current_trace_stack_ids)

    def end_trace(
        self,
        trace_id: Optional[str] = None,
        trace_map: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        """Run when an overall trace is exited."""
        current_trace_stack_ids = global_stack_trace_ids.get().copy()

        if trace_id is not None and len(current_trace_stack_ids) > 0:
            current_trace_stack_ids.pop()
            if len(current_trace_stack_ids) == 0:
                self._trace_map = trace_map or defaultdict(list)
                self._end_time = datetime.now()

                # Log the trace map to wandb
                # We can control what trace ids we want to log here.
                self.log_trace_tree()

                current_trace_stack_ids = []

        global_stack_trace_ids.set(current_trace_stack_ids)

    def on_event_start(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]] = None,
        event_id: Optional[str] = None,
        parent_id: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        """Store event start data by event type.

        Args:
            event_type (CBEventType): event type to store.
            payload (Optional[Dict[str, Any]]): payload to store.
            event_id (str): event id to store.
            parent_id (str): parent event id.

        """

        event_id = event_id or str(UUID())
        if event_type not in handler.event_starts_to_ignore:
            event = CBEvent(event_type, payload=payload, id_=event_id)
            self._event_pairs_by_id[event.id_].append(event)
            return event.id_

        if event_type not in LEAF_EVENTS:
            # copy the stack trace to prevent conflicts with threads/coroutines
            current_trace_stack = global_stack_trace.get().copy()
            current_trace_stack.append(event_id)
            global_stack_trace.set(current_trace_stack)

        return event_id

    def on_event_end(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]] = None,
        event_id: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Store event end data by event type.

        Args:
            event_type (CBEventType): event type to store.
            payload (Optional[Dict[str, Any]]): payload to store.
            event_id (str): event id to store.

        """

        event_id = event_id or str(UUID())
        if event_type not in handler.event_ends_to_ignore:
            event = CBEvent(event_type, payload=payload, id_=event_id)
            self._event_pairs_by_id[event.id_].append(event)
            self._trace_map = defaultdict(list)

        if event_type not in LEAF_EVENTS:
            # copy the stack trace to prevent conflicts with threads/coroutines
            current_trace_stack = global_stack_trace.get().copy()
            current_trace_stack.pop()
            global_stack_trace.set(current_trace_stack)

    def add_handler(self, handler: BaseCallbackHandler) -> None:
        """Add a handler to the callback manager."""
        self.log.debug("Only support Langfuse, no other handler accepted")

    def remove_handler(self, handler: BaseCallbackHandler) -> None:
        """Remove a handler from the callback manager."""
        self.log.debug("Can't remove default handler : Langfuse")

    def set_handlers(self, handlers: List[BaseCallbackHandler]) -> None:
        """Set handlers as the only handlers on the callback manager."""
        self.log.debug("Set to default handler : Langfuse")

    @contextmanager
    def event(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]] = None,
        event_id: Optional[str] = None,
    ) -> Generator["EventContext", None, None]:
        """Context manager for lanching and shutdown of events.

        Handles sending on_evnt_start and on_event_end to handlers for specified event.

        Usage:
            with callback_manager.event(CBEventType.QUERY, payload={key, val}) as event:
                ...
                event.on_end(payload={key, val})  # optional
        """
        event = EventContext(self, event_type, event_id=event_id)
        event.on_start(payload=payload)

        try:
            yield event
        except Exception as e:
            self.on_event_start(CBEventType.EXCEPTION, payload={EventPayload.EXCEPTION: e})
            raise
        finally:
            # ensure event is ended
            if not event.finished:
                event.on_end()

    @contextmanager
    def as_trace(self, trace_id: str) -> Generator[None, None, None]:
        """Context manager tracer for lanching and shutdown of traces."""
        self.start_trace(trace_id=trace_id)

        try:
            yield
        except Exception as e:
            self.on_event_start(CBEventType.EXCEPTION, payload={EventPayload.EXCEPTION: e})
            raise
        finally:
            # ensure trace is ended
            self.end_trace(trace_id=trace_id)

    def _reset_trace_events(self) -> None:
        """Helper function to reset the current trace."""
        self._trace_map = defaultdict(list)
        global_stack_trace.set([BASE_TRACE_EVENT])

    @property
    def trace_map(self) -> Dict[str, List[str]]:
        return self._trace_map

    def _log_trace_tree(self) -> None:
        try:
            child_nodes = self._trace_map["root"]
            root_event_pair = self._event_pairs_by_id[child_nodes[0]]
            trace_id = self._cur_trace_id if len(child_nodes) > 1 else None

            if trace_id is None:
                event_type = root_event_pair[0].event_type
            else:
                event_type = trace_id  # type: ignore

            trace = self.langfuse.trace(
                CreateTrace(
                    name=event_type,
                )
            )

        except Exception as e:
            print(f"Failed to log trace tree to W&B: {e}")

    def _convert_event_pair_to_langfuse_trace_and_span(self, event_pair: List[CBEvent], trace_id: Optional[str] = None):
        start_time_sec, end_time_sec = self._get_time_in_sec(event_pair)

        if trace_id is None:
            event_type = event_pair[0].event_type
            if event_type == CBEventType.QUERY:
                is_span = False
            else:
                is_span = True
        else:
            event_type = trace_id  # type: ignore
            is_span = True

        trace = self.langfuse.trace(
            CreateTrace(
                name=event_type,
                metadata=self.__join_tags_and_metadata(tags, metadata),
            )
        )

        inputs, outputs, metadata = self._get_payload_data(event_pair)

        if is_span:
            span = CreateSpan(
                id=self.nextSpanId,
                name=event_type,
                metadata=metadata,
                input=inputs,
                output=outputs,
                startTime=start_time_sec,
                endTime=end_time_sec,
            )

    def _get_payload_data(self, event_pair: List[CBEvent]) -> None:
        assert len(event_pair) == 2
        event_type = event_pair[0].event_type
        inputs = None
        outputs = None
        metadata = None

        if event_type == CBEventType.NODE_PARSING:
            # TODO: disabled full detailed inputs/outputs due to UI lag
            inputs, outputs = self._handle_node_parsing_payload(event_pair)
        elif event_type == CBEventType.LLM:
            inputs, outputs, metadata = self._handle_llm_payload(event_pair)
        elif event_type == CBEventType.QUERY:
            inputs, outputs = self._handle_query_payload(event_pair)
        elif event_type == CBEventType.EMBEDDING:
            inputs, outputs = self._handle_embedding_payload(event_pair)

        return inputs, outputs, metadata

    def _handle_node_parsing_payload(self, event_pair: List[CBEvent]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Handle the payload of a NODE_PARSING event."""
        inputs = event_pair[0].payload
        outputs = event_pair[-1].payload

        if inputs and EventPayload.DOCUMENTS in inputs:
            documents = inputs.pop(EventPayload.DOCUMENTS)
            inputs["num_documents"] = len(documents)

        if outputs and EventPayload.NODES in outputs:
            nodes = outputs.pop(EventPayload.NODES)
            outputs["num_nodes"] = len(nodes)

        return inputs or {}, outputs or {}

    def _handle_embedding_payload(
        self,
        event_pair: List[CBEvent],
    ) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
        event_pair[0].payload
        outputs = event_pair[-1].payload

        chunks = []
        if outputs:
            chunks = outputs.get(EventPayload.CHUNKS, [])

        return {}, {"num_chunks": len(chunks)}

    def _handle_query_payload(self, event_pair: List[CBEvent]) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
        """Handle the payload of a QUERY event."""

        inputs = event_pair[0].payload
        outputs = event_pair[-1].payload

        if outputs:
            response_obj = outputs[EventPayload.RESPONSE]
            response = str(outputs[EventPayload.RESPONSE])

            if type(response).__name__ == "Response":
                response = response_obj.response
            elif type(response).__name__ == "StreamingResponse":
                response = response_obj.get_response().response
        else:
            response = " "

        outputs = {"response": response}

        return inputs, outputs

    def _handle_llm_payload(self, event_pair: List[CBEvent]) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        """Handle the payload of a LLM event."""
        inputs = event_pair[0].payload
        outputs = event_pair[-1].payload

        assert isinstance(inputs, dict) and isinstance(outputs, dict)

        # Get `original_template` from Prompt
        if EventPayload.PROMPT in inputs:
            inputs[EventPayload.PROMPT] = inputs[EventPayload.PROMPT]

        # Format messages
        if EventPayload.MESSAGES in inputs:
            inputs[EventPayload.MESSAGES] = "\n".join([str(x) for x in inputs[EventPayload.MESSAGES]])

        token_counts = get_llm_token_counts(self.tokenizer, outputs)
        metadata = {
            "prompt_token_count": token_counts.prompt_token_count,
            "completion_token_count": token_counts.completion_token_count,
            "total_tokens_used": token_counts.total_token_count,
        }

        # Make `response` part of `outputs`
        outputs = {EventPayload.RESPONSE: str(outputs[EventPayload.RESPONSE])}

        return inputs, outputs, metadata

    def _get_time_in_sec(self, event_pair: List[CBEvent]) -> Tuple[int, int]:
        """Get the start and end time of an event pair in milliseconds."""

        start_time = datetime.strptime(event_pair[0].time, TIMESTAMP_FORMAT)
        end_time = datetime.strptime(event_pair[1].time, TIMESTAMP_FORMAT)

        start_time_in_sec = int((start_time - datetime(1970, 1, 1)).total_seconds())
        end_time_in_sec = int((end_time - datetime(1970, 1, 1)).total_seconds())

        return start_time_in_sec, end_time_in_sec

    def flush(self):
        if self.trace is not None:
            self.trace.task_manager.flush()
        elif self.rootSpan is not None:
            self.rootSpan.task_manager.flush()
        else:
            self.log.debug("There was no trace yet, hence no flushing possible.")


class EventContext:
    """
    Simple wrapper to call callbacks on event starts and ends
    with an event type and id.
    """

    def __init__(
        self,
        callback_handler: CallbackHandler,
        event_type: CBEventType,
        event_id: Optional[str] = None,
        debug: bool = False,
    ):
        self._callback_handler = callback_handler
        self._event_type = event_type
        self._event_id = event_id or str(UUID())
        self.started = False
        self.finished = False

        if debug:
            # Ensures that debug level messages are logged when debug mode is on.
            # Otherwise, defaults to WARNING level.
            # See https://docs.python.org/3/howto/logging.html#what-happens-if-no-configuration-is-provided

            logging.basicConfig()
            self.log.setLevel(logging.DEBUG)

            self.log.debug("Debug mode is on. Logging debug level messages.")
        else:
            self.log.setLevel(logging.WARNING)

    def on_start(self, payload: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
        if not self.started:
            self.started = True
            self._callback_handler.on_event_start(self._event_type, payload=payload, event_id=self._event_id, **kwargs)
        else:
            self.log.debug(f"Event {self._event_type!s}: {self._event_id} already started!")

    def on_end(self, payload: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
        if not self.finished:
            self.finished = True
            self._callback_handler.on_event_end(self._event_type, payload=payload, event_id=self._event_id, **kwargs)
