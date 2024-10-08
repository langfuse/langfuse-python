import httpx
import uuid
from contextlib import contextmanager
from typing import Optional, Dict, Any, List
from logging import getLogger
from langfuse import Langfuse

from langfuse.client import StatefulTraceClient, StateType
from langfuse.utils.langfuse_singleton import LangfuseSingleton

from ._context import InstrumentorContext
from ._span_handler import LlamaIndexSpanHandler
from ._event_handler import LlamaIndexEventHandler


try:
    from llama_index.core.instrumentation import get_dispatcher
except ImportError:
    raise ModuleNotFoundError(
        "Please install llama-index to use the Langfuse llama-index integration: 'pip install llama-index'"
    )

logger = getLogger(__name__)


class LlamaIndexInstrumentor:
    def __init__(
        self,
        *,
        public_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        host: Optional[str] = None,
        debug: Optional[bool] = None,
        threads: Optional[int] = None,
        flush_at: Optional[int] = None,
        flush_interval: Optional[int] = None,
        max_retries: Optional[int] = None,
        timeout: Optional[int] = None,
        httpx_client: Optional[httpx.Client] = None,
        enabled: Optional[bool] = None,
        sample_rate: Optional[float] = None,
    ):
        self._langfuse = LangfuseSingleton().get(
            public_key=public_key,
            secret_key=secret_key,
            host=host,
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
        self._observation_updates = {}
        self._span_handler = LlamaIndexSpanHandler(
            langfuse_client=self._langfuse,
            observation_updates=self._observation_updates,
        )
        self._event_handler = LlamaIndexEventHandler(
            langfuse_client=self._langfuse,
            observation_updates=self._observation_updates,
        )
        self._context = InstrumentorContext()

    def instrument(self):
        self._context.reset()
        dispatcher = get_dispatcher()

        # Span Handler
        if not any(
            isinstance(handler, type(self._span_handler))
            for handler in dispatcher.span_handlers
        ):
            dispatcher.add_span_handler(self._span_handler)

        # Event Handler
        if not any(
            isinstance(handler, type(self._event_handler))
            for handler in dispatcher.event_handlers
        ):
            dispatcher.add_event_handler(self._event_handler)

    def uninstrument(self):
        self._context.reset()
        dispatcher = get_dispatcher()

        # Span Handler, in-place filter
        dispatcher.span_handlers[:] = filter(
            lambda h: not isinstance(h, type(self._span_handler)),
            dispatcher.span_handlers,
        )

        # Event Handler, in-place filter
        dispatcher.event_handlers[:] = filter(
            lambda h: not isinstance(h, type(self._event_handler)),
            dispatcher.event_handlers,
        )

    @contextmanager
    def observe(
        self,
        *,
        trace_id: Optional[str] = None,
        parent_observation_id: Optional[str] = None,
        update_parent: Optional[bool] = None,
        trace_name: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        version: Optional[str] = None,
        release: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        public: Optional[bool] = None,
    ):
        was_instrumented = self.is_instrumented

        if not was_instrumented:
            self.instrument()

        if parent_observation_id is not None and trace_id is None:
            logger.warning(
                "trace_id must be provided if parent_observation_id is provided. Ignoring parent_observation_id."
            )
            parent_observation_id = None

        final_trace_id = trace_id or str(uuid.uuid4())

        self._context.update(
            is_user_managed_trace=True,
            trace_id=final_trace_id,
            parent_observation_id=parent_observation_id,
            update_parent=update_parent,
            trace_name=trace_name,
            user_id=user_id,
            session_id=session_id,
            version=version,
            release=release,
            metadata=metadata,
            tags=tags,
            public=public,
        )

        yield self._get_trace_client(final_trace_id)

        self._context.reset()

        if not was_instrumented:
            self.uninstrument()

    @property
    def is_instrumented(self) -> bool:
        """Check if the dispatcher is instrumented."""
        dispatcher = get_dispatcher()

        return any(
            isinstance(handler, type(self._span_handler))
            for handler in dispatcher.span_handlers
        ) and any(
            isinstance(handler, type(self._event_handler))
            for handler in dispatcher.event_handlers
        )

    def _get_trace_client(self, trace_id: str) -> StatefulTraceClient:
        return StatefulTraceClient(
            client=self._langfuse.client,
            id=trace_id,
            trace_id=trace_id,
            task_manager=self._langfuse.task_manager,
            state_type=StateType.TRACE,
        )

    @property
    def client_instance(self) -> Langfuse:
        """Get the Langfuse client instance."""
        return self._langfuse

    def flush(self) -> None:
        """Flushes the Langfuse client."""
        self.client_instance.flush()
