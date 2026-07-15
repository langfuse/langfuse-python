"""Tracer implementation for Langfuse OpenTelemetry integration.

This module provides the LangfuseTracer class, a thread-safe singleton that manages OpenTelemetry
tracing infrastructure for Langfuse. It handles tracer initialization, span processors,
API clients, and coordinates background tasks for efficient data processing and media handling.

Key features:
- Thread-safe OpenTelemetry tracer with Langfuse-specific span processors and sampling
- Configurable batch processing of spans and scores with intelligent flushing behavior
- Asynchronous background media upload processing with dedicated worker threads
- Concurrent score ingestion with batching and retry mechanisms
- Automatic project ID discovery and caching
- Graceful shutdown handling with proper resource cleanup
- Fault tolerance with detailed error logging and recovery mechanisms
"""

import atexit
import os
import sys
import threading
import urllib.request
import weakref
from queue import Full, Queue
from typing import Any, Callable, Dict, List, Optional, cast

import httpx
from opentelemetry import trace as otel_trace_api
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SpanExporter
from opentelemetry.sdk.trace.id_generator import IdGenerator
from opentelemetry.sdk.trace.sampling import Decision, TraceIdRatioBased
from opentelemetry.trace import Tracer

from langfuse._client.attributes import LangfuseOtelSpanAttributes
from langfuse._client.constants import LANGFUSE_TRACER_NAME
from langfuse._client.environment_variables import (
    LANGFUSE_MEDIA_UPLOAD_ENABLED,
    LANGFUSE_MEDIA_UPLOAD_THREAD_COUNT,
    LANGFUSE_RELEASE,
    LANGFUSE_TRACING_ENVIRONMENT,
)
from langfuse._client.span_processor import LangfuseSpanProcessor
from langfuse._task_manager.media_manager import MediaManager
from langfuse._task_manager.media_upload_consumer import MediaUploadConsumer
from langfuse._task_manager.score_ingestion_consumer import ScoreIngestionConsumer
from langfuse._utils.environment import get_common_release_envs
from langfuse._utils.prompt_cache import PromptCache
from langfuse._utils.request import LangfuseClient
from langfuse.api import AsyncLangfuseAPI, LangfuseAPI
from langfuse.logger import langfuse_logger
from langfuse.types import MaskFunction, MaskOtelSpansFunction

from .._version import __version__ as langfuse_version


class LangfuseResourceManager:
    """Thread-safe singleton that provides access to the OpenTelemetry tracer and processors.

    This class implements a thread-safe singleton pattern keyed by the public API key,
    ensuring that only one tracer instance exists per API key combination. It manages
    the lifecycle of the OpenTelemetry tracer provider, span processors, and resource
    attributes, as well as background threads for media uploads and score ingestion.

    The tracer is responsible for:
    1. Setting up the OpenTelemetry tracer with appropriate sampling and configuration
    2. Managing the span processor for exporting spans to the Langfuse API
    3. Creating and managing Langfuse API clients (both synchronous and asynchronous)
    4. Handling background media upload processing via dedicated worker threads
    5. Processing and batching score ingestion events with configurable flush settings
    6. Retrieving and caching project information for URL generation and media handling
    7. Coordinating graceful shutdown of all background processes with proper resource cleanup

    This implementation follows best practices for resource management in long-running
    applications, including thread-safe singleton pattern, bounded queues to prevent memory
    exhaustion, proper resource cleanup on shutdown, and fault-tolerant error handling with
    detailed logging.

    Thread safety is ensured through the use of locks, thread-safe queues, and atomic operations,
    making this implementation suitable for multi-threaded and asyncio applications.
    """

    _instances: Dict[str, "LangfuseResourceManager"] = {}
    _lock = threading.RLock()
    _otel_tracer: Tracer
    _media_manager: MediaManager
    _media_upload_consumers: List[MediaUploadConsumer]
    _ingestion_consumers: List[ScoreIngestionConsumer]

    @classmethod
    def get_singleton_httpx_client(cls) -> Optional[httpx.Client]:
        with cls._lock:
            instances = list(cls._instances.values())

            if not instances:
                return None

            if len(instances) > 1:
                # Mirror get_client's safety stance: with multiple clients we
                # cannot tell which one produced a given reference, so fall back
                # to a default httpx client rather than silently using an
                # arbitrary instance's transport config (proxy / CA / mTLS).
                langfuse_logger.warning(
                    "Multiple Langfuse clients are instantiated; falling back to a "
                    "default httpx client for LangfuseMediaReference fetches. Pass an "
                    "explicit `client` to fetch_bytes/fetch_base64/fetch_data_uri to "
                    "honor per-client transport settings."
                )
                return None

            return instances[0].httpx_client

    def __new__(
        cls,
        *,
        public_key: str,
        secret_key: str,
        base_url: str,
        environment: Optional[str] = None,
        release: Optional[str] = None,
        timeout: Optional[int] = None,
        flush_at: Optional[int] = None,
        flush_interval: Optional[float] = None,
        httpx_client: Optional[httpx.Client] = None,
        media_upload_thread_count: Optional[int] = None,
        sample_rate: Optional[float] = None,
        mask: Optional[MaskFunction] = None,
        mask_otel_spans: Optional[MaskOtelSpansFunction] = None,
        tracing_enabled: Optional[bool] = None,
        blocked_instrumentation_scopes: Optional[List[str]] = None,
        should_export_span: Optional[Callable[[ReadableSpan], bool]] = None,
        additional_headers: Optional[Dict[str, str]] = None,
        tracer_provider: Optional[TracerProvider] = None,
        id_generator: Optional[IdGenerator] = None,
        span_exporter: Optional[SpanExporter] = None,
    ) -> "LangfuseResourceManager":
        if public_key in cls._instances:
            return cls._instances[public_key]

        with cls._lock:
            if public_key not in cls._instances:
                instance = super(LangfuseResourceManager, cls).__new__(cls)

                # Initialize tracer (will be noop until init instance)
                instance._otel_tracer = otel_trace_api.get_tracer(
                    LANGFUSE_TRACER_NAME,
                    langfuse_version,
                    attributes={"public_key": public_key},
                )

                instance._initialize_instance(
                    public_key=public_key,
                    secret_key=secret_key,
                    base_url=base_url,
                    timeout=timeout,
                    environment=environment,
                    release=release,
                    flush_at=flush_at,
                    flush_interval=flush_interval,
                    httpx_client=httpx_client,
                    media_upload_thread_count=media_upload_thread_count,
                    sample_rate=sample_rate,
                    mask=mask,
                    mask_otel_spans=mask_otel_spans,
                    tracing_enabled=tracing_enabled
                    if tracing_enabled is not None
                    else True,
                    blocked_instrumentation_scopes=blocked_instrumentation_scopes,
                    should_export_span=should_export_span,
                    additional_headers=additional_headers,
                    tracer_provider=tracer_provider,
                    id_generator=id_generator,
                    span_exporter=span_exporter,
                )

                cls._instances[public_key] = instance

            return cls._instances[public_key]

    def _initialize_instance(
        self,
        *,
        public_key: str,
        secret_key: str,
        base_url: str,
        environment: Optional[str] = None,
        release: Optional[str] = None,
        timeout: Optional[int] = None,
        flush_at: Optional[int] = None,
        flush_interval: Optional[float] = None,
        media_upload_thread_count: Optional[int] = None,
        httpx_client: Optional[httpx.Client] = None,
        sample_rate: Optional[float] = None,
        mask: Optional[MaskFunction] = None,
        mask_otel_spans: Optional[MaskOtelSpansFunction] = None,
        tracing_enabled: bool = True,
        blocked_instrumentation_scopes: Optional[List[str]] = None,
        should_export_span: Optional[Callable[[ReadableSpan], bool]] = None,
        additional_headers: Optional[Dict[str, str]] = None,
        tracer_provider: Optional[TracerProvider] = None,
        id_generator: Optional[IdGenerator] = None,
        span_exporter: Optional[SpanExporter] = None,
    ) -> None:
        self.public_key = public_key
        self.secret_key = secret_key
        self.tracing_enabled = tracing_enabled
        self.base_url = base_url
        self.mask = mask
        self.mask_otel_spans = mask_otel_spans
        self.environment = environment
        self._shutdown = False

        # Store additional client settings for get_client() to use
        self.timeout = timeout
        self.flush_at = flush_at
        self.flush_interval = flush_interval
        self.release = release
        self.media_upload_thread_count = media_upload_thread_count
        self.sample_rate = sample_rate
        self.blocked_instrumentation_scopes = blocked_instrumentation_scopes
        self.should_export_span = should_export_span
        self.additional_headers = additional_headers
        self.id_generator = id_generator
        self.span_exporter = span_exporter
        self.tracer_provider: Optional[TracerProvider] = None

        self._custom_httpx_client = httpx_client
        self._init_api_clients()

        # Media
        self._media_upload_enabled = os.environ.get(
            LANGFUSE_MEDIA_UPLOAD_ENABLED, "True"
        ).lower() not in ("false", "0")

        self._media_upload_thread_count = media_upload_thread_count or max(
            int(os.getenv(LANGFUSE_MEDIA_UPLOAD_THREAD_COUNT, 1)), 1
        )

        self._init_media_manager()

        # OTEL Tracer
        if tracing_enabled:
            tracer_provider = tracer_provider or _init_tracer_provider(
                environment=environment,
                release=release,
                sample_rate=sample_rate,
                id_generator=id_generator,
            )
            self.tracer_provider = tracer_provider

            langfuse_processor = LangfuseSpanProcessor(
                public_key=self.public_key,
                secret_key=secret_key,
                base_url=base_url,
                timeout=timeout,
                flush_at=flush_at,
                flush_interval=flush_interval,
                blocked_instrumentation_scopes=blocked_instrumentation_scopes,
                should_export_span=should_export_span,
                additional_headers=additional_headers,
                span_exporter=span_exporter,
                media_manager=self._media_manager,
                mask_otel_spans=mask_otel_spans,
            )
            tracer_provider.add_span_processor(langfuse_processor)

            self._otel_tracer = tracer_provider.get_tracer(
                LANGFUSE_TRACER_NAME,
                langfuse_version,
                attributes={"public_key": self.public_key},
            )

        self._init_consumer_threads()

        # Prompt cache
        self.prompt_cache = PromptCache()

        # Register shutdown handler
        atexit.register(self.shutdown)

        # Register fork handler to reinitialize consumer threads in child process.
        # When using Gunicorn with --preload, os.fork() copies memory but not threads
        # (POSIX.1: https://pubs.opengroup.org/onlinepubs/9699919799/functions/fork.html).
        # Without this, media upload and score ingestion threads are lost after fork,
        # causing silent data loss.
        #
        # Note: LangfuseSpanProcessor (BatchSpanProcessor) already handles fork-safety
        # for span export via its own os.register_at_fork. This handler covers the
        # remaining background threads managed by LangfuseResourceManager.
        #
        # weakref.WeakMethod prevents os.register_at_fork from holding a permanent strong
        # reference to this instance, which would block garbage collection.
        # See: https://github.com/open-telemetry/opentelemetry-python/blob/main/opentelemetry-sdk/src/opentelemetry/sdk/_shared_internal/__init__.py
        if hasattr(os, "register_at_fork"):
            weak_reinit = weakref.WeakMethod(self._at_fork_reinit)
            os.register_at_fork(
                # Walrus operator resolves the weak reference once and stores it in
                # a temporary variable before calling it. This avoids a TOCTOU window
                # where GC could collect the referent between checking for None and
                # invoking the method.
                after_in_child=lambda: (m := weak_reinit()) and m()
            )

        langfuse_logger.info(
            f"Startup: Langfuse tracer successfully initialized | "
            f"public_key={self.public_key} | "
            f"base_url={base_url} | "
            f"environment={environment or 'default'} | "
            f"sample_rate={sample_rate if sample_rate is not None else 1.0} | "
            f"media_threads={self._media_upload_thread_count}"
        )

    def _init_media_manager(self) -> None:
        """Initialize or reset media upload state while preserving manager references."""
        self._media_upload_queue: Queue[Any] = Queue(100_000)
        if hasattr(self, "_media_manager"):
            self._media_manager.reinitialize(
                api_client=self.api,
                httpx_client=self.httpx_client,
                media_upload_queue=self._media_upload_queue,
            )
        else:
            self._media_manager = MediaManager(
                api_client=self.api,
                httpx_client=self.httpx_client,
                media_upload_queue=self._media_upload_queue,
                max_retries=3,
            )

        self._media_upload_consumers = []

    def _init_api_clients(self) -> None:
        """Initialize HTTP-backed API clients.

        Internally-managed httpx clients are recreated when this method is
        called after fork. Caller-provided clients are preserved because their
        lifecycle belongs to the caller.
        """
        if self._custom_httpx_client is not None:
            self.httpx_client = self._custom_httpx_client
        else:
            client_headers = self.additional_headers if self.additional_headers else {}
            self.httpx_client = httpx.Client(
                timeout=self.timeout, headers=client_headers
            )

        self.api = LangfuseAPI(
            base_url=self.base_url,
            username=self.public_key,
            password=self.secret_key,
            x_langfuse_sdk_name="python",
            x_langfuse_sdk_version=langfuse_version,
            x_langfuse_public_key=self.public_key,
            httpx_client=self.httpx_client,
            timeout=self.timeout,
        )
        self.async_api = AsyncLangfuseAPI(
            base_url=self.base_url,
            username=self.public_key,
            password=self.secret_key,
            x_langfuse_sdk_name="python",
            x_langfuse_sdk_version=langfuse_version,
            x_langfuse_public_key=self.public_key,
            timeout=self.timeout,
        )
        self._score_ingestion_client = LangfuseClient(
            public_key=self.public_key,
            secret_key=self.secret_key,
            base_url=self.base_url,
            version=langfuse_version,
            timeout=self.timeout or 20,
            session=self.httpx_client,
        )

    def _init_consumer_threads(self) -> None:
        """Initialize media upload and score ingestion consumer threads."""
        if self._media_upload_enabled:
            for i in range(self._media_upload_thread_count):
                media_upload_consumer = MediaUploadConsumer(
                    identifier=i,
                    media_manager=self._media_manager,
                )
                media_upload_consumer.start()
                self._media_upload_consumers.append(media_upload_consumer)

        # Score ingestion
        self._score_ingestion_queue: Queue[Any] = Queue(100_000)
        self._ingestion_consumers = []

        ingestion_consumer = ScoreIngestionConsumer(
            ingestion_queue=self._score_ingestion_queue,
            identifier=0,
            client=self._score_ingestion_client,
            flush_at=self.flush_at,
            flush_interval=self.flush_interval,
            max_retries=3,
            public_key=self.public_key,
        )
        ingestion_consumer.start()
        self._ingestion_consumers.append(ingestion_consumer)

    def _at_fork_reinit(self) -> None:
        """Reinitialize consumer threads after fork in child process.

        Called automatically via os.register_at_fork() after fork().
        Necessary for Gunicorn --preload deployments where os.fork() is used:
        threads are not copied to child processes (POSIX standard), so without
        reinitialization, the child process has no consumer threads and all
        media upload and score ingestion events are silently lost.

        Note: LangfuseSpanProcessor (BatchSpanProcessor) handles span export
        fork-safety separately via its own os.register_at_fork handler.

        Skipped if shutdown() was already called on this instance, to avoid
        restarting threads on an intentionally torn-down manager.
        """
        # The class-level lock may have been held by a thread in the parent at fork time.
        # That thread does not exist in the child, so the lock can never be released and
        # any attempt to acquire it would deadlock. Replace it before the shutdown check:
        # the lock is class-level state needed by the child (e.g. to create a new client)
        # even if this particular instance was already shut down.
        LangfuseResourceManager._lock = threading.RLock()

        if self._shutdown:
            return

        if sys.platform == "darwin" and not urllib.request.getproxies_environment():
            # urllib proxy discovery falls back to macOS SystemConfiguration APIs that
            # are not safe to invoke after fork(). Setting no_proxy="*" makes httpx and
            # requests skip that lookup entirely in this child process. Skipped when
            # proxies are configured via environment variables: urllib then never touches
            # SystemConfiguration (no segfault risk), and overriding no_proxy would
            # disable the user's proxy setup process-wide.
            os.environ["no_proxy"] = "*"
            os.environ["NO_PROXY"] = "*"

        langfuse_logger.debug(
            f"[PID {os.getpid()}] Fork detected: reinitializing Langfuse consumer threads."
        )

        # Queues are intentionally recreated after fork. Items enqueued before fork
        # belong to the preloaded parent process and must not be processed by every
        # worker — otherwise uploads/scores would be duplicated across workers.
        #
        # Internally-managed httpx clients must also be recreated: fork() duplicates the
        # parent's connection pool (TCP socket file descriptors) into the child. Both
        # processes then share the same underlying sockets, causing data corruption and
        # SSL/TLS state mismatch under concurrent use. Fresh clients start with an empty
        # pool owned solely by this child process.
        #
        # Custom httpx clients provided by the caller are NOT recreated. The fork-inherited
        # copy is reused as-is, giving the caller the opportunity to handle process-safety
        # themselves (e.g. by registering their own os.register_at_fork handler).
        try:
            self._init_api_clients()
        except Exception as e:
            langfuse_logger.error(
                f"[PID {os.getpid()}] Failed to recreate HTTP clients after fork: {e}. "
                f"Network requests may fail in this worker."
            )

        try:
            self._init_media_manager()
            self._init_consumer_threads()
            self.prompt_cache = PromptCache()
        except Exception as e:
            langfuse_logger.error(
                f"[PID {os.getpid()}] Failed to reinitialize consumer threads after fork: {e}. "
                f"Media upload, score ingestion, and prompt cache refresh will be unavailable in this worker."
            )

        langfuse_logger.debug(
            f"[PID {os.getpid()}] Langfuse consumer threads and prompt cache reinitialized after fork"
        )

    @classmethod
    def reset(cls) -> None:
        with cls._lock:
            for key in cls._instances:
                cls._instances[key].shutdown()

            cls._instances.clear()

    def add_score_task(self, event: dict, *, force_sample: bool = False) -> None:
        try:
            # Sample scores with the same sampler that is used for tracing
            tracer_provider = cast(TracerProvider, otel_trace_api.get_tracer_provider())
            should_sample = (
                force_sample
                or isinstance(
                    tracer_provider, otel_trace_api.ProxyTracerProvider
                )  # default to in-sample if otel sampler is not available
                or (
                    (
                        tracer_provider.sampler.should_sample(
                            parent_context=None,
                            trace_id=int(event["body"].trace_id, 16),
                            name="score",
                        ).decision
                        == Decision.RECORD_AND_SAMPLE
                        if hasattr(event["body"], "trace_id")
                        else True
                    )
                    if event["body"].trace_id
                    is not None  # do not sample out session / dataset run scores
                    else True
                )
            )

            if should_sample:
                langfuse_logger.debug(
                    f"Score: Enqueuing event type={event['type']} for trace_id={event['body'].trace_id} name={event['body'].name} value={event['body'].value}"
                )
                self._score_ingestion_queue.put(event, block=False)

        except Full:
            langfuse_logger.warning(
                "System overload: Score ingestion queue has reached capacity (100,000 items). Score will be dropped. Consider increasing flush frequency or decreasing event volume."
            )

            return
        except Exception as e:
            langfuse_logger.error(
                f"Unexpected error: Failed to process score event. The score will be dropped. Error details: {e}"
            )

            return

    def add_trace_task(
        self,
        event: dict,
    ) -> None:
        try:
            langfuse_logger.debug(
                f"Trace: Enqueuing event type={event['type']} for trace_id={event['body'].id}"
            )
            self._score_ingestion_queue.put(event, block=False)

        except Full:
            langfuse_logger.warning(
                "System overload: Trace ingestion queue has reached capacity (100,000 items). Trace update will be dropped. Consider increasing flush frequency or decreasing event volume."
            )

            return
        except Exception as e:
            langfuse_logger.error(
                f"Unexpected error: Failed to process trace event. The trace update will be dropped. Error details: {e}"
            )

            return

    @property
    def tracer(self) -> Optional[Tracer]:
        return self._otel_tracer

    @staticmethod
    def get_current_span() -> Any:
        return otel_trace_api.get_current_span()

    def _stop_and_join_consumer_threads(self) -> None:
        """End the consumer threads once the queue is empty.

        Blocks execution until finished
        """
        langfuse_logger.debug(
            f"Shutdown: Waiting for {len(self._media_upload_consumers)} media upload thread(s) to complete processing"
        )
        for media_upload_consumer in self._media_upload_consumers:
            media_upload_consumer.pause()

        self._media_manager.signal_shutdown(count=len(self._media_upload_consumers))

        for media_upload_consumer in self._media_upload_consumers:
            try:
                media_upload_consumer.join()
            except RuntimeError:
                # consumer thread has not started
                pass

            langfuse_logger.debug(
                f"Shutdown: Media upload thread #{media_upload_consumer._identifier} successfully terminated"
            )

        langfuse_logger.debug(
            f"Shutdown: Waiting for {len(self._ingestion_consumers)} score ingestion thread(s) to complete processing"
        )
        for score_ingestion_consumer in self._ingestion_consumers:
            score_ingestion_consumer.pause()

        for score_ingestion_consumer in self._ingestion_consumers:
            try:
                score_ingestion_consumer.join()
            except RuntimeError:
                # consumer thread has not started
                pass

            langfuse_logger.debug(
                f"Shutdown: Score ingestion thread #{score_ingestion_consumer._identifier} successfully terminated"
            )

    def flush(self) -> None:
        if self.tracer_provider is not None and not isinstance(
            self.tracer_provider, otel_trace_api.ProxyTracerProvider
        ):
            self.tracer_provider.force_flush()
            langfuse_logger.debug("Successfully flushed OTEL tracer provider")

        self._score_ingestion_queue.join()
        langfuse_logger.debug("Successfully flushed score ingestion queue")

        self._media_upload_queue.join()
        langfuse_logger.debug("Successfully flushed media upload queue")

    def shutdown(self) -> None:
        self._shutdown = True

        # Unregister the atexit handler first
        atexit.unregister(self.shutdown)

        self.flush()
        self._stop_and_join_consumer_threads()


def _init_tracer_provider(
    *,
    environment: Optional[str] = None,
    release: Optional[str] = None,
    sample_rate: Optional[float] = None,
    id_generator: Optional[IdGenerator] = None,
) -> TracerProvider:
    environment = environment or os.environ.get(LANGFUSE_TRACING_ENVIRONMENT)
    release = release or os.environ.get(LANGFUSE_RELEASE) or get_common_release_envs()

    resource_attributes = {
        LangfuseOtelSpanAttributes.ENVIRONMENT: environment,
        LangfuseOtelSpanAttributes.RELEASE: release,
    }

    resource = Resource.create(
        {k: v for k, v in resource_attributes.items() if v is not None}
    )

    provider = None
    default_provider = cast(TracerProvider, otel_trace_api.get_tracer_provider())

    if isinstance(default_provider, otel_trace_api.ProxyTracerProvider):
        provider = TracerProvider(
            resource=resource,
            sampler=TraceIdRatioBased(sample_rate)
            if sample_rate is not None and sample_rate < 1
            else None,
            id_generator=id_generator,
        )
        otel_trace_api.set_tracer_provider(provider)

    else:
        if id_generator is not None:
            langfuse_logger.warning(
                "Configuration: id_generator was ignored because an OpenTelemetry TracerProvider is already registered. "
                "Pass a TracerProvider configured with the desired id_generator to Langfuse(tracer_provider=...) instead."
            )

        provider = default_provider

    return provider
