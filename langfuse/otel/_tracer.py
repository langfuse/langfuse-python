import atexit
import os
import threading
from queue import Queue
from typing import Dict, Optional, cast

import httpx
from opentelemetry import trace as otel_trace_api
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider

from langfuse._task_manager.media_manager import MediaManager
from langfuse._task_manager.media_upload_consumer import MediaUploadConsumer
from langfuse.api.client import AsyncFernLangfuse, FernLangfuse
from langfuse.environment import get_common_release_envs
from langfuse.otel._span_processor import LangfuseSpanProcessor
from langfuse.otel.attributes import LangfuseSpanAttributes
from langfuse.otel.constants import LANGFUSE_TRACER_NAME
from langfuse.otel.environment_variables import (
    LANGFUSE_MEDIA_UPLOAD_THREAD_COUNT,
    LANGFUSE_RELEASE,
    LANGFUSE_TRACING_ENVIRONMENT,
)

from ..version import __version__ as langfuse_version
from ._logger import langfuse_logger


class LangfuseTracer:
    """Singleton that provides access to the OTEL tracer."""

    _instances: Dict[str, "LangfuseTracer"] = {}
    _lock = threading.Lock()

    def __new__(
        cls,
        *,
        public_key: str,
        secret_key: str,
        host: str,
        environment: Optional[str] = None,
        release: Optional[str] = None,
        timeout: Optional[int] = None,
        flush_at: Optional[int] = None,
        flush_interval: Optional[float] = None,
        httpx_client: Optional[httpx.Client] = None,
        media_upload_thread_count: Optional[int] = None,
    ) -> "LangfuseTracer":
        if public_key in cls._instances:
            return cls._instances[public_key]

        with cls._lock:
            if public_key not in cls._instances:
                instance = super(LangfuseTracer, cls).__new__(cls)
                instance._otel_tracer = None
                instance._initialize_instance(
                    public_key=public_key,
                    secret_key=secret_key,
                    host=host,
                    timeout=timeout,
                    environment=environment,
                    release=release,
                    flush_at=flush_at,
                    flush_interval=flush_interval,
                    httpx_client=httpx_client,
                    media_upload_thread_count=media_upload_thread_count,
                )

                cls._instances[public_key] = instance

            return cls._instances[public_key]

    def _initialize_instance(
        self,
        *,
        public_key: str,
        secret_key: str,
        host: str,
        environment: Optional[str] = None,
        release: Optional[str] = None,
        timeout: Optional[int] = None,
        flush_at: Optional[int] = None,
        flush_interval: Optional[float] = None,
        media_upload_thread_count: Optional[int] = None,
        httpx_client: Optional[httpx.Client] = None,
    ):
        # OTEL Tracer
        tracer_provider = _init_tracer_provider(
            environment=environment, release=release
        )

        langfuse_processor = LangfuseSpanProcessor(
            public_key=public_key,
            secret_key=secret_key,
            host=host,
            timeout=timeout,
            flush_at=flush_at,
            flush_interval=flush_interval,
        )
        tracer_provider.add_span_processor(langfuse_processor)

        tracer_provider = otel_trace_api.get_tracer_provider()
        self.name = f"{LANGFUSE_TRACER_NAME}:{public_key}"
        self._otel_tracer = tracer_provider.get_tracer(self.name, langfuse_version)

        # API Clients

        ## API clients must be singletons because the underlying HTTPX clients
        ## use connection pools with limited capacity. Creating multiple instances
        ## could exhaust the OS's maximum number of available TCP sockets (file descriptors),
        ## leading to connection errors.
        self.httpx_client = httpx_client or httpx.Client(timeout=timeout)
        self.api = FernLangfuse(
            base_url=host,
            username=public_key,
            password=secret_key,
            x_langfuse_sdk_name="python",
            x_langfuse_sdk_version=langfuse_version,
            x_langfuse_public_key=public_key,
            httpx_client=self.httpx_client,
            timeout=timeout,
        )
        self.async_api = AsyncFernLangfuse(
            base_url=host,
            username=public_key,
            password=secret_key,
            x_langfuse_sdk_name="python",
            x_langfuse_sdk_version=langfuse_version,
            x_langfuse_public_key=public_key,
            timeout=timeout,
        )

        # Media
        self._media_upload_queue = Queue(100_000)
        self._media_manager = MediaManager(
            api_client=self.api,
            media_upload_queue=self._media_upload_queue,
            max_retries=3,
        )
        self._media_upload_consumers = []

        media_upload_thread_count = media_upload_thread_count or max(
            int(os.getenv(LANGFUSE_MEDIA_UPLOAD_THREAD_COUNT, 1)), 1
        )

        for i in range(media_upload_thread_count):
            media_upload_consumer = MediaUploadConsumer(
                identifier=i,
                media_manager=self._media_manager,
            )
            media_upload_consumer.start()
            self._media_upload_consumers.append(media_upload_consumer)

        # Project ID handling
        self._project_id = None
        self._project_id_fetched = threading.Event()
        self._fetch_project_id_thread = threading.Thread(
            target=self._fetch_project_id_background,
            name="langfuse-project-id-fetcher",
            daemon=True,
        )
        self._fetch_project_id_thread.start()

        # Register shutdown handler
        atexit.register(self.shutdown)

    def _fetch_project_id_background(self):
        try:
            projects = self.api.projects.get(
                request_options={"max_retries": 3, "timeout_in_seconds": 5}
            )
            self._project_id = projects.data[0].id if projects.data else None

            langfuse_logger.debug(
                f"Successfully fetched project ID: {self._project_id}"
            )
        except Exception as e:
            langfuse_logger.warning(f"Failed to fetch project ID: {str(e)}")

        finally:
            self._project_id_fetched.set()

    @property
    def project_id(self):
        if self._project_id:
            return self._project_id

        if self._project_id_fetched.is_set():
            langfuse_logger.warning(
                "Failed to fetch project ID. This may affect features like media uploads and project-specific configurations."
            )
            return None

        fetch_completed = self._project_id_fetched.wait(0.5)

        if not self._project_id:
            langfuse_logger.warning(
                "Failed to fetch project ID. This may affect features like media uploads and project-specific configurations."
                if fetch_completed
                else "Project ID not available as it is currently being fetched.This may affect features like media uploads and project-specific configurations."
            )

        return self._project_id

    @property
    def tracer(self):
        return self._otel_tracer

    @staticmethod
    def get_current_span():
        return otel_trace_api.get_current_span()

    def _join_consumer_threads(self):
        """End the consumer threads once the queue is empty.

        Blocks execution until finished
        """
        langfuse_logger.debug(
            f"joining {len(self._media_upload_consumers)} media upload consumer threads"
        )
        for media_upload_consumer in self._media_upload_consumers:
            media_upload_consumer.pause()

        for media_upload_consumer in self._media_upload_consumers:
            try:
                media_upload_consumer.join()
            except RuntimeError:
                # consumer thread has not started
                pass

            langfuse_logger.debug(
                f"MediaUploadConsumer thread {media_upload_consumer._identifier} joined"
            )

    def flush(self):
        tracer_provider = cast(TracerProvider, otel_trace_api.get_tracer_provider())
        if isinstance(tracer_provider, otel_trace_api.ProxyTracerProvider):
            return

        tracer_provider.force_flush()

    def shutdown(self):
        # Unregister the atexit handler first
        atexit.unregister(self.shutdown)

        tracer_provider = cast(TracerProvider, otel_trace_api.get_tracer_provider())
        if isinstance(tracer_provider, otel_trace_api.ProxyTracerProvider):
            return

        tracer_provider.force_flush()

        self._join_consumer_threads()


def _init_tracer_provider(
    *,
    environment: Optional[str] = None,
    release: Optional[str] = None,
) -> TracerProvider:
    environment = environment or os.environ.get(LANGFUSE_TRACING_ENVIRONMENT)
    release = release or os.environ.get(LANGFUSE_RELEASE) or get_common_release_envs()

    resource_attributes = {
        LangfuseSpanAttributes.ENVIRONMENT: environment,
        LangfuseSpanAttributes.RELEASE: release,
    }

    resource = Resource.create(
        {k: v for k, v in resource_attributes.items() if v is not None}
    )

    provider = None
    default_provider = cast(TracerProvider, otel_trace_api.get_tracer_provider())

    if isinstance(default_provider, otel_trace_api.ProxyTracerProvider):
        provider = TracerProvider(resource=resource)
        otel_trace_api.set_tracer_provider(provider)

    else:
        provider = default_provider

    return provider
