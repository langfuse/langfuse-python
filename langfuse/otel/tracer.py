import os
import threading
from typing import Dict, Optional, cast

from opentelemetry import trace as otel_trace_api
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider

from langfuse.environment import get_common_release_envs
from langfuse.otel._span_processor import LangfuseSpanProcessor
from langfuse.otel.attributes import LangfuseSpanAttributes
from langfuse.otel.constants import LANGFUSE_TRACER_NAME
from langfuse.otel.environment_variables import (
    LANGFUSE_RELEASE,
    LANGFUSE_TRACING_ENVIRONMENT,
)

from ..version import __version__ as langfuse_version


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
    ):
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

    @property
    def tracer(self):
        return self._otel_tracer

    @staticmethod
    def get_current_span():
        return otel_trace_api.get_current_span()


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
