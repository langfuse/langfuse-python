import base64
import os
import threading
from typing import Optional, cast

from opentelemetry import trace as otel_trace_api
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
    SimpleSpanProcessor,
)

from langfuse.environment import get_common_release_envs
from langfuse.otel._utils import span_formatter
from langfuse.otel.attributes import LangfuseSpanAttributes
from langfuse.otel.constants import LANGFUSE_TRACER_NAME
from langfuse.otel.environment_variables import (
    LANGFUSE_RELEASE,
    LANGFUSE_TRACING_ENVIRONMENT,
)

from ..version import __version__ as langfuse_version


class LangfuseTracer:
    """Singleton that provides access to the OTEL tracer."""

    _instance: Optional["LangfuseTracer"] = None
    _lock = threading.Lock()

    def __new__(
        cls,
        *,
        public_key: str,
        secret_key: str,
        host: str,
        timeout: Optional[int] = None,
        environment: Optional[str] = None,
        release: Optional[str] = None,
        debug: Optional[bool] = False,
    ):
        if cls._instance:
            return cls._instance

        with cls._lock:
            if not cls._instance:
                cls._instance = super(LangfuseTracer, cls).__new__(cls)

                cls._instance._otel_tracer = None
                cls._instance._initialize(
                    public_key=public_key,
                    secret_key=secret_key,
                    host=host,
                    timeout=timeout,
                    environment=environment,
                    release=release,
                    debug=debug,
                )

            return cls._instance

    def _initialize(
        self,
        *,
        public_key: str,
        secret_key: str,
        host: str,
        environment: Optional[str] = None,
        release: Optional[str] = None,
        timeout: Optional[int] = None,
        debug: Optional[bool] = False,
    ):
        tracer_provider = _init_tracer_provider(
            environment=environment, release=release
        )

        if debug:
            console_span_exporter = ConsoleSpanExporter(formatter=span_formatter)
            console_span_processor = SimpleSpanProcessor(
                span_exporter=console_span_exporter
            )
            tracer_provider.add_span_processor(console_span_processor)

        langfuse_exporter = OTLPSpanExporter(
            endpoint=f"{host}/api/public/otel/v1/traces",
            headers={
                "Authorization": "Basic "
                + base64.b64encode(f"{public_key}:{secret_key}".encode("utf-8")).decode(
                    "ascii"
                ),
                "x_langfuse_sdk_name": "python",
                "x_langfuse_sdk_version": langfuse_version,
                "x_langfuse_public_key": public_key,
            },
            timeout=timeout,
        )
        langfuse_processor = BatchSpanProcessor(span_exporter=langfuse_exporter)
        tracer_provider.add_span_processor(langfuse_processor)

        self.name = LANGFUSE_TRACER_NAME
        self._otel_tracer = tracer_provider.get_tracer(self.name, langfuse_version)

    @property
    def tracer(self):
        return self._otel_tracer

    def get_current_span(self):
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
