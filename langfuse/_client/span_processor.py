"""Span processor for Langfuse OpenTelemetry integration.

This module defines the LangfuseSpanProcessor class, which extends OpenTelemetry's
BatchSpanProcessor with Langfuse-specific functionality. It handles exporting
spans to the Langfuse API with proper authentication and filtering.

Key features:
- HTTP-based span export to Langfuse API
- Basic authentication with Langfuse API keys
- Configurable batch processing behavior
- Project-scoped span filtering to prevent cross-project data leakage
"""

import base64
import os
from typing import Dict, List, Optional, Union, Mapping, Callable, Any

from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from requests import Session, PreparedRequest, Response

from langfuse._client.constants import LANGFUSE_TRACER_NAME
from langfuse._client.environment_variables import (
    LANGFUSE_FLUSH_AT,
    LANGFUSE_FLUSH_INTERVAL,
    LANGFUSE_OTEL_TRACES_EXPORT_PATH,
)
from langfuse._client.utils import span_formatter
from langfuse.logger import langfuse_logger
from langfuse.version import __version__ as langfuse_version


class LangfuseSpanProcessor(BatchSpanProcessor):
    """OpenTelemetry span processor that exports spans to the Langfuse API.

    This processor extends OpenTelemetry's BatchSpanProcessor with Langfuse-specific functionality:
    1. Project-scoped span filtering to prevent cross-project data leakage
    2. Instrumentation scope filtering to block spans from specific libraries/frameworks
    3. Configurable batch processing parameters for optimal performance
    4. HTTP-based span export to the Langfuse OTLP endpoint
    5. Debug logging for span processing operations
    6. Authentication with Langfuse API using Basic Auth

    The processor is designed to efficiently handle large volumes of spans with
    minimal overhead, while ensuring spans are only sent to the correct project.
    It integrates with OpenTelemetry's standard span lifecycle, adding Langfuse-specific
    filtering and export capabilities.
    """

    def __init__(
        self,
        *,
        public_key: str,
        secret_key: str,
        host: str,
        timeout: Optional[int] = None,
        flush_at: Optional[int] = None,
        flush_interval: Optional[float] = None,
        blocked_instrumentation_scopes: Optional[List[str]] = None,
        additional_headers: Optional[Dict[str, str]] = None,
    ):
        self.public_key = public_key
        self.blocked_instrumentation_scopes = (
            blocked_instrumentation_scopes
            if blocked_instrumentation_scopes is not None
            else []
        )

        env_flush_at = os.environ.get(LANGFUSE_FLUSH_AT, None)
        flush_at = flush_at or int(env_flush_at) if env_flush_at is not None else None

        env_flush_interval = os.environ.get(LANGFUSE_FLUSH_INTERVAL, None)
        flush_interval = (
            flush_interval or float(env_flush_interval)
            if env_flush_interval is not None
            else None
        )

        # Instead of creating a static headers dict, we create a session that will
        # dynamically generate them for each request made by the exporter.
        instrumented_session = _LangfuseInstrumentedSession(
            public_key=public_key,
            secret_key=secret_key,
            additional_headers=additional_headers,
        )

        traces_export_path = os.environ.get(LANGFUSE_OTEL_TRACES_EXPORT_PATH, None)

        endpoint = (
            f"{host}/{traces_export_path}"
            if traces_export_path
            else f"{host}/api/public/otel/v1/traces"
        )

        langfuse_span_exporter = OTLPSpanExporter(
            endpoint=endpoint,
            session=instrumented_session,
            timeout=timeout,
        )

        super().__init__(
            span_exporter=langfuse_span_exporter,
            export_timeout_millis=timeout * 1_000 if timeout else None,
            max_export_batch_size=flush_at,
            schedule_delay_millis=flush_interval * 1_000
            if flush_interval is not None
            else None,
        )

    def on_end(self, span: ReadableSpan) -> None:
        # Only export spans that belong to the scoped project
        # This is important to not send spans to wrong project in multi-project setups
        if self._is_langfuse_span(span) and not self._is_langfuse_project_span(span):
            langfuse_logger.debug(
                f"Security: Span rejected - belongs to project '{span.instrumentation_scope.attributes.get('public_key') if span.instrumentation_scope and span.instrumentation_scope.attributes else None}' but processor is for '{self.public_key}'. "
                f"This prevents cross-project data leakage in multi-project environments."
            )
            return

        # Do not export spans from blocked instrumentation scopes
        if self._is_blocked_instrumentation_scope(span):
            return

        langfuse_logger.debug(
            f"Trace: Processing span name='{span._name}' | Full details:\n{span_formatter(span)}"
        )

        super().on_end(span)

    @staticmethod
    def _is_langfuse_span(span: ReadableSpan) -> bool:
        return (
            span.instrumentation_scope is not None
            and span.instrumentation_scope.name == LANGFUSE_TRACER_NAME
        )

    def _is_blocked_instrumentation_scope(self, span: ReadableSpan) -> bool:
        return (
            span.instrumentation_scope is not None
            and span.instrumentation_scope.name in self.blocked_instrumentation_scopes
        )

    def _is_langfuse_project_span(self, span: ReadableSpan) -> bool:
        if not LangfuseSpanProcessor._is_langfuse_span(span):
            return False

        if span.instrumentation_scope is not None:
            public_key_on_span = (
                span.instrumentation_scope.attributes.get("public_key", None)
                if span.instrumentation_scope.attributes
                else None
            )

            return public_key_on_span == self.public_key

        return False


class _LangfuseInstrumentedSession(Session):
    """
    A custom requests.Session that adds dynamic headers before sending a request.
    This is used to inject fresh headers into the OTEL exporter's
    HTTP requests, bypassing the exporter's internal header caching.
    """

    def __init__(
        self,
        public_key: str,
        secret_key: str,
        additional_headers: Optional[
            Union[Mapping[str, Any], Callable[[], Mapping[str, Any]]]
        ],
    ) -> None:
        """Initializes the session with authentication and header details."""
        super().__init__()
        self._public_key = public_key
        self._secret_key = secret_key
        self._additional_headers = additional_headers

    def send(self, request: PreparedRequest, **kwargs: Any) -> Response:
        """
        Overrides the default send method to inject dynamic headers.
        This method is called just before any HTTP request is sent by the session.
        """
        # Prepare default headers with basic authentication
        basic_auth_header = "Basic " + base64.b64encode(
            f"{self._public_key}:{self._secret_key}".encode("utf-8")
        ).decode("ascii")

        default_headers: Dict[str, Any] = {
            "Authorization": basic_auth_header,
            "x_langfuse_sdk_name": "python",
            "x_langfuse_sdk_version": langfuse_version,
            "x_langfuse_public_key": self._public_key,
        }

        # Evaluate dynamic headers if they are provided
        dynamic_headers: Dict[str, Any] = {}
        if self._additional_headers is not None:
            if callable(self._additional_headers):
                # If it's a function, call it to get the headers
                dynamic_headers = dict(self._additional_headers())
            elif isinstance(self._additional_headers, Mapping):
                # If it's a mapping, convert it to a dict
                dynamic_headers = dict(self._additional_headers)

        # Merge default and dynamic headers. Dynamic headers will overwrite defaults on conflict.
        final_headers = {**default_headers, **dynamic_headers}

        # Update the request with the final, merged headers
        request.headers.update(final_headers)

        # Call the original send method with the modified request
        return super().send(request, **kwargs)
