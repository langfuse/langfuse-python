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
import json
import os
from typing import Dict, List, Optional

from opentelemetry import baggage, context as context_api
from opentelemetry.context import Context
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import ReadableSpan, Span
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from langfuse._client.constants import LANGFUSE_TRACER_NAME
from langfuse._client.environment_variables import (
    LANGFUSE_FLUSH_AT,
    LANGFUSE_FLUSH_INTERVAL,
    LANGFUSE_OTEL_TRACES_EXPORT_PATH,
)
from langfuse._client.utils import span_formatter
from langfuse.logger import langfuse_logger
from langfuse.version import __version__ as langfuse_version
from langfuse._client.context_propagation import (
    LANGFUSE_CTX_USER_ID,
    LANGFUSE_CTX_SESSION_ID,
    LANGFUSE_CTX_METADATA,
)


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

        basic_auth_header = "Basic " + base64.b64encode(
            f"{public_key}:{secret_key}".encode("utf-8")
        ).decode("ascii")

        # Prepare default headers
        default_headers = {
            "Authorization": basic_auth_header,
            "x_langfuse_sdk_name": "python",
            "x_langfuse_sdk_version": langfuse_version,
            "x_langfuse_public_key": public_key,
        }

        # Merge additional headers if provided
        headers = {**default_headers, **(additional_headers or {})}

        traces_export_path = os.environ.get(LANGFUSE_OTEL_TRACES_EXPORT_PATH, None)

        endpoint = (
            f"{host}/{traces_export_path}"
            if traces_export_path
            else f"{host}/api/public/otel/v1/traces"
        )

        langfuse_span_exporter = OTLPSpanExporter(
            endpoint=endpoint,
            headers=headers,
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

    def on_start(self, span: Span, parent_context: Optional[Context] = None) -> None:
        """Handle span start event and propagate context and baggage to span attributes.

        This method is called when a span starts and applies context propagation:
        1. Propagates all baggage keys as span attributes
        2. Propagates langfuse.ctx.* context variables as span attributes
        3. Distributes langfuse.ctx.metadata keys as individual langfuse.metadata.* attributes

        Args:
            span: The span that is starting
            parent_context: The context when the span was created (optional)
        """
        if self._is_langfuse_span(span) and not self._is_langfuse_project_span(span):
            langfuse_logger.debug(
                f"Security: Span rejected - belongs to project '{span.instrumentation_scope.attributes.get('public_key') if span.instrumentation_scope and span.instrumentation_scope.attributes else None}' but processor is for '{self.public_key}'. "
                f"This prevents cross-project data leakage in multi-project environments."
            )
            return super().on_start(span, parent_context)

        # Get the current context (use parent_context if available, otherwise current)
        current_context = parent_context or context_api.get_current()

        # Dictionary to collect span attributes that were propagated
        propagated_attributes = {}

        # 1. Propagate all baggage keys as span attributes
        baggage_entries = baggage.get_all(context=current_context)
        for key, value in baggage_entries.items():
            # Check if this baggage entry is already present as a span attribute
            if not hasattr(span.attributes, key) or (
                span.attributes is not None and span.attributes.get(key) != value
            ):
                propagated_attributes[key] = value
                langfuse_logger.debug(
                    f"Propagated baggage key '{key}' = '{value}' to span '{span.name}'"
                )

        # 2. Propagate langfuse.ctx.* context variables
        langfuse_ctx_keys = [LANGFUSE_CTX_USER_ID, LANGFUSE_CTX_SESSION_ID]
        for ctx_key in langfuse_ctx_keys:
            try:
                value = context_api.get_value(ctx_key, context=current_context)
                if value is not None:
                    # Convert context key to span attribute name (remove langfuse.ctx. prefix)
                    attr_key = ctx_key.replace("langfuse.ctx.", "")

                    # Only propagate if not already set on span
                    if not hasattr(span.attributes, attr_key) or (
                        span.attributes is not None
                        and span.attributes.get(attr_key) != value
                    ):
                        propagated_attributes[attr_key] = value
                        langfuse_logger.debug(
                            f"Propagated context key '{ctx_key}' = '{value}' to span '{span.name}'"
                        )
            except Exception as e:
                langfuse_logger.debug(f"Could not read context key '{ctx_key}': {e}")

        # 3. Handle langfuse.ctx.metadata - distribute keys as individual attributes
        try:
            # Get metadata dict from context
            metadata_dict = context_api.get_value(
                LANGFUSE_CTX_METADATA, context=current_context
            )
            if metadata_dict is not None and isinstance(metadata_dict, dict):
                # Set each metadata key as a separate span attribute with langfuse.metadata. prefix
                for key, value in metadata_dict.items():
                    attr_key = f"langfuse.metadata.{key}"

                    # Convert value to appropriate type for span attribute
                    if isinstance(value, (str, int, float, bool)):
                        attr_value = value
                    else:
                        # For complex types, convert to JSON string
                        attr_value = json.dumps(value)

                    # Only propagate if not already set or different
                    existing_value = (
                        span.attributes.get(attr_key)
                        if hasattr(span, "attributes") and span.attributes is not None
                        else None
                    )
                    if existing_value != attr_value:
                        propagated_attributes[attr_key] = attr_value
                        langfuse_logger.debug(
                            f"Propagated metadata key '{key}' = '{attr_value}' to span '{span.name}'"
                        )
        except Exception as e:
            langfuse_logger.debug(f"Could not read metadata from context: {e}")

        # Log summary of propagated attributes
        if propagated_attributes:
            langfuse_logger.debug(
                f"Propagated {len(propagated_attributes)} attributes to span '{span.name}': {list(propagated_attributes.keys())}"
            )

        # Set all propagated attributes on the span
        for key, value in propagated_attributes.items():
            span.set_attribute(key, value)  # type: ignore[arg-type]

        return super().on_start(span, parent_context)

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
