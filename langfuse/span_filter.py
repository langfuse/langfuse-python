"""Public span filter helpers for Langfuse OpenTelemetry export control."""

from langfuse._client.span_filter import (
    KNOWN_LLM_INSTRUMENTATION_SCOPES,
    KNOWN_LLM_INSTRUMENTATION_SCOPE_PREFIXES,
    is_default_export_span,
    is_genai_span,
    is_known_llm_instrumentor,
    is_langfuse_span,
)

__all__ = [
    "is_default_export_span",
    "is_langfuse_span",
    "is_genai_span",
    "is_known_llm_instrumentor",
    "KNOWN_LLM_INSTRUMENTATION_SCOPES",
    "KNOWN_LLM_INSTRUMENTATION_SCOPE_PREFIXES",
]
