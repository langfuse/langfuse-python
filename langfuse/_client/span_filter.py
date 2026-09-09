"""Span filter predicates for controlling OpenTelemetry span export.

This module provides composable filter functions that determine which spans
the LangfuseSpanProcessor forwards to the Langfuse backend.
"""

from opentelemetry.sdk.trace import ReadableSpan

from langfuse._client.constants import LANGFUSE_TRACER_NAME

KNOWN_LLM_INSTRUMENTATION_SCOPE_PREFIXES = frozenset(
    {
        LANGFUSE_TRACER_NAME,
        "agent_framework",
        "autogen-core",
        "ai",
        "haystack",
        "langsmith",
        "litellm",
        "openinference",
        "opentelemetry.instrumentation.agno",
        "opentelemetry.instrumentation.alephalpha",
        "opentelemetry.instrumentation.anthropic",
        "opentelemetry.instrumentation.bedrock",
        "opentelemetry.instrumentation.cohere",
        "opentelemetry.instrumentation.crewai",
        "opentelemetry.instrumentation.google_generativeai",
        "opentelemetry.instrumentation.groq",
        "opentelemetry.instrumentation.haystack",
        "opentelemetry.instrumentation.langchain",
        "opentelemetry.instrumentation.llamaindex",
        "opentelemetry.instrumentation.mistralai",
        "opentelemetry.instrumentation.ollama",
        "opentelemetry.instrumentation.openai",
        "opentelemetry.instrumentation.openai_agents",
        "opentelemetry.instrumentation.openai_v2",
        "opentelemetry.instrumentation.replicate",
        "opentelemetry.instrumentation.sagemaker",
        "opentelemetry.instrumentation.together",
        "opentelemetry.instrumentation.transformers",
        "opentelemetry.instrumentation.vertexai",
        "opentelemetry.instrumentation.voyageai",
        "opentelemetry.instrumentation.watsonx",
        "opentelemetry.instrumentation.writer",
        "pydantic-ai",
        "strands-agents",
        "vllm",
    }
)
"""Known instrumentation scope namespace prefixes.

Prefix matching is boundary-aware:
- exact match (``scope == prefix``)
- direct descendant scopes (``scope.startswith(prefix + ".")``)

Please create a GitHub issue in https://github.com/langfuse/langfuse if you'd like to expand this default allow list.
"""

APP_ROOT_INELIGIBLE_SPANS = frozenset(
    {
        ("litellm", "raw_gen_ai_request"),
    }
)
"""Instrumentor spans that should not become app roots when they have a parent."""


def is_langfuse_span(span: ReadableSpan) -> bool:
    """Return whether the span was created by the Langfuse SDK tracer."""
    return (
        span.instrumentation_scope is not None
        and span.instrumentation_scope.name == LANGFUSE_TRACER_NAME
    )


def is_genai_span(span: ReadableSpan) -> bool:
    """Return whether the span has any ``gen_ai.*`` semantic convention attribute."""
    if span.attributes is None:
        return False

    return any(
        isinstance(key, str) and key.startswith("gen_ai")
        for key in span.attributes.keys()
    )


def _matches_scope_prefix(scope_name: str, prefix: str) -> bool:
    """Return whether a scope matches a prefix using namespace boundaries."""
    return scope_name == prefix or scope_name.startswith(f"{prefix}.")


def is_known_llm_instrumentor(span: ReadableSpan) -> bool:
    """Return whether the span comes from a known LLM instrumentation scope."""
    if span.instrumentation_scope is None:
        return False

    scope_name = span.instrumentation_scope.name

    return any(
        _matches_scope_prefix(scope_name, prefix)
        for prefix in KNOWN_LLM_INSTRUMENTATION_SCOPE_PREFIXES
    )


def is_default_export_span(span: ReadableSpan) -> bool:
    """Return whether a span should be exported by default."""
    return (
        is_langfuse_span(span) or is_genai_span(span) or is_known_llm_instrumentor(span)
    )


def is_app_root_eligible(span: ReadableSpan) -> bool:
    """Return whether an exported span may be marked as an application root."""
    if span.parent is None or span.instrumentation_scope is None:
        return True

    return (
        span.instrumentation_scope.name,
        span.name,
    ) not in APP_ROOT_INELIGIBLE_SPANS
