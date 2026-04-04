"""Tests for span filter predicates used by the Langfuse span processor."""

from types import SimpleNamespace
from typing import Any, Optional

from langfuse.span_filter import (
    is_default_export_span,
    is_genai_span,
    is_known_llm_instrumentor,
    is_langfuse_span,
)


def _make_span(
    *,
    scope_name: Optional[str] = None,
    attributes: Optional[dict[Any, Any]] = None,
):
    scope = None if scope_name is None else SimpleNamespace(name=scope_name)
    return SimpleNamespace(instrumentation_scope=scope, attributes=attributes)


def test_is_langfuse_span_true():
    """Return true for a Langfuse SDK instrumentation scope."""
    assert is_langfuse_span(_make_span(scope_name="langfuse-sdk")) is True


def test_is_langfuse_span_false():
    """Return false for non-Langfuse instrumentation scopes."""
    assert is_langfuse_span(_make_span(scope_name="other-lib")) is False


def test_is_langfuse_span_no_scope():
    """Return false when instrumentation scope is missing."""
    assert is_langfuse_span(_make_span(scope_name=None)) is False


def test_is_genai_span_with_genai_attributes():
    """Return true when span attributes include a gen_ai.* key."""
    assert (
        is_genai_span(
            _make_span(
                attributes={"gen_ai.request.model": "gpt-4o", "http.method": "POST"}
            )
        )
        is True
    )


def test_is_genai_span_ignores_non_string_keys():
    """Ignore non-string keys when checking gen_ai.* attributes."""
    assert (
        is_genai_span(_make_span(attributes={1: "value", "http.method": "POST"}))
        is False
    )


def test_is_genai_span_no_attributes():
    """Return false when attributes are missing."""
    assert is_genai_span(_make_span(attributes=None)) is False


def test_is_known_llm_instrumentor_exact_match():
    """Return true for exact allowlisted scope names."""
    assert is_known_llm_instrumentor(_make_span(scope_name="ai")) is True


def test_is_known_llm_instrumentor_prefix_match():
    """Return true for allowlisted namespace scope descendants."""
    assert (
        is_known_llm_instrumentor(
            _make_span(scope_name="openinference.instrumentation.agno.worker")
        )
        is True
    )


def test_is_known_llm_instrumentor_unknown():
    """Return false for unknown instrumentation scopes."""
    assert is_known_llm_instrumentor(_make_span(scope_name="unknown.scope")) is False


def test_is_default_export_span_langfuse():
    """Export Langfuse spans with the default filter."""
    assert is_default_export_span(_make_span(scope_name="langfuse-sdk")) is True


def test_is_default_export_span_genai():
    """Export gen_ai spans with the default filter."""
    assert (
        is_default_export_span(
            _make_span(
                scope_name="unknown.scope", attributes={"gen_ai.prompt": "hello"}
            )
        )
        is True
    )


def test_is_default_export_span_known_scope():
    """Export known instrumentation scopes with the default filter."""
    assert is_default_export_span(_make_span(scope_name="langsmith")) is True


def test_is_default_export_span_rejects_unknown():
    """Reject unknown scopes without gen_ai attributes in default filter."""
    assert (
        is_default_export_span(
            _make_span(scope_name="unknown.scope", attributes={"http.method": "GET"})
        )
        is False
    )
