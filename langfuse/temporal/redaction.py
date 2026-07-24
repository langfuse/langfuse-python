"""Payload serialization, redaction, and size-limit helpers for the plugin.

The base plugin defaults to *metadata-only* capture. Payload capture is
opt-in per-surface (workflow inputs, workflow outputs, activity inputs,
activity outputs). When enabled, payloads always flow through this module
before being attached to a span so that size limits and user-provided
redaction callbacks are applied consistently.

This module is intentionally free of ``temporalio`` imports so that it can
run under a workflow sandbox and be tested without installing Temporal.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Optional

# A redaction callback receives the already-serialized JSON-ish value and
# returns a value that is safe to export. Users can drop fields, mask PII,
# or replace the whole payload with a summary.
RedactCallback = Callable[[Any], Any]


def _default_serialize(value: Any) -> Any:
    """Best-effort JSON-compatible coercion.

    Temporal activity / workflow payloads are arbitrary Python objects. We
    only capture them for observability, so a loose ``default=str`` fallback
    is appropriate here — correctness of application state is Temporal's job,
    not the plugin's.
    """
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        pass

    if hasattr(value, "model_dump"):
        try:
            return value.model_dump()
        except Exception:
            pass

    if hasattr(value, "__dict__"):
        try:
            attrs = vars(value)
            if attrs:
                return {k: _default_serialize(v) for k, v in attrs.items()}
        except Exception:
            pass

    return repr(value)


def _apply_size_limit(serialized: str, limit_bytes: Optional[int]) -> str:
    if limit_bytes is None or limit_bytes <= 0:
        return serialized
    encoded = serialized.encode("utf-8")
    if len(encoded) <= limit_bytes:
        return serialized
    # Truncate on a byte boundary; the marker keeps downstream consumers
    # aware that the payload is lossy.
    truncated = encoded[:limit_bytes].decode("utf-8", errors="ignore")
    return truncated + "…[truncated]"


def prepare_payload(
    value: Any,
    *,
    redact: Optional[RedactCallback] = None,
    size_limit_bytes: Optional[int] = None,
) -> Optional[str]:
    """Serialize ``value`` into a JSON string suitable for attaching to a span.

    Returns ``None`` if the payload was dropped by the redaction callback.
    Applies a size limit *after* serialization so the limit reflects what is
    actually exported.
    """
    if redact is not None:
        try:
            value = redact(value)
        except Exception:
            # A broken redact callback must not take down a workflow/activity.
            # Falling back to a placeholder is preferable to crashing the
            # user's code for the sake of observability.
            return "[redaction-error]"
        if value is None:
            return None

    serialized_value = _default_serialize(value)
    try:
        serialized = json.dumps(serialized_value, default=str, ensure_ascii=False)
    except Exception:
        serialized = repr(serialized_value)

    return _apply_size_limit(serialized, size_limit_bytes)


__all__ = ["RedactCallback", "prepare_payload"]
