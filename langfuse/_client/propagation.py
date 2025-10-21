from typing import Any, Dict, Generator, List, Literal, Optional, Union

from opentelemetry import baggage
from opentelemetry import (
    baggage as otel_baggage_api,
)
from opentelemetry import (
    context as otel_context_api,
)
from opentelemetry import (
    trace as otel_trace_api,
)
from opentelemetry.util._decorator import _agnosticcontextmanager

from langfuse._client.attributes import LangfuseOtelSpanAttributes
from langfuse.logger import langfuse_logger

PropagatedKeys = Literal["user_id", "session_id", "metadata"]


@_agnosticcontextmanager
def propagate_attributes(
    *,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    metadata: Optional[Dict[str, str]] = None,
    as_baggage: bool = False,
) -> Generator[Any, Any, Any]:
    context = otel_context_api.get_current()
    current_span = otel_trace_api.get_current_span()

    if user_id is not None:
        context = _set_propagated_attribute(
            key="user_id",
            value=user_id,
            context=context,
            span=current_span,
            as_baggage=as_baggage,
        )

    if session_id is not None:
        context = _set_propagated_attribute(
            key="session_id",
            value=session_id,
            context=context,
            span=current_span,
            as_baggage=as_baggage,
        )

    if metadata is not None:
        context = _set_propagated_attribute(
            key="metadata",
            value=_validate_propagated_metadata(metadata),
            context=context,
            span=current_span,
            as_baggage=as_baggage,
        )

    # Activate context, execute, and detach context
    token = otel_context_api.attach(context=context)

    try:
        yield

    finally:
        otel_context_api.detach(token)


def _get_propagated_attributes_from_context(
    context: otel_context_api.Context,
) -> Dict[str, str]:
    propagated_attributes: Dict[str, str] = {}

    # Handle baggage
    baggage_entries = baggage.get_all(context=context)
    for baggage_key, baggage_value in baggage_entries.items():
        if baggage_key.startswith(LANGFUSE_BAGGAGE_PREFIX):
            span_key = _get_span_key_from_baggage_key(baggage_key)

            if span_key:
                propagated_attributes[span_key] = str(baggage_value)

    # Handle OTEL context
    propagated_keys: List[PropagatedKeys] = ["user_id", "session_id", "metadata"]

    for key in propagated_keys:
        context_key = _get_propagated_context_key(key)
        value = otel_context_api.get_value(key=context_key, context=context)

        if isinstance(value, dict):
            # Handle metadata
            for k, v in value.items():
                span_key = f"{LangfuseOtelSpanAttributes.TRACE_METADATA}.{k}"
                propagated_attributes[span_key] = v

        else:
            span_key = {
                "user_id": LangfuseOtelSpanAttributes.TRACE_USER_ID,
                "session_id": LangfuseOtelSpanAttributes.TRACE_SESSION_ID,
            }.get(key, f"{LangfuseOtelSpanAttributes.TRACE_METADATA}.{key}")

            propagated_attributes[span_key] = str(value)

    return propagated_attributes


def _set_propagated_attribute(
    *,
    key: PropagatedKeys,
    value: Union[str, Dict[str, str]],
    context: otel_context_api.Context,
    span: otel_trace_api.Span,
    as_baggage: bool,
) -> otel_context_api.Context:
    # Get key names
    context_key = _get_propagated_context_key(key)
    span_key = _get_propagated_span_key(key)
    baggage_key = _get_propagated_baggage_key(key)

    # Set in context
    context = otel_context_api.set_value(
        key=context_key,
        value=value,
        context=context,
    )

    # Set on current span
    if span is not None and span.is_recording():
        if isinstance(value, dict):
            # Handle metadata
            for k, v in value.items():
                span.set_attribute(
                    key=f"{LangfuseOtelSpanAttributes.TRACE_METADATA}.{k}",
                    value=v,
                )

        else:
            span.set_attribute(key=span_key, value=value)

    # Set on baggage
    if as_baggage:
        if isinstance(value, dict):
            # Handle metadata
            for k, v in value.items():
                context = otel_baggage_api.set_baggage(
                    name=f"{baggage_key}_{k}", value=v, context=context
                )
        else:
            context = otel_baggage_api.set_baggage(
                name=baggage_key, value=value, context=context
            )

    return context


def _validate_propagated_metadata(metadata: Dict[str, str]) -> Dict[str, str]:
    validated_metadata: Dict[str, str] = {}

    for key, value in metadata.items():
        if not isinstance(value, str):
            langfuse_logger.warning(  # type: ignore
                f"Propagated attribute value of '{key}' not a string. Dropping value."
            )
            continue

        if len(value) > 200:
            langfuse_logger.warning(
                f"Propagated attribute value of '{key}' is over 200 characters ({len(value)} chars). Dropping value."
            )
            continue

        validated_metadata[key] = value

    return validated_metadata


def _get_propagated_context_key(key: PropagatedKeys) -> str:
    return f"langfuse.propagated.{key}"


LANGFUSE_BAGGAGE_PREFIX = "langfuse_"


def _get_propagated_baggage_key(key: PropagatedKeys) -> str:
    return f"{LANGFUSE_BAGGAGE_PREFIX}{key}"


def _get_span_key_from_baggage_key(key: str) -> Optional[str]:
    if not key.startswith(LANGFUSE_BAGGAGE_PREFIX):
        return None

    if "user_id" in key:
        return LangfuseOtelSpanAttributes.TRACE_USER_ID

    if "session_id" in key:
        return LangfuseOtelSpanAttributes.TRACE_SESSION_ID

    return f"{LangfuseOtelSpanAttributes.TRACE_METADATA}.{key}"


def _get_propagated_span_key(key: PropagatedKeys) -> str:
    return {
        "session_id": LangfuseOtelSpanAttributes.TRACE_SESSION_ID,
        "user_id": LangfuseOtelSpanAttributes.TRACE_USER_ID,
    }.get(key) or f"{LangfuseOtelSpanAttributes.TRACE_METADATA}.{key}"
