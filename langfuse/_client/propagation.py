"""Attribute propagation utilities for Langfuse OpenTelemetry integration.

This module provides the `propagate_attributes` context manager for setting trace-level
attributes (user_id, session_id, metadata) that automatically propagate to all child spans
within the context.
"""

from typing import Any, Dict, Generator, List, Literal, Optional, Union, cast

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

PropagatedKeys = Literal[
    "user_id", "session_id", "metadata", "version", "tags", "public"
]
propagated_keys: List[PropagatedKeys] = [
    "user_id",
    "session_id",
    "metadata",
    "version",
    "tags",
    "public",
]


@_agnosticcontextmanager
def propagate_attributes(
    *,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    metadata: Optional[Dict[str, str]] = None,
    version: Optional[str] = None,
    tags: Optional[List[str]] = None,
    public: Optional[bool] = None,
    as_baggage: bool = False,
) -> Generator[Any, Any, Any]:
    """Propagate trace-level attributes to all spans created within this context.

    This context manager sets attributes on the currently active span AND automatically
    propagates them to all new child spans created within the context. This is the
    recommended way to set trace-level attributes like user_id, session_id, and metadata
    dimensions that should be consistently applied across all observations in a trace.

    **IMPORTANT**: Call this as early as possible within your trace/workflow. Only the
    currently active span and spans created after entering this context will have these
    attributes. Pre-existing spans will NOT be retroactively updated.

    **Why this matters**: Langfuse aggregation queries (e.g., total cost by user_id,
    filtering by session_id) only include observations that have the attribute set.
    If you call `propagate_attributes` late in your workflow, earlier spans won't be
    included in aggregations for that attribute.

    Args:
        user_id: User identifier to associate with all spans in this context.
            Must be US-ASCII string, ≤200 characters. Use this to track which user
            generated each trace and enable e.g. per-user cost/performance analysis.
        session_id: Session identifier to associate with all spans in this context.
            Must be US-ASCII string, ≤200 characters. Use this to group related traces
            within a user session (e.g., a conversation thread, multi-turn interaction).
        metadata: Additional key-value metadata to propagate to all spans.
            - Keys and values must be US-ASCII strings
            - All values must be ≤200 characters
            - Use for dimensions like internal correlating identifiers
            - AVOID: large payloads, sensitive data, non-string values (will be dropped with warning)
        version: Version identfier for parts of your application that are independently versioned, e.g. agents
        tags: List of tags to categorize the trace
        public: Whether the trace should be publicly accessible
        as_baggage: If True, propagates attributes using OpenTelemetry baggage for
            cross-process/service propagation. **Security warning**: When enabled,
            attribute values are added to HTTP headers on ALL outbound requests.
            Only enable if values are safe to transmit via HTTP headers and you need
            cross-service tracing. Default: False.

    Returns:
        Context manager that propagates attributes to all child spans.

    Example:
        Basic usage with user and session tracking:

        ```python
        from langfuse import Langfuse

        langfuse = Langfuse()

        # Set attributes early in the trace
        with langfuse.start_as_current_span(name="user_workflow") as span:
            with langfuse.propagate_attributes(
                user_id="user_123",
                session_id="session_abc",
                metadata={"experiment": "variant_a", "environment": "production"}
            ):
                # All spans created here will have user_id, session_id, and metadata
                with langfuse.start_span(name="llm_call") as llm_span:
                    # This span inherits: user_id, session_id, experiment, environment
                    ...

                with langfuse.start_generation(name="completion") as gen:
                    # This span also inherits all attributes
                    ...
        ```

        Late propagation (anti-pattern):

        ```python
        with langfuse.start_as_current_span(name="workflow") as span:
            # These spans WON'T have user_id
            early_span = langfuse.start_span(name="early_work")
            early_span.end()

            # Set attributes in the middle
            with langfuse.propagate_attributes(user_id="user_123"):
                # Only spans created AFTER this point will have user_id
                late_span = langfuse.start_span(name="late_work")
                late_span.end()

            # Result: Aggregations by user_id will miss "early_work" span
        ```

        Cross-service propagation with baggage (advanced):

        ```python
        # Service A - originating service
        with langfuse.start_as_current_span(name="api_request"):
            with langfuse.propagate_attributes(
                user_id="user_123",
                session_id="session_abc",
                as_baggage=True  # Propagate via HTTP headers
            ):
                # Make HTTP request to Service B
                response = requests.get("https://service-b.example.com/api")
                # user_id and session_id are now in HTTP headers

        # Service B - downstream service
        # OpenTelemetry will automatically extract baggage from HTTP headers
        # and propagate to spans in Service B
        ```

    Note:
        - **Validation**: All attribute values (user_id, session_id, metadata values)
          must be strings ≤200 characters. Invalid values will be dropped with a
          warning logged. Ensure values meet constraints before calling.
        - **OpenTelemetry**: This uses OpenTelemetry context propagation under the hood,
          making it compatible with other OTel-instrumented libraries.

    Raises:
        No exceptions are raised. Invalid values are logged as warnings and dropped.
    """
    context = otel_context_api.get_current()
    current_span = otel_trace_api.get_current_span()

    if user_id is not None and _validate_propagated_string(user_id, "user_id"):
        context = _set_propagated_attribute(
            key="user_id",
            value=user_id,
            context=context,
            span=current_span,
            as_baggage=as_baggage,
        )

    if session_id is not None and _validate_propagated_string(session_id, "session_id"):
        context = _set_propagated_attribute(
            key="session_id",
            value=session_id,
            context=context,
            span=current_span,
            as_baggage=as_baggage,
        )

    if version is not None and _validate_propagated_string(version, "version"):
        context = _set_propagated_attribute(
            key="version",
            value=version,
            context=context,
            span=current_span,
            as_baggage=as_baggage,
        )

    if public is not None and _validate_propagated_string(public, "public"):
        context = _set_propagated_attribute(
            key="public",
            value=public,
            context=context,
            span=current_span,
            as_baggage=as_baggage,
        )

    if tags is not None and all(
        _validate_propagated_string(tag, "tag") for tag in tags
    ):
        context = _set_propagated_attribute(
            key="tags",
            value=tags,
            context=context,
            span=current_span,
            as_baggage=as_baggage,
        )

    if metadata is not None:
        # Filter metadata to only include valid string values
        validated_metadata: Dict[str, str] = {}
        for key, value in metadata.items():
            if _validate_propagated_string(value, f"metadata.{key}"):
                validated_metadata[key] = value

        if validated_metadata:
            context = _set_propagated_attribute(
                key="metadata",
                value=validated_metadata,
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
) -> Dict[str, Union[str, bool, List[str]]]:
    propagated_attributes: Dict[str, Union[str, bool, List[str]]] = {}

    # Handle baggage
    baggage_entries = baggage.get_all(context=context)
    for baggage_key, baggage_value in baggage_entries.items():
        if baggage_key.startswith(LANGFUSE_BAGGAGE_PREFIX):
            span_key = _get_span_key_from_baggage_key(baggage_key)

            if span_key:
                propagated_attributes[span_key] = (
                    baggage_value
                    if isinstance(baggage_value, (str, list, bool))
                    else str(baggage_value)
                )

    # Handle OTEL context
    for key in propagated_keys:
        context_key = _get_propagated_context_key(key)
        value = otel_context_api.get_value(key=context_key, context=context)

        if value is None:
            continue

        if isinstance(value, dict):
            # Handle metadata
            for k, v in value.items():
                span_key = f"{LangfuseOtelSpanAttributes.TRACE_METADATA}.{k}"
                propagated_attributes[span_key] = v

        else:
            span_key = _get_propagated_span_key(key)

            propagated_attributes[span_key] = str(value)

    return propagated_attributes


def _set_propagated_attribute(
    *,
    key: PropagatedKeys,
    value: Union[str, bool, List[str], Dict[str, str]],
    context: otel_context_api.Context,
    span: otel_trace_api.Span,
    as_baggage: bool,
) -> otel_context_api.Context:
    # Get key names
    context_key = _get_propagated_context_key(key)
    span_key = _get_propagated_span_key(key)
    baggage_key = _get_propagated_baggage_key(key)

    # Merge metadata with previously set metadata keys
    if isinstance(value, dict):
        existing_metadata_in_context = cast(
            dict, otel_context_api.get_value(context_key) or {}
        )
        value = existing_metadata_in_context | value

    # Merge tags with previously set tags
    if isinstance(value, list):
        existing_tags_in_context = cast(
            list, otel_context_api.get_value(context_key) or []
        )
        merged_tags = list(existing_tags_in_context)
        merged_tags.extend(tag for tag in value if tag not in existing_tags_in_context)

        value = merged_tags

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


def _validate_propagated_string(value: str | bool, attribute_name: str) -> bool:
    """Validate a propagated attribute string value.

    Args:
        value: The string value to validate
        attribute_name: Name of the attribute for error messages

    Returns:
        True if valid, False otherwise (with warning logged)
    """
    if isinstance(value, bool):
        return True

    if not isinstance(value, str):
        langfuse_logger.warning(  # type: ignore
            f"Propagated attribute '{attribute_name}' value is not a string. Dropping value."
        )
        return False

    if len(value) > 200:
        langfuse_logger.warning(
            f"Propagated attribute '{attribute_name}' value is over 200 characters ({len(value)} chars). Dropping value."
        )
        return False

    return True


def _get_propagated_context_key(key: PropagatedKeys) -> str:
    return f"langfuse.propagated.{key}"


LANGFUSE_BAGGAGE_PREFIX = "langfuse_"


def _get_propagated_baggage_key(key: PropagatedKeys) -> str:
    return f"{LANGFUSE_BAGGAGE_PREFIX}{key}"


def _get_span_key_from_baggage_key(key: str) -> Optional[str]:
    if not key.startswith(LANGFUSE_BAGGAGE_PREFIX):
        return None

    # Remove prefix to get the actual key name
    suffix = key[len(LANGFUSE_BAGGAGE_PREFIX) :]

    # Exact match for user_id and session_id
    if suffix == "user_id":
        return LangfuseOtelSpanAttributes.TRACE_USER_ID

    if suffix == "session_id":
        return LangfuseOtelSpanAttributes.TRACE_SESSION_ID

    if suffix == "version":
        return LangfuseOtelSpanAttributes.VERSION

    if suffix == "tags":
        return LangfuseOtelSpanAttributes.TRACE_TAGS

    if suffix == "public":
        return LangfuseOtelSpanAttributes.TRACE_PUBLIC

    # Metadata keys have format: langfuse_metadata_{key_name}
    if suffix.startswith("metadata_"):
        metadata_key = suffix[len("metadata_") :]

        return f"{LangfuseOtelSpanAttributes.TRACE_METADATA}.{metadata_key}"

    return None


def _get_propagated_span_key(key: PropagatedKeys) -> str:
    return {
        "session_id": LangfuseOtelSpanAttributes.TRACE_SESSION_ID,
        "user_id": LangfuseOtelSpanAttributes.TRACE_USER_ID,
        "version": LangfuseOtelSpanAttributes.VERSION,
        "tags": LangfuseOtelSpanAttributes.TRACE_TAGS,
        "public": LangfuseOtelSpanAttributes.TRACE_PUBLIC,
    }.get(key) or f"{LangfuseOtelSpanAttributes.TRACE_METADATA}.{key}"
