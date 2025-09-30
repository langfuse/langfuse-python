"""Context propagation utilities for Langfuse tracing.

This module provides a mixin class that enables automatic propagation of trace
attributes (session_id, user_id, metadata) from parent contexts to child spans
using OpenTelemetry's context and baggage mechanisms.

The mixin is shared between the main Langfuse client and span classes to provide
consistent context propagation behavior across the SDK.
"""

import json
from typing import Any, Generator

from opentelemetry import (
    baggage as otel_baggage_api,
    context as otel_context_api,
    trace as otel_trace_api,
)
from opentelemetry.util._decorator import _agnosticcontextmanager

# Context key constants for Langfuse context propagation
LANGFUSE_CTX_USER_ID = "langfuse.ctx.user.id"
LANGFUSE_CTX_SESSION_ID = "langfuse.ctx.session.id"
LANGFUSE_CTX_METADATA = "langfuse.ctx.metadata"


class LangfuseContextPropagationMixin:
    """Mixin providing context managers for automatic trace attribute propagation.

    This mixin adds three context managers (session, user, metadata) that enable
    automatic propagation of trace attributes to all child spans created within
    their scope. The propagation works through OpenTelemetry's context mechanism
    for local (same-service) propagation, with optional baggage for cross-service
    propagation.

    Classes that inherit this mixin gain the ability to create contexts where
    certain attributes are automatically applied to all spans without manual
    specification.
    """

    @_agnosticcontextmanager
    def session(
        self, id: str, *, as_baggage: bool = False
    ) -> Generator[None, None, None]:
        """Create a session context manager that propagates session_id to all child spans.

        Args:
            id (str): The session identifier to propagate to child spans.
            as_baggage (bool, optional): If True, stores the session_id in OpenTelemetry baggage
                for cross-service propagation. If False, stores only in local context for
                current-service propagation. Defaults to False.

        Returns:
            Context manager that sets session_id on all spans created within its scope.

        Warning:
            When as_baggage=True, the session_id will be included in HTTP headers of any
            outbound requests made within this context. Only use this for non-sensitive
            identifiers that are safe to transmit across service boundaries.

        Example:
            ```python
            # Local context only (default)
            with langfuse.session(id="session_123"):
                with langfuse.start_as_current_span(name="process-request") as span:
                    # This span and all its children will have session_id="session_123"
                    child_span = langfuse.start_span(name="child-operation")

            # Cross-service propagation (use with caution)
            with langfuse.session(id="session_123", as_baggage=True):
                # session_id will be propagated to external service calls
                response = requests.get("https://api.example.com/data")
            ```
        """
        # Set context variable
        new_context = otel_context_api.set_value(LANGFUSE_CTX_SESSION_ID, id)
        token = otel_context_api.attach(new_context)

        # Set attribute on currently active span if exists
        current_span = otel_trace_api.get_current_span()
        if current_span is not None and current_span.is_recording():
            current_span.set_attribute("session.id", id)

        # Set baggage if requested
        baggage_token = None
        if as_baggage:
            new_baggage = otel_baggage_api.set_baggage("session.id", id)
            baggage_token = otel_context_api.attach(new_baggage)

        try:
            yield
        finally:
            # Always detach context token
            otel_context_api.detach(token)

            # Detach baggage token if it was set
            if baggage_token is not None:
                otel_context_api.detach(baggage_token)

    @_agnosticcontextmanager
    def user(self, id: str, *, as_baggage: bool = False) -> Generator[None, None, None]:
        """Create a user context manager that propagates user_id to all child spans.

        Args:
            id (str): The user identifier to propagate to child spans.
            as_baggage (bool, optional): If True, stores the user_id in OpenTelemetry baggage
                for cross-service propagation. If False, stores only in local context for
                current-service propagation. Defaults to False.

        Returns:
            Context manager that sets user_id on all spans created within its scope.

        Warning:
            When as_baggage=True, the user_id will be included in HTTP headers of any
            outbound requests made within this context. This may leak sensitive user
            information to external services. Use with extreme caution.

        Example:
            ```python
            # Local context only (default, recommended for user IDs)
            with langfuse.user(id="user_456"):
                with langfuse.start_as_current_span(name="user-action") as span:
                    # This span and all its children will have user_id="user_456"
                    pass

            # Cross-service propagation (NOT recommended for sensitive user IDs)
            with langfuse.user(id="public_user_456", as_baggage=True):
                # user_id will be propagated to external service calls
                response = requests.get("https://api.example.com/data")
            ```
        """
        # Set context variable
        new_context = otel_context_api.set_value(LANGFUSE_CTX_USER_ID, id)
        token = otel_context_api.attach(new_context)

        # Set attribute on currently active span if exists
        current_span = otel_trace_api.get_current_span()
        if current_span is not None and current_span.is_recording():
            current_span.set_attribute("user.id", id)

        # Set baggage if requested
        baggage_token = None
        if as_baggage:
            new_baggage = otel_baggage_api.set_baggage("user.id", id)
            baggage_token = otel_context_api.attach(new_baggage)

        try:
            yield
        finally:
            # Always detach context token
            otel_context_api.detach(token)

            # Detach baggage token if it was set
            if baggage_token is not None:
                otel_context_api.detach(baggage_token)

    @_agnosticcontextmanager
    def metadata(
        self, *, as_baggage: bool = False, **kwargs: Any
    ) -> Generator[None, None, None]:
        """Create a metadata context manager that propagates metadata to all child spans.

        Args:
            as_baggage (bool, optional): If True, stores the metadata in OpenTelemetry baggage
                for cross-service propagation. If False, stores only in local context for
                current-service propagation. Defaults to False.
            **kwargs: Metadata key-value pairs. Values should not exceed 200 characters.

        Returns:
            Context manager that sets metadata on all spans created within its scope.

        Warning:
            When as_baggage=True, all metadata key-value pairs will be included in HTTP
            headers of any outbound requests made within this context. Ensure no sensitive
            information is included in the metadata when using cross-service propagation.

        Example:
            ```python
            # Local context only (default)
            with langfuse.metadata(experiment="A/B", version="1.2.3"):
                with langfuse.start_as_current_span(name="experiment-run") as span:
                    # This span and all its children will have the metadata
                    pass

            # Cross-service propagation (use with caution)
            with langfuse.metadata(as_baggage=True, experiment="A/B", service="api"):
                # metadata will be propagated to external service calls
                response = requests.get("https://api.example.com/data")
            ```
        """
        if not kwargs:
            # No metadata to set, just yield
            yield
            return

        # Store metadata as a dict in context (not JSON string)
        # This allows span_processor to distribute keys as individual attributes
        new_context = otel_context_api.set_value(LANGFUSE_CTX_METADATA, kwargs)
        token = otel_context_api.attach(new_context)

        # Set attributes on currently active span if exists
        current_span = otel_trace_api.get_current_span()
        if current_span is not None and current_span.is_recording():
            for key, value in kwargs.items():
                attr_key = f"langfuse.metadata.{key}"
                # Convert value to appropriate type for span attribute
                if isinstance(value, (str, int, float, bool)):
                    attr_value = value
                else:
                    # For complex types, convert to JSON string
                    attr_value = json.dumps(value)
                current_span.set_attribute(attr_key, attr_value)

        # Set baggage if requested
        baggage_token = None
        if as_baggage:
            # Start with None context and chain baggage settings
            new_baggage = None

            # Add each metadata key-value pair to baggage
            for key, value in kwargs.items():
                # Convert value to string and truncate if needed for baggage
                str_value = str(value)
                if len(str_value) > 200:
                    str_value = str_value[:200]

                baggage_key = f"metadata.{key}"
                new_baggage = otel_baggage_api.set_baggage(
                    baggage_key, str_value, new_baggage
                )

            # Attach the new baggage context
            if new_baggage is not None:
                baggage_token = otel_context_api.attach(new_baggage)

        try:
            yield
        finally:
            # Always detach context token
            otel_context_api.detach(token)

            # Detach all baggage tokens if they were set
            if baggage_token is not None:
                otel_context_api.detach(baggage_token)
