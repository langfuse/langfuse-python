"""Utility functions for Langfuse OpenTelemetry integration.

This module provides utility functions for working with OpenTelemetry spans,
including formatting and serialization of span data.
"""

import json

from opentelemetry import trace as otel_trace_api
from opentelemetry.sdk import util
from opentelemetry.sdk.trace import ReadableSpan


def span_formatter(span: ReadableSpan):
    parent_id = (
        otel_trace_api.format_span_id(span.parent.span_id) if span.parent else None
    )
    start_time = util.ns_to_iso_str(span._start_time) if span._start_time else None
    end_time = util.ns_to_iso_str(span._end_time) if span._end_time else None
    status = {
        "status_code": str(span._status.status_code.name),
    }

    if span._status.description:
        status["description"] = span._status.description

    context = (
        {
            "trace_id": otel_trace_api.format_trace_id(span._context.trace_id),
            "span_id": otel_trace_api.format_span_id(span._context.span_id),
            "trace_state": repr(span._context.trace_state),
        }
        if span._context
        else None
    )

    instrumentationScope = json.loads(
        span._instrumentation_scope.to_json() if span._instrumentation_scope else "{}"
    )

    return (
        json.dumps(
            {
                "name": span._name,
                "context": context,
                "kind": str(span.kind),
                "parent_id": parent_id,
                "start_time": start_time,
                "end_time": end_time,
                "status": status,
                "attributes": span._format_attributes(span._attributes),
                "events": span._format_events(span._events),
                "links": span._format_links(span._links),
                "resource": json.loads(span.resource.to_json()),
                "instrumentationScope": instrumentationScope,
            },
            indent=2,
        )
        + "\n"
    )
