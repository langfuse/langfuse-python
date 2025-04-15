import json

from opentelemetry import trace as otel_trace_api
from opentelemetry.sdk import util
from opentelemetry.sdk.trace import ReadableSpan


def span_formatter(span: ReadableSpan):
    parent_id = None
    if span.parent is not None:
        parent_id = f"0x{otel_trace_api.format_span_id(span.parent.span_id)}"

    start_time = None
    if span._start_time:
        start_time = util.ns_to_iso_str(span._start_time)

    end_time = None
    if span._end_time:
        end_time = util.ns_to_iso_str(span._end_time)

    status = {
        "status_code": str(span._status.status_code.name),
    }
    if span._status.description:
        status["description"] = span._status.description

    return (
        json.dumps(
            {
                "name": span._name,
                "context": (
                    span._format_context(span._context) if span._context else None
                ),
                "kind": str(span.kind),
                "parent_id": parent_id,
                "start_time": start_time,
                "end_time": end_time,
                "status": status,
                "attributes": span._format_attributes(span._attributes),
                "events": span._format_events(span._events),
                "links": span._format_links(span._links),
                "resource": json.loads(span.resource.to_json()),
                "instrumentationScope": json.loads(
                    span._instrumentation_scope.to_json()
                    if span._instrumentation_scope
                    else "{}"
                ),
            },
            indent=2,
        )
        + "\n"
    )
