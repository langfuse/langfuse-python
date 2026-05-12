from opentelemetry import baggage
from opentelemetry import context as otel_context_api
from opentelemetry import trace as trace_api
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.trace import NonRecordingSpan, SpanContext, TraceFlags

from langfuse._client.attributes import LangfuseOtelSpanAttributes
from langfuse._client.constants import LANGFUSE_TRACER_NAME
from langfuse._client.propagation import LANGFUSE_TRACE_ID_BAGGAGE_KEY
from langfuse._client.span_processor import LangfuseSpanProcessor

PUBLIC_KEY = "test-public-key"
SECRET_KEY = "test-secret-key"


def _create_processor(memory_exporter, **kwargs):
    tracer_provider = TracerProvider()
    processor = LangfuseSpanProcessor(
        public_key=PUBLIC_KEY,
        secret_key=SECRET_KEY,
        base_url="http://test-host",
        span_exporter=memory_exporter,
        **kwargs,
    )
    tracer_provider.add_span_processor(processor)

    return tracer_provider, processor


def _get_spans_by_name(memory_exporter):
    return {span.name: span for span in memory_exporter.get_finished_spans()}


def _langfuse_tracer(tracer_provider):
    return tracer_provider.get_tracer(
        LANGFUSE_TRACER_NAME,
        "test",
        attributes={"public_key": PUBLIC_KEY},
    )


def test_filtered_parent_marks_exported_children_as_app_roots(memory_exporter):
    tracer_provider, processor = _create_processor(memory_exporter)
    filtered_tracer = tracer_provider.get_tracer("requests")
    langfuse_tracer = _langfuse_tracer(tracer_provider)

    with filtered_tracer.start_as_current_span("filtered-parent"):
        with langfuse_tracer.start_as_current_span("child-a"):
            pass

        with langfuse_tracer.start_as_current_span("child-b"):
            pass

    processor.force_flush()

    spans = _get_spans_by_name(memory_exporter)

    assert "filtered-parent" not in spans
    assert spans["child-a"].attributes[LangfuseOtelSpanAttributes.IS_APP_ROOT] is True
    assert spans["child-b"].attributes[LangfuseOtelSpanAttributes.IS_APP_ROOT] is True
    assert processor._app_root_traces == {}


def test_exported_parent_suppresses_exported_child_app_root(memory_exporter):
    tracer_provider, processor = _create_processor(memory_exporter)
    langfuse_tracer = _langfuse_tracer(tracer_provider)

    with langfuse_tracer.start_as_current_span("parent"):
        with langfuse_tracer.start_as_current_span("child"):
            pass

    processor.force_flush()

    spans = _get_spans_by_name(memory_exporter)

    assert spans["parent"].attributes[LangfuseOtelSpanAttributes.IS_APP_ROOT] is True
    assert LangfuseOtelSpanAttributes.IS_APP_ROOT not in spans["child"].attributes
    assert processor._app_root_traces == {}


def test_filtered_direct_parent_marks_child_even_when_grandparent_exports(
    memory_exporter,
):
    tracer_provider, processor = _create_processor(memory_exporter)
    filtered_tracer = tracer_provider.get_tracer("requests")
    langfuse_tracer = _langfuse_tracer(tracer_provider)

    with langfuse_tracer.start_as_current_span("grandparent"):
        with filtered_tracer.start_as_current_span("filtered-parent"):
            with langfuse_tracer.start_as_current_span("child"):
                pass

    processor.force_flush()

    spans = _get_spans_by_name(memory_exporter)

    assert "filtered-parent" not in spans
    assert (
        spans["grandparent"].attributes[LangfuseOtelSpanAttributes.IS_APP_ROOT] is True
    )
    assert spans["child"].attributes[LangfuseOtelSpanAttributes.IS_APP_ROOT] is True
    assert processor._app_root_traces == {}


def test_same_trace_baggage_claim_suppresses_local_app_root(memory_exporter):
    tracer_provider, processor = _create_processor(memory_exporter)
    langfuse_tracer = _langfuse_tracer(tracer_provider)
    trace_id = int("1" * 32, 16)
    parent_context = _remote_parent_context(trace_id=trace_id)
    parent_context = baggage.set_baggage(
        name=LANGFUSE_TRACE_ID_BAGGAGE_KEY,
        value=format(trace_id, "032x"),
        context=parent_context,
    )

    span = langfuse_tracer.start_span("downstream-root", context=parent_context)
    span.end()
    processor.force_flush()

    spans = _get_spans_by_name(memory_exporter)

    assert (
        LangfuseOtelSpanAttributes.IS_APP_ROOT
        not in spans["downstream-root"].attributes
    )
    assert processor._app_root_traces == {}


def test_different_trace_baggage_claim_does_not_suppress_local_app_root(
    memory_exporter,
):
    tracer_provider, processor = _create_processor(memory_exporter)
    langfuse_tracer = _langfuse_tracer(tracer_provider)
    trace_id = int("1" * 32, 16)
    parent_context = _remote_parent_context(trace_id=trace_id)
    parent_context = baggage.set_baggage(
        name=LANGFUSE_TRACE_ID_BAGGAGE_KEY,
        value="2" * 32,
        context=parent_context,
    )

    span = langfuse_tracer.start_span("downstream-root", context=parent_context)
    span.end()
    processor.force_flush()

    spans = _get_spans_by_name(memory_exporter)

    assert (
        spans["downstream-root"].attributes[LangfuseOtelSpanAttributes.IS_APP_ROOT]
        is True
    )
    assert processor._app_root_traces == {}


def test_local_baggage_claim_does_not_suppress_child_of_filtered_parent(
    memory_exporter,
):
    def should_export_span(span: ReadableSpan) -> bool:
        return span.name != "parent"

    tracer_provider, processor = _create_processor(
        memory_exporter,
        should_export_span=should_export_span,
    )
    langfuse_tracer = _langfuse_tracer(tracer_provider)

    with langfuse_tracer.start_as_current_span("parent") as parent:
        context_with_claim = baggage.set_baggage(
            name=LANGFUSE_TRACE_ID_BAGGAGE_KEY,
            value=format(parent.context.trace_id, "032x"),
            context=otel_context_api.get_current(),
        )
        token = otel_context_api.attach(context_with_claim)

        try:
            with langfuse_tracer.start_as_current_span("child"):
                pass
        finally:
            otel_context_api.detach(token)

    processor.force_flush()

    spans = _get_spans_by_name(memory_exporter)

    assert "parent" not in spans
    assert spans["child"].attributes[LangfuseOtelSpanAttributes.IS_APP_ROOT] is True
    assert processor._app_root_traces == {}


def test_start_time_false_positive_can_leave_exported_child_without_app_root(
    memory_exporter,
):
    def should_export_span(span: ReadableSpan) -> bool:
        if span.name == "parent":
            return span.end_time is None

        return True

    tracer_provider, processor = _create_processor(
        memory_exporter,
        should_export_span=should_export_span,
    )
    langfuse_tracer = _langfuse_tracer(tracer_provider)

    with langfuse_tracer.start_as_current_span("parent"):
        with langfuse_tracer.start_as_current_span("child"):
            pass

    processor.force_flush()

    spans = _get_spans_by_name(memory_exporter)

    assert "parent" not in spans
    assert LangfuseOtelSpanAttributes.IS_APP_ROOT not in spans["child"].attributes
    assert processor._app_root_traces == {}


def test_active_langfuse_scope_sets_baggage_after_root_start(
    langfuse_memory_client,
    memory_exporter,
):
    with langfuse_memory_client.start_as_current_observation(name="root") as root:
        baggage_entries = baggage.get_all(context=otel_context_api.get_current())

        assert baggage_entries[LANGFUSE_TRACE_ID_BAGGAGE_KEY] == root.trace_id

        with langfuse_memory_client.start_as_current_observation(name="child"):
            pass

    langfuse_memory_client.flush()

    spans = _get_spans_by_name(memory_exporter)

    assert spans["root"].attributes[LangfuseOtelSpanAttributes.IS_APP_ROOT] is True
    assert LangfuseOtelSpanAttributes.IS_APP_ROOT not in spans["child"].attributes
    assert "langfuse.trace.metadata.trace_id" not in spans["child"].attributes


def _remote_parent_context(*, trace_id: int):
    span_context = SpanContext(
        trace_id=trace_id,
        span_id=int("a" * 16, 16),
        is_remote=True,
        trace_flags=TraceFlags(TraceFlags.SAMPLED),
    )

    return trace_api.set_span_in_context(NonRecordingSpan(span_context))
