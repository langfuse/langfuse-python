import threading
from concurrent.futures import ThreadPoolExecutor

from opentelemetry import baggage
from opentelemetry import context as otel_context_api
from opentelemetry import trace as trace_api
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.trace import NonRecordingSpan, SpanContext, TraceFlags

from langfuse._client.attributes import LangfuseOtelSpanAttributes
from langfuse._client.constants import LANGFUSE_TRACER_NAME
from langfuse._client.propagation import (
    LANGFUSE_TRACE_ID_BAGGAGE_KEY,
    _get_langfuse_trace_id_from_baggage,
    _set_langfuse_trace_id_in_baggage,
)
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
    assert processor._span_export_expectation_by_id == {}


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
    assert processor._span_export_expectation_by_id == {}


def test_grandparent_baggage_claim_suppresses_child_through_filtered_parent(
    memory_exporter,
):
    tracer_provider, processor = _create_processor(memory_exporter)
    filtered_tracer = tracer_provider.get_tracer("requests")
    langfuse_tracer = _langfuse_tracer(tracer_provider)

    with langfuse_tracer.start_as_current_span("grandparent") as grandparent:
        context_with_claim = baggage.set_baggage(
            name=LANGFUSE_TRACE_ID_BAGGAGE_KEY,
            value=format(grandparent.context.trace_id, "032x"),
            context=otel_context_api.get_current(),
        )
        token = otel_context_api.attach(context_with_claim)
        try:
            with filtered_tracer.start_as_current_span("filtered-parent"):
                with langfuse_tracer.start_as_current_span("child"):
                    pass
        finally:
            otel_context_api.detach(token)

    processor.force_flush()

    spans = _get_spans_by_name(memory_exporter)

    assert "filtered-parent" not in spans
    assert (
        spans["grandparent"].attributes[LangfuseOtelSpanAttributes.IS_APP_ROOT] is True
    )
    assert LangfuseOtelSpanAttributes.IS_APP_ROOT not in spans["child"].attributes
    assert processor._span_export_expectation_by_id == {}


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
    assert processor._span_export_expectation_by_id == {}


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
    assert processor._span_export_expectation_by_id == {}


def test_local_baggage_claim_suppresses_child_even_when_parent_is_filtered(
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
    assert LangfuseOtelSpanAttributes.IS_APP_ROOT not in spans["child"].attributes
    assert processor._span_export_expectation_by_id == {}


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
    assert processor._span_export_expectation_by_id == {}


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


def test_blocked_instrumentation_scope_parent_marks_child_as_app_root(
    memory_exporter,
):
    tracer_provider, processor = _create_processor(
        memory_exporter,
        blocked_instrumentation_scopes=["blocked.scope"],
    )
    blocked_tracer = tracer_provider.get_tracer("blocked.scope")
    langfuse_tracer = _langfuse_tracer(tracer_provider)

    with blocked_tracer.start_as_current_span("blocked-parent"):
        with langfuse_tracer.start_as_current_span("child"):
            pass

    processor.force_flush()

    spans = _get_spans_by_name(memory_exporter)

    assert "blocked-parent" not in spans
    assert spans["child"].attributes[LangfuseOtelSpanAttributes.IS_APP_ROOT] is True
    assert processor._span_export_expectation_by_id == {}


def test_foreign_project_langfuse_parent_marks_child_as_app_root(memory_exporter):
    tracer_provider, processor = _create_processor(memory_exporter)
    foreign_tracer = tracer_provider.get_tracer(
        LANGFUSE_TRACER_NAME,
        "test",
        attributes={"public_key": "different-public-key"},
    )
    langfuse_tracer = _langfuse_tracer(tracer_provider)

    with foreign_tracer.start_as_current_span("foreign-parent"):
        with langfuse_tracer.start_as_current_span("child"):
            pass

    processor.force_flush()

    spans = _get_spans_by_name(memory_exporter)

    assert "foreign-parent" not in spans
    assert spans["child"].attributes[LangfuseOtelSpanAttributes.IS_APP_ROOT] is True
    assert processor._span_export_expectation_by_id == {}


def test_should_export_span_raising_does_not_mark_app_root(memory_exporter):
    def should_export_span(span: ReadableSpan) -> bool:
        raise RuntimeError("boom")

    tracer_provider, processor = _create_processor(
        memory_exporter,
        should_export_span=should_export_span,
    )
    langfuse_tracer = _langfuse_tracer(tracer_provider)

    with langfuse_tracer.start_as_current_span("root"):
        pass

    processor.force_flush()

    spans = _get_spans_by_name(memory_exporter)

    assert "root" not in spans
    assert processor._span_export_expectation_by_id == {}


def test_mark_app_root_candidate_exception_is_swallowed(memory_exporter, monkeypatch):
    tracer_provider, processor = _create_processor(memory_exporter)
    langfuse_tracer = _langfuse_tracer(tracer_provider)

    def raise_boom(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(processor, "_mark_app_root_candidate", raise_boom)

    with langfuse_tracer.start_as_current_span("root"):
        pass

    processor.force_flush()

    spans = _get_spans_by_name(memory_exporter)

    assert "root" in spans
    assert LangfuseOtelSpanAttributes.IS_APP_ROOT not in spans["root"].attributes


def test_concurrent_traces_keep_state_consistent(memory_exporter):
    tracer_provider, processor = _create_processor(memory_exporter)
    langfuse_tracer = _langfuse_tracer(tracer_provider)

    thread_count = 16
    spans_per_thread = 25
    barrier = threading.Barrier(thread_count)

    def worker(worker_id: int) -> None:
        barrier.wait()
        for span_index in range(spans_per_thread):
            with langfuse_tracer.start_as_current_span(
                f"root-{worker_id}-{span_index}"
            ):
                with langfuse_tracer.start_as_current_span(
                    f"child-{worker_id}-{span_index}"
                ):
                    pass

    with ThreadPoolExecutor(max_workers=thread_count) as pool:
        list(pool.map(worker, range(thread_count)))

    processor.force_flush()

    spans = _get_spans_by_name(memory_exporter)

    expected_total = thread_count * spans_per_thread
    root_spans = [span for name, span in spans.items() if name.startswith("root-")]
    child_spans = [span for name, span in spans.items() if name.startswith("child-")]

    assert len(root_spans) == expected_total
    assert len(child_spans) == expected_total
    assert all(
        span.attributes[LangfuseOtelSpanAttributes.IS_APP_ROOT] is True
        for span in root_spans
    )
    assert all(
        LangfuseOtelSpanAttributes.IS_APP_ROOT not in span.attributes
        for span in child_spans
    )
    assert processor._span_export_expectation_by_id == {}


def test_multiple_interleaved_traces_track_active_span_state_independently(
    memory_exporter,
):
    tracer_provider, processor = _create_processor(memory_exporter)
    langfuse_tracer = _langfuse_tracer(tracer_provider)

    trace_a_root = langfuse_tracer.start_span("trace-a-root")
    trace_a_ctx = trace_api.set_span_in_context(trace_a_root)
    trace_b_root = langfuse_tracer.start_span("trace-b-root")
    trace_b_ctx = trace_api.set_span_in_context(trace_b_root)

    assert len(processor._span_export_expectation_by_id) == 2

    trace_a_child = langfuse_tracer.start_span("trace-a-child", context=trace_a_ctx)
    trace_b_child = langfuse_tracer.start_span("trace-b-child", context=trace_b_ctx)

    assert len(processor._span_export_expectation_by_id) == 4

    trace_b_child.end()
    trace_b_root.end()

    assert len(processor._span_export_expectation_by_id) == 2

    trace_a_child.end()
    trace_a_root.end()

    processor.force_flush()

    spans = _get_spans_by_name(memory_exporter)

    assert (
        spans["trace-a-root"].attributes[LangfuseOtelSpanAttributes.IS_APP_ROOT] is True
    )
    assert (
        spans["trace-b-root"].attributes[LangfuseOtelSpanAttributes.IS_APP_ROOT] is True
    )
    assert (
        LangfuseOtelSpanAttributes.IS_APP_ROOT not in spans["trace-a-child"].attributes
    )
    assert (
        LangfuseOtelSpanAttributes.IS_APP_ROOT not in spans["trace-b-child"].attributes
    )
    assert processor._span_export_expectation_by_id == {}


def test_child_started_after_parent_end_is_marked_as_app_root(memory_exporter):
    tracer_provider, processor = _create_processor(memory_exporter)
    langfuse_tracer = _langfuse_tracer(tracer_provider)

    parent = langfuse_tracer.start_span("parent")
    parent_ctx = trace_api.set_span_in_context(parent)

    parent.end()

    child = langfuse_tracer.start_span("child", context=parent_ctx)
    child.end()

    processor.force_flush()

    spans = _get_spans_by_name(memory_exporter)

    assert spans["parent"].attributes[LangfuseOtelSpanAttributes.IS_APP_ROOT] is True
    assert spans["child"].attributes[LangfuseOtelSpanAttributes.IS_APP_ROOT] is True
    assert processor._span_export_expectation_by_id == {}


def test_set_langfuse_trace_id_in_baggage_sets_value():
    trace_id = "a" * 32
    context = _set_langfuse_trace_id_in_baggage(
        trace_id=trace_id,
        context=otel_context_api.Context(),
    )

    assert _get_langfuse_trace_id_from_baggage(context) == trace_id


def test_set_langfuse_trace_id_in_baggage_normalizes_case():
    context = _set_langfuse_trace_id_in_baggage(
        trace_id="ABCDEF" + "0" * 26,
        context=otel_context_api.Context(),
    )

    assert _get_langfuse_trace_id_from_baggage(context) == "abcdef" + "0" * 26


def test_set_langfuse_trace_id_in_baggage_is_idempotent_for_same_trace():
    trace_id = "a" * 32
    context = _set_langfuse_trace_id_in_baggage(
        trace_id=trace_id,
        context=otel_context_api.Context(),
    )

    same_context = _set_langfuse_trace_id_in_baggage(
        trace_id=trace_id,
        context=context,
    )

    assert same_context is context


def test_set_langfuse_trace_id_in_baggage_overwrites_for_different_trace():
    first = _set_langfuse_trace_id_in_baggage(
        trace_id="a" * 32,
        context=otel_context_api.Context(),
    )

    second = _set_langfuse_trace_id_in_baggage(
        trace_id="b" * 32,
        context=first,
    )

    assert second is not first
    assert _get_langfuse_trace_id_from_baggage(second) == "b" * 32


def _remote_parent_context(*, trace_id: int):
    span_context = SpanContext(
        trace_id=trace_id,
        span_id=int("a" * 16, 16),
        is_remote=True,
        trace_flags=TraceFlags(TraceFlags.SAMPLED),
    )

    return trace_api.set_span_in_context(NonRecordingSpan(span_context))
