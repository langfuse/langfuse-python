"""A flush_at larger than the OpenTelemetry default queue size must be usable."""

from typing import Optional, Sequence

import pytest
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from langfuse import Langfuse
from langfuse._client.resource_manager import LangfuseResourceManager
from langfuse._client.span_processor import LangfuseSpanProcessor


class NoOpSpanExporter(SpanExporter):
    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        pass


def _build_client(**kwargs) -> Langfuse:
    """Build a client with its own TracerProvider.

    Without this the processor is added to the globally registered provider, where
    processors from earlier clients are still attached.
    """
    with LangfuseResourceManager._lock:
        LangfuseResourceManager._instances.clear()

    return Langfuse(
        public_key="pk-flush-at",
        secret_key="sk-flush-at",
        tracer_provider=TracerProvider(),
        **kwargs,
    )


def _queue_capacity(client: Langfuse) -> Optional[int]:
    """Read the queue capacity of this client's span processor.

    Returns None when the OpenTelemetry internals this relies on are not present,
    so the caller can skip instead of failing on an upstream refactor.
    """
    provider = client._resources.tracer_provider
    processors = [
        processor
        for processor in provider._active_span_processor._span_processors
        if isinstance(processor, LangfuseSpanProcessor)
    ]

    assert len(processors) == 1, "the client must own its TracerProvider"

    queue = getattr(processors[0], "_queue", None) or getattr(
        processors[0], "_batch_processor", None
    )
    target = getattr(queue, "_queue", queue)
    capacity = getattr(target, "maxlen", None) or getattr(target, "capacity", None)

    return capacity if isinstance(capacity, int) else None


@pytest.fixture
def isolated_client():
    client = _build_client(span_exporter=NoOpSpanExporter(), flush_at=5000)

    yield client

    client.shutdown()


@pytest.mark.unit
def test_flush_at_above_the_default_queue_size_is_accepted(isolated_client):
    """Configuring a large flush_at must not fail client construction."""
    assert isolated_client is not None


@pytest.mark.unit
def test_client_with_large_flush_at_can_flush(isolated_client):
    isolated_client.flush()


@pytest.mark.unit
def test_flush_at_uses_the_effective_default_queue_size(monkeypatch):
    """A smaller OTEL_BSP_MAX_QUEUE_SIZE must still be widened when flush_at needs it."""
    monkeypatch.setenv("OTEL_BSP_MAX_QUEUE_SIZE", "1000")

    client = _build_client(span_exporter=NoOpSpanExporter(), flush_at=1500)

    try:
        capacity = _queue_capacity(client)

        if capacity is None:
            pytest.skip("OpenTelemetry no longer exposes the queue capacity")

        assert capacity == 1500
    finally:
        client.shutdown()


@pytest.mark.unit
def test_a_larger_default_queue_size_is_not_shrunk(monkeypatch):
    """Widening the batch must not quietly narrow an already larger queue."""
    monkeypatch.setenv("OTEL_BSP_MAX_QUEUE_SIZE", "5000")

    client = _build_client(span_exporter=NoOpSpanExporter(), flush_at=3000)

    try:
        capacity = _queue_capacity(client)

        if capacity is None:
            pytest.skip("OpenTelemetry no longer exposes the queue capacity")

        assert capacity == 5000
    finally:
        client.shutdown()


@pytest.mark.unit
def test_spans_are_still_exported_with_a_large_flush_at():
    """A large flush_at must not silently drop or hold back finished spans."""
    exporter = InMemorySpanExporter()

    client = _build_client(span_exporter=exporter, flush_at=5000)

    try:
        client.start_observation(name="flush-at-span").end()

        client.flush()

        span_names = [span.name for span in exporter.get_finished_spans()]
        assert "flush-at-span" in span_names
    finally:
        client.shutdown()
