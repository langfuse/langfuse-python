from typing import Sequence

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

from langfuse._client.environment_variables import (
    LANGFUSE_FLUSH_AT,
    LANGFUSE_FLUSH_INTERVAL,
    LANGFUSE_MAX_QUEUE_SIZE,
)
from langfuse._client.span_processor import LangfuseSpanProcessor


class NoOpSpanExporter(SpanExporter):
    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        pass


def test_span_processor_uses_constructor_flush_settings_without_env(monkeypatch):
    monkeypatch.delenv(LANGFUSE_FLUSH_AT, raising=False)
    monkeypatch.delenv(LANGFUSE_FLUSH_INTERVAL, raising=False)
    processor = LangfuseSpanProcessor(
        public_key="pk-test",
        secret_key="sk-test",
        base_url="http://localhost:3000",
        flush_at=17,
        flush_interval=2.5,
        span_exporter=NoOpSpanExporter(),
    )

    try:
        assert processor._batch_processor._max_export_batch_size == 17
        assert processor._batch_processor._schedule_delay_millis == 2500
    finally:
        processor.shutdown()


def test_span_processor_uses_env_flush_settings_when_constructor_omits_them(
    monkeypatch,
):
    monkeypatch.setenv(LANGFUSE_FLUSH_AT, "19")
    monkeypatch.setenv(LANGFUSE_FLUSH_INTERVAL, "3.25")
    processor = LangfuseSpanProcessor(
        public_key="pk-test",
        secret_key="sk-test",
        base_url="http://localhost:3000",
        span_exporter=NoOpSpanExporter(),
    )

    try:
        assert processor._batch_processor._max_export_batch_size == 19
        assert processor._batch_processor._schedule_delay_millis == 3250
    finally:
        processor.shutdown()


def test_span_processor_uses_constructor_max_queue_size_without_env(monkeypatch):
    monkeypatch.delenv(LANGFUSE_MAX_QUEUE_SIZE, raising=False)
    processor = LangfuseSpanProcessor(
        public_key="pk-test",
        secret_key="sk-test",
        base_url="http://localhost:3000",
        max_queue_size=5000,
        span_exporter=NoOpSpanExporter(),
    )

    try:
        assert processor._batch_processor._max_queue_size == 5000
    finally:
        processor.shutdown()


def test_span_processor_uses_env_max_queue_size_when_constructor_omits_it(monkeypatch):
    monkeypatch.setenv(LANGFUSE_MAX_QUEUE_SIZE, "4096")
    processor = LangfuseSpanProcessor(
        public_key="pk-test",
        secret_key="sk-test",
        base_url="http://localhost:3000",
        span_exporter=NoOpSpanExporter(),
    )

    try:
        assert processor._batch_processor._max_queue_size == 4096
    finally:
        processor.shutdown()


def test_span_processor_defaults_to_otel_max_queue_size(monkeypatch):
    # When neither the constructor arg nor the Langfuse env var is set, the queue
    # size falls back to OpenTelemetry's default (OTEL_BSP_MAX_QUEUE_SIZE, 2048),
    # keeping behavior unchanged for existing users.
    monkeypatch.delenv(LANGFUSE_MAX_QUEUE_SIZE, raising=False)
    monkeypatch.delenv("OTEL_BSP_MAX_QUEUE_SIZE", raising=False)
    processor = LangfuseSpanProcessor(
        public_key="pk-test",
        secret_key="sk-test",
        base_url="http://localhost:3000",
        span_exporter=NoOpSpanExporter(),
    )

    try:
        assert processor._batch_processor._max_queue_size == 2048
    finally:
        processor.shutdown()
