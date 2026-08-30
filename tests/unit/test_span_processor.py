import logging
from typing import Sequence
from unittest.mock import patch

import pytest
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

import langfuse._client.span_processor as span_processor_module
from langfuse._client.environment_variables import (
    LANGFUSE_FLUSH_AT,
    LANGFUSE_FLUSH_INTERVAL,
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


@pytest.fixture
def tracer_with_processor():
    processor = LangfuseSpanProcessor(
        public_key="pk-test",
        secret_key="sk-test",
        base_url="http://localhost:3000",
        span_exporter=NoOpSpanExporter(),
    )
    provider = TracerProvider()
    provider.add_span_processor(processor)
    yield provider.get_tracer("test-instrumentor")
    processor.shutdown()


@pytest.mark.parametrize(
    ("level", "expected_formatter_calls"),
    [(logging.WARNING, 0), (logging.DEBUG, 1)],
)
def test_on_end_formats_span_only_when_debug_enabled(
    caplog, tracer_with_processor, level, expected_formatter_calls
):
    caplog.set_level(level, logger="langfuse")

    with patch.object(
        span_processor_module, "span_formatter", return_value="{}"
    ) as span_formatter:
        # gen_ai.* attribute makes the span pass the default export filter
        with tracer_with_processor.start_as_current_span(
            "llm-call", attributes={"gen_ai.system": "test"}
        ):
            pass

    assert span_formatter.call_count == expected_formatter_calls
    assert ("Processing span name='llm-call'" in caplog.text) == bool(
        expected_formatter_calls
    )
