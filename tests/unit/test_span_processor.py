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
    LANGFUSE_OTEL_MAX_BATCH_SIZE_BYTES,
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


class RecordingOTLPSpanExporter(NoOpSpanExporter):
    init_kwargs: dict = {}

    def __init__(self, *, endpoint=None, headers=None, timeout=None, **kwargs):
        type(self).init_kwargs = kwargs


class RecordingOTLPSpanExporterWithRequestLimit(RecordingOTLPSpanExporter):
    def __init__(
        self,
        *,
        endpoint=None,
        headers=None,
        timeout=None,
        max_request_size=None,
    ):
        super().__init__(max_request_size=max_request_size)


def _build_default_exporter_processor(monkeypatch, exporter_class):
    exporter_class.init_kwargs = {}
    monkeypatch.setattr(span_processor_module, "OTLPSpanExporter", exporter_class)
    return LangfuseSpanProcessor(
        public_key="pk-test",
        secret_key="sk-test",
        base_url="http://localhost:3000",
    )


def test_default_exporter_receives_configured_max_batch_size_bytes(monkeypatch):
    monkeypatch.setenv(LANGFUSE_OTEL_MAX_BATCH_SIZE_BYTES, " 1024 ")
    processor = _build_default_exporter_processor(
        monkeypatch, RecordingOTLPSpanExporterWithRequestLimit
    )

    try:
        assert RecordingOTLPSpanExporterWithRequestLimit.init_kwargs == {
            "max_request_size": 1024
        }
    finally:
        processor.shutdown()


def test_default_exporter_keeps_upstream_limit_when_env_unset(monkeypatch):
    monkeypatch.delenv(LANGFUSE_OTEL_MAX_BATCH_SIZE_BYTES, raising=False)
    processor = _build_default_exporter_processor(
        monkeypatch, RecordingOTLPSpanExporterWithRequestLimit
    )

    try:
        assert RecordingOTLPSpanExporterWithRequestLimit.init_kwargs == {
            "max_request_size": None
        }
    finally:
        processor.shutdown()


@pytest.mark.parametrize("raw_value", ["0", "-5", "abc", "1.5"])
def test_invalid_max_batch_size_bytes_falls_back_to_upstream_limit(
    monkeypatch, caplog, raw_value
):
    monkeypatch.setenv(LANGFUSE_OTEL_MAX_BATCH_SIZE_BYTES, raw_value)

    with caplog.at_level(logging.WARNING, logger="langfuse"):
        processor = _build_default_exporter_processor(
            monkeypatch, RecordingOTLPSpanExporterWithRequestLimit
        )

    try:
        assert RecordingOTLPSpanExporterWithRequestLimit.init_kwargs == {
            "max_request_size": None
        }
        assert LANGFUSE_OTEL_MAX_BATCH_SIZE_BYTES in caplog.text
    finally:
        processor.shutdown()


def test_max_batch_size_bytes_warns_when_exporter_lacks_request_limit(
    monkeypatch, caplog
):
    monkeypatch.setenv(LANGFUSE_OTEL_MAX_BATCH_SIZE_BYTES, "1024")

    with caplog.at_level(logging.WARNING, logger="langfuse"):
        processor = _build_default_exporter_processor(
            monkeypatch, RecordingOTLPSpanExporter
        )

    try:
        assert RecordingOTLPSpanExporter.init_kwargs == {}
        assert "opentelemetry-exporter-otlp-proto-http>=1.45.0" in caplog.text
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
