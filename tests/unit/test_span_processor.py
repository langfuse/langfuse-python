import logging
from typing import Sequence
from unittest.mock import Mock, patch

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

import langfuse._client.span_processor as span_processor_module
from langfuse._client.environment_variables import (
    LANGFUSE_FLUSH_AT,
    LANGFUSE_FLUSH_INTERVAL,
)
from langfuse._client.span_processor import LangfuseSpanProcessor
from langfuse.logger import langfuse_logger


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


def _make_processor() -> LangfuseSpanProcessor:
    return LangfuseSpanProcessor(
        public_key="pk-test",
        secret_key="sk-test",
        base_url="http://localhost:3000",
        span_exporter=NoOpSpanExporter(),
    )


def _reach_debug_log_line(processor, monkeypatch):
    """Force on_end past its early-return guards to the debug-log statement."""
    monkeypatch.setattr(processor, "_is_langfuse_project_span", lambda span: True)
    monkeypatch.setattr(
        processor, "_is_blocked_instrumentation_scope", lambda span: False
    )
    monkeypatch.setattr(processor, "_should_export_span", lambda span: True)
    monkeypatch.setattr(processor, "_cleanup_app_root_state", lambda span: None)
    # Isolate from the batch exporter's on_end
    monkeypatch.setattr(
        span_processor_module.BatchSpanProcessor,
        "on_end",
        lambda self, span: None,
    )


def test_on_end_does_not_call_span_formatter_when_debug_disabled(monkeypatch):
    processor = _make_processor()
    _reach_debug_log_line(processor, monkeypatch)
    span = Mock()
    span._name = "test-span"

    original_level = langfuse_logger.level
    try:
        langfuse_logger.setLevel(logging.WARNING)
        with patch.object(span_processor_module, "span_formatter") as mock_formatter:
            processor.on_end(span)
        mock_formatter.assert_not_called()
    finally:
        langfuse_logger.setLevel(original_level)
        processor.shutdown()


def test_on_end_calls_span_formatter_when_debug_enabled(monkeypatch):
    processor = _make_processor()
    _reach_debug_log_line(processor, monkeypatch)
    span = Mock()
    span._name = "test-span"

    original_level = langfuse_logger.level
    try:
        langfuse_logger.setLevel(logging.DEBUG)
        with patch.object(span_processor_module, "span_formatter") as mock_formatter:
            processor.on_end(span)
        mock_formatter.assert_called_once()
    finally:
        langfuse_logger.setLevel(original_level)
        processor.shutdown()
