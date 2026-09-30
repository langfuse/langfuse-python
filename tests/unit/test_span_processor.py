import logging
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import List, Sequence
from unittest.mock import patch

import pytest
from opentelemetry.exporter.otlp.proto.common.trace_encoder import encode_spans
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import (
    SimpleSpanProcessor,
    SpanExporter,
    SpanExportResult,
)
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

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


class _RecordingOTLPHandler(BaseHTTPRequestHandler):
    received_body_sizes: List[int]

    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        self.received_body_sizes.append(len(body))
        self.send_response(200)
        self.end_headers()

    def log_message(self, *args):
        pass


@pytest.fixture
def otlp_http_server():
    received_body_sizes: List[int] = []
    handler = type(
        "Handler",
        (_RecordingOTLPHandler,),
        {"received_body_sizes": received_body_sizes},
    )
    server = HTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    yield f"http://127.0.0.1:{server.server_port}", received_body_sizes

    server.shutdown()
    server.server_close()


def _finished_spans(payload: str) -> List[ReadableSpan]:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    with provider.get_tracer("test").start_as_current_span("span") as span:
        span.set_attribute("input", payload)
    provider.shutdown()

    return list(exporter.get_finished_spans())


def _serialized_request_size(spans: List[ReadableSpan]) -> int:
    return len(encode_spans(spans).SerializePartialToString())


def _default_exporter_processor(base_url: str) -> LangfuseSpanProcessor:
    return LangfuseSpanProcessor(
        public_key="pk-test",
        secret_key="sk-test",
        base_url=base_url,
    )


@pytest.mark.parametrize(
    ("limit_offset", "expected_result"),
    [
        (-1, SpanExportResult.FAILURE),
        (0, SpanExportResult.SUCCESS),
        (1, SpanExportResult.SUCCESS),
    ],
)
def test_default_exporter_enforces_max_batch_size_bytes_at_boundary(
    monkeypatch, otlp_http_server, limit_offset, expected_result
):
    base_url, received_body_sizes = otlp_http_server
    spans = _finished_spans("x" * 1_000)
    request_size = _serialized_request_size(spans)
    monkeypatch.setenv(
        LANGFUSE_OTEL_MAX_BATCH_SIZE_BYTES, str(request_size + limit_offset)
    )
    processor = _default_exporter_processor(base_url)

    try:
        result = processor._batch_processor._exporter.export(spans)
    finally:
        processor.shutdown()

    assert result == expected_result
    expected_requests = (
        [] if expected_result == SpanExportResult.FAILURE else [request_size]
    )
    assert received_body_sizes == expected_requests


def test_oversized_batch_is_dropped_on_flush_without_blocking_later_batches(
    monkeypatch, caplog, otlp_http_server
):
    base_url, received_body_sizes = otlp_http_server
    oversized_spans = _finished_spans("secret-payload" * 1_000)
    small_spans = _finished_spans("ok")
    monkeypatch.setenv(
        LANGFUSE_OTEL_MAX_BATCH_SIZE_BYTES,
        str(_serialized_request_size(oversized_spans) - 1),
    )
    processor = _default_exporter_processor(base_url)

    try:
        with caplog.at_level(logging.WARNING):
            super(LangfuseSpanProcessor, processor).on_end(oversized_spans[0])
            assert processor.force_flush()

        assert received_body_sizes == []
        assert "Dropping span batch" in caplog.text
        assert "secret-payload" not in caplog.text

        super(LangfuseSpanProcessor, processor).on_end(small_spans[0])
        assert processor.force_flush()
    finally:
        processor.shutdown()

    assert received_body_sizes == [_serialized_request_size(small_spans)]


def test_default_exporter_uses_64_mib_limit_when_env_unset(monkeypatch):
    monkeypatch.delenv(LANGFUSE_OTEL_MAX_BATCH_SIZE_BYTES, raising=False)
    processor = _default_exporter_processor("http://localhost:3000")

    try:
        exporter = processor._batch_processor._exporter
        assert exporter._max_request_size == 64 * 1024 * 1024
    finally:
        processor.shutdown()


@pytest.mark.parametrize("raw_value", ["0", "-5", "abc", "1.5"])
def test_invalid_max_batch_size_bytes_falls_back_to_default_limit(
    monkeypatch, caplog, raw_value
):
    monkeypatch.setenv(LANGFUSE_OTEL_MAX_BATCH_SIZE_BYTES, raw_value)

    with caplog.at_level(logging.WARNING, logger="langfuse"):
        processor = _default_exporter_processor("http://localhost:3000")

    try:
        exporter = processor._batch_processor._exporter
        assert exporter._max_request_size == 64 * 1024 * 1024
        assert LANGFUSE_OTEL_MAX_BATCH_SIZE_BYTES in caplog.text
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
