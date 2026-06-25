from typing import Sequence

from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

from langfuse import propagate_attributes
from langfuse._client.attributes import LangfuseOtelSpanAttributes
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


class InMemorySpanExporter(SpanExporter):
    def __init__(self) -> None:
        self.spans: list[ReadableSpan] = []

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        self.spans.extend(spans)

        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        pass


def _export_third_party_span(
    *,
    processor_environment: str,
    span_attributes: dict[str, str] | None = None,
    resource_attributes: dict[str, str] | None = None,
    propagated_environment: str | None = None,
) -> ReadableSpan:
    exporter = InMemorySpanExporter()
    provider = TracerProvider(
        resource=Resource.create(
            {"service.name": "test", **(resource_attributes or {})}
        )
    )
    processor = LangfuseSpanProcessor(
        public_key="pk-test",
        secret_key="sk-test",
        base_url="http://localhost:3000",
        flush_at=10,
        flush_interval=1,
        span_exporter=exporter,
        environment=processor_environment,
        should_export_span=lambda span: True,
    )
    provider.add_span_processor(processor)

    try:
        tracer = provider.get_tracer("third-party.instrumentation", "1.0.0")

        if propagated_environment is None:
            with tracer.start_as_current_span(
                "third-party-span", attributes=span_attributes
            ):
                pass
        else:
            with propagate_attributes(environment=propagated_environment):
                with tracer.start_as_current_span(
                    "third-party-span", attributes=span_attributes
                ):
                    pass

        provider.force_flush()

        assert len(exporter.spans) == 1

        return exporter.spans[0]
    finally:
        provider.shutdown()


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


def test_span_processor_applies_environment_to_third_party_spans():
    span = _export_third_party_span(processor_environment="proxy-prod")

    assert span.attributes is not None
    assert span.attributes[LangfuseOtelSpanAttributes.ENVIRONMENT] == "proxy-prod"


def test_span_processor_prefers_propagated_environment_for_third_party_spans():
    span = _export_third_party_span(
        processor_environment="proxy-prod",
        propagated_environment="staging",
    )

    assert span.attributes is not None
    assert span.attributes[LangfuseOtelSpanAttributes.ENVIRONMENT] == "staging"


def test_span_processor_preserves_explicit_third_party_span_environment():
    span = _export_third_party_span(
        processor_environment="proxy-prod",
        span_attributes={LangfuseOtelSpanAttributes.ENVIRONMENT: "span-env"},
    )

    assert span.attributes is not None
    assert span.attributes[LangfuseOtelSpanAttributes.ENVIRONMENT] == "span-env"


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
