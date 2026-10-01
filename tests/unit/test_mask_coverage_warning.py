import logging
from typing import Any, Optional

import pytest
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SpanExporter

from langfuse._client.client import Langfuse
from langfuse._client.constants import LANGFUSE_TRACER_NAME
from langfuse._client.span_processor import LangfuseSpanProcessor
from langfuse.types import MaskFunction, MaskOtelSpansFunction, MaskOtelSpansParams

COVERAGE_WARNING = (
    "does not inspect attributes set by other OpenTelemetry instrumentations"
)
RAW_PROMPT = "raw prompt that names a client"


def _mask(*, data: Any, **kwargs: Any) -> Any:
    return "masked"


def _tracer_provider(
    *,
    exporter: SpanExporter,
    mask: Optional[MaskFunction] = None,
    mask_otel_spans: Optional[MaskOtelSpansFunction] = None,
) -> TracerProvider:
    provider = TracerProvider(resource=Resource.create({"service.name": "test"}))
    provider.add_span_processor(
        LangfuseSpanProcessor(
            public_key="test-public-key",
            secret_key="test-secret-key",
            base_url="http://localhost:3000",
            flush_at=10,
            flush_interval=1,
            span_exporter=exporter,
            mask=mask,
            mask_otel_spans=mask_otel_spans,
        )
    )

    return provider


def _coverage_warnings(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [
        record for record in caplog.records if COVERAGE_WARNING in record.getMessage()
    ]


def test_mask_warns_once_when_third_party_spans_carry_input_attributes(
    memory_exporter, caplog
):
    provider = _tracer_provider(exporter=memory_exporter, mask=_mask)
    tracer = provider.get_tracer("openinference.instrumentation.openai")

    with caplog.at_level(logging.WARNING, logger="langfuse"):
        for name in ("first-llm-call", "second-llm-call"):
            with tracer.start_as_current_span(name) as span:
                span.set_attribute("input.value", RAW_PROMPT)
                span.set_attribute("llm.token_count.prompt", 12)

        provider.force_flush()

    warnings = _coverage_warnings(caplog)

    assert len(warnings) == 1
    assert "input.value" in warnings[0].getMessage()
    assert "mask_otel_spans" in warnings[0].getMessage()
    assert "llm.token_count.prompt" not in warnings[0].getMessage()


def test_mask_coverage_warning_never_logs_attribute_values(memory_exporter, caplog):
    provider = _tracer_provider(exporter=memory_exporter, mask=_mask)
    tracer = provider.get_tracer("openinference.instrumentation.openai")

    with caplog.at_level(logging.WARNING, logger="langfuse"):
        with tracer.start_as_current_span("third-party-llm-call") as span:
            span.set_attribute("input.value", RAW_PROMPT)

        provider.force_flush()

    assert _coverage_warnings(caplog)
    assert all(RAW_PROMPT not in record.getMessage() for record in caplog.records)


def test_mask_coverage_warning_does_not_change_exported_attributes(memory_exporter):
    provider = _tracer_provider(exporter=memory_exporter, mask=_mask)
    tracer = provider.get_tracer("openinference.instrumentation.openai")

    with tracer.start_as_current_span("third-party-llm-call") as span:
        span.set_attribute("input.value", RAW_PROMPT)

    provider.force_flush()

    exported_span = memory_exporter.get_finished_spans()[0]

    assert exported_span.attributes["input.value"] == RAW_PROMPT


def test_mask_warns_for_third_party_attributes_on_langfuse_tracer_spans(
    memory_exporter, caplog
):
    # Instrumentations wired through the Langfuse tracer, e.g.
    # openlit.init(tracer=langfuse._otel_tracer), keep the langfuse-sdk scope,
    # so detection has to key on attributes rather than instrumentation scope.
    provider = _tracer_provider(exporter=memory_exporter, mask=_mask)
    tracer = provider.get_tracer(
        LANGFUSE_TRACER_NAME, attributes={"public_key": "test-public-key"}
    )

    with caplog.at_level(logging.WARNING, logger="langfuse"):
        with tracer.start_as_current_span("third-party-llm-call") as span:
            span.set_attribute("input.value", RAW_PROMPT)

        provider.force_flush()

    assert len(_coverage_warnings(caplog)) == 1


def test_mask_does_not_warn_for_native_langfuse_input_attributes(
    memory_exporter, caplog
):
    provider = _tracer_provider(exporter=memory_exporter, mask=_mask)
    tracer = provider.get_tracer(
        LANGFUSE_TRACER_NAME, attributes={"public_key": "test-public-key"}
    )

    with caplog.at_level(logging.WARNING, logger="langfuse"):
        with tracer.start_as_current_span("langfuse-span") as span:
            span.set_attribute("langfuse.observation.type", "span")
            span.set_attribute("langfuse.observation.input", '"masked"')

        provider.force_flush()

    assert _coverage_warnings(caplog) == []


def test_no_coverage_warning_when_mask_otel_spans_is_configured(
    memory_exporter, caplog
):
    def mask_otel_spans(*, params: MaskOtelSpansParams):
        return None

    provider = _tracer_provider(
        exporter=memory_exporter, mask=_mask, mask_otel_spans=mask_otel_spans
    )
    tracer = provider.get_tracer("openinference.instrumentation.openai")

    with caplog.at_level(logging.WARNING, logger="langfuse"):
        with tracer.start_as_current_span("third-party-llm-call") as span:
            span.set_attribute("input.value", RAW_PROMPT)

        provider.force_flush()

    assert _coverage_warnings(caplog) == []


def test_no_coverage_warning_when_mask_is_not_configured(memory_exporter, caplog):
    provider = _tracer_provider(exporter=memory_exporter)
    tracer = provider.get_tracer("openinference.instrumentation.openai")

    with caplog.at_level(logging.WARNING, logger="langfuse"):
        with tracer.start_as_current_span("third-party-llm-call") as span:
            span.set_attribute("input.value", RAW_PROMPT)

        provider.force_flush()

    assert _coverage_warnings(caplog) == []


def test_langfuse_client_mask_reaches_export_stage_warning(memory_exporter, caplog):
    tracer_provider = TracerProvider(resource=Resource.create({"service.name": "test"}))
    client = Langfuse(
        public_key="test-public-key",
        secret_key="test-secret-key",
        base_url="http://localhost:3000",
        mask=_mask,
        span_exporter=memory_exporter,
        tracer_provider=tracer_provider,
    )
    tracer = tracer_provider.get_tracer("openinference.instrumentation.openai")

    with caplog.at_level(logging.WARNING, logger="langfuse"):
        with tracer.start_as_current_span("third-party-llm-call") as span:
            span.set_attribute("input.value", RAW_PROMPT)

        client.flush()

    client.shutdown()

    assert len(_coverage_warnings(caplog)) == 1
