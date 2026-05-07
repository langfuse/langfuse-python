import base64
import json
from queue import Queue
from types import SimpleNamespace
from typing import Sequence
from unittest.mock import Mock

from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

import langfuse._client.span_exporter as span_exporter_module
from langfuse._client.constants import LANGFUSE_TRACER_NAME
from langfuse._client.span_processor import LangfuseSpanProcessor
from langfuse._task_manager.media_manager import MediaManager
from langfuse.types import (
    MaskOtelSpansParams,
    MaskOtelSpansResult,
    OtelSpanIdentifier,
    OtelSpanPatch,
)


class InMemorySpanExporter(SpanExporter):
    def __init__(self) -> None:
        self._finished_spans: list[ReadableSpan] = []

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        self._finished_spans.extend(spans)
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        pass

    def get_finished_spans(self) -> list[ReadableSpan]:
        return list(self._finished_spans)


def _media_manager() -> tuple[MediaManager, Queue]:
    queue: Queue = Queue()

    return (
        MediaManager(
            api_client=SimpleNamespace(media=Mock()),
            httpx_client=Mock(),
            media_upload_queue=queue,
        ),
        queue,
    )


def _tracer_provider(
    *,
    exporter: InMemorySpanExporter,
    media_manager: MediaManager,
    mask_otel_spans=None,
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
            media_manager=media_manager,
            mask_otel_spans=mask_otel_spans,
        )
    )

    return provider


def test_mask_otel_spans_receives_post_media_batch_and_applies_sparse_patch():
    exporter = InMemorySpanExporter()
    media_manager, media_queue = _media_manager()
    seen_params: list[MaskOtelSpansParams] = []
    image_base64 = base64.b64encode(b"image-bytes").decode("utf-8")

    def mask_otel_spans(*, params: MaskOtelSpansParams):
        seen_params.append(params)

        target_identifier = _find_identifier_by_attribute(params, "gen_ai.prompt")
        payload = json.loads(
            params.spans[target_identifier].attributes["gen_ai.prompt"]
        )

        assert len(params.spans) == 2
        assert payload["inline_data"]["data"].startswith("@@@langfuseMedia:")

        return MaskOtelSpansResult(
            span_patches={
                target_identifier: OtelSpanPatch(
                    set_attributes={
                        "gen_ai.request.model": "masked-model",
                        "langfuse.masking.applied": True,
                    }
                )
            }
        )

    provider = _tracer_provider(
        exporter=exporter,
        media_manager=media_manager,
        mask_otel_spans=mask_otel_spans,
    )
    tracer = provider.get_tracer("openinference.instrumentation.openai")

    with tracer.start_as_current_span("third-party-media-span") as span:
        span.set_attribute("gen_ai.request.model", "gpt-4o")
        span.set_attribute(
            "gen_ai.prompt",
            json.dumps(
                {
                    "inline_data": {
                        "mime_type": "image/jpeg",
                        "data": image_base64,
                    }
                }
            ),
        )

    with tracer.start_as_current_span("third-party-unchanged-span") as span:
        span.set_attribute("gen_ai.request.model", "gpt-4o-mini")

    provider.force_flush()

    exported_spans = exporter.get_finished_spans()
    exported_by_name = {span.name: span for span in exported_spans}
    exported_payload = json.loads(
        exported_by_name["third-party-media-span"].attributes["gen_ai.prompt"]
    )

    assert len(seen_params) == 1
    assert len(exported_spans) == 2
    assert (
        exported_by_name["third-party-media-span"].attributes["gen_ai.request.model"]
        == "masked-model"
    )
    assert (
        exported_by_name["third-party-media-span"].attributes[
            "langfuse.masking.applied"
        ]
        is True
    )
    assert (
        exported_by_name["third-party-unchanged-span"].attributes[
            "gen_ai.request.model"
        ]
        == "gpt-4o-mini"
    )
    assert exported_payload["inline_data"]["data"].startswith("@@@langfuseMedia:")
    assert not media_queue.empty()


def test_export_stage_media_prefilter_skips_json_without_media_hints(monkeypatch):
    exporter = InMemorySpanExporter()
    media_manager, media_queue = _media_manager()
    json_load_calls = 0
    original_json_loads = json.loads

    def count_json_loads(*args, **kwargs):
        nonlocal json_load_calls
        json_load_calls += 1
        return original_json_loads(*args, **kwargs)

    monkeypatch.setattr(span_exporter_module.json, "loads", count_json_loads)

    transforming_exporter = span_exporter_module.LangfuseTransformingSpanExporter(
        exporter=exporter,
        media_manager=media_manager,
        mask_otel_spans=None,
    )
    span = SimpleNamespace(
        name="third-party-json-without-media",
        context=SimpleNamespace(trace_id=1, span_id=2),
    )
    payload = '{"messages": [{"role": "user", "content": "hello"}]}'

    result = transforming_exporter._process_media_string(
        span=span,
        attribute_key="gen_ai.prompt",
        value=payload,
    )

    assert json_load_calls == 0
    assert result == payload
    assert media_queue.empty()


def test_export_stage_media_processes_direct_data_uri_string():
    exporter = InMemorySpanExporter()
    media_manager, media_queue = _media_manager()
    image_base64 = base64.b64encode(b"image-bytes").decode("utf-8")

    provider = _tracer_provider(exporter=exporter, media_manager=media_manager)
    tracer = provider.get_tracer("openinference.instrumentation.openai")

    with tracer.start_as_current_span("third-party-direct-media-span") as span:
        span.set_attribute("gen_ai.prompt", f"data:image/jpeg;base64,{image_base64}")

    provider.force_flush()

    exported_span = exporter.get_finished_spans()[0]

    assert exported_span.attributes["gen_ai.prompt"].startswith("@@@langfuseMedia:")
    assert not media_queue.empty()


def test_mask_otel_spans_runs_for_langfuse_sdk_spans():
    exporter = InMemorySpanExporter()
    media_manager, _ = _media_manager()

    def mask_otel_spans(*, params: MaskOtelSpansParams):
        identifier = next(iter(params.spans))

        return MaskOtelSpansResult(
            span_patches={
                identifier: OtelSpanPatch(set_attributes={"secret": "masked"})
            }
        )

    provider = _tracer_provider(
        exporter=exporter,
        media_manager=media_manager,
        mask_otel_spans=mask_otel_spans,
    )
    tracer = provider.get_tracer(
        LANGFUSE_TRACER_NAME, attributes={"public_key": "test-public-key"}
    )

    with tracer.start_as_current_span("langfuse-span") as span:
        span.set_attribute("langfuse.observation.type", "span")
        span.set_attribute("secret", "raw")

    provider.force_flush()

    exported_span = exporter.get_finished_spans()[0]

    assert exported_span.attributes["secret"] == "masked"


def test_mask_otel_spans_exception_drops_batch():
    exporter = InMemorySpanExporter()
    media_manager, _ = _media_manager()

    def mask_otel_spans(*, params: MaskOtelSpansParams):
        raise RuntimeError("mask failed")

    provider = _tracer_provider(
        exporter=exporter,
        media_manager=media_manager,
        mask_otel_spans=mask_otel_spans,
    )
    tracer = provider.get_tracer("openinference.instrumentation.openai")

    with tracer.start_as_current_span("first-span") as span:
        span.set_attribute("gen_ai.request.model", "gpt-4o")

    with tracer.start_as_current_span("second-span") as span:
        span.set_attribute("gen_ai.request.model", "gpt-4o-mini")

    provider.force_flush()

    assert exporter.get_finished_spans() == []


def test_mask_otel_spans_invalid_result_drops_batch():
    exporter = InMemorySpanExporter()
    media_manager, _ = _media_manager()

    def mask_otel_spans(*, params: MaskOtelSpansParams):
        return {"span_patches": {}}

    provider = _tracer_provider(
        exporter=exporter,
        media_manager=media_manager,
        mask_otel_spans=mask_otel_spans,
    )
    tracer = provider.get_tracer("openinference.instrumentation.openai")

    with tracer.start_as_current_span("third-party-span") as span:
        span.set_attribute("gen_ai.request.model", "gpt-4o")

    provider.force_flush()

    assert exporter.get_finished_spans() == []


def test_mask_otel_spans_unknown_identifier_drops_batch():
    exporter = InMemorySpanExporter()
    media_manager, _ = _media_manager()
    unknown_identifier = OtelSpanIdentifier(trace_id="1" * 32, span_id="2" * 16)

    def mask_otel_spans(*, params: MaskOtelSpansParams):
        return MaskOtelSpansResult(
            span_patches={
                unknown_identifier: OtelSpanPatch(set_attributes={"secret": "masked"})
            }
        )

    provider = _tracer_provider(
        exporter=exporter,
        media_manager=media_manager,
        mask_otel_spans=mask_otel_spans,
    )
    tracer = provider.get_tracer("openinference.instrumentation.openai")

    with tracer.start_as_current_span("third-party-span") as span:
        span.set_attribute("gen_ai.request.model", "gpt-4o")

    provider.force_flush()

    assert exporter.get_finished_spans() == []


def test_mask_otel_spans_invalid_patch_drops_only_that_span():
    exporter = InMemorySpanExporter()
    media_manager, _ = _media_manager()

    def mask_otel_spans(*, params: MaskOtelSpansParams):
        target_identifier = _find_identifier_by_name(params, "drop-me")

        return MaskOtelSpansResult(
            span_patches={
                target_identifier: {"set_attributes": {"secret": "masked"}},
            }
        )

    provider = _tracer_provider(
        exporter=exporter,
        media_manager=media_manager,
        mask_otel_spans=mask_otel_spans,
    )
    tracer = provider.get_tracer("openinference.instrumentation.openai")

    with tracer.start_as_current_span("drop-me") as span:
        span.set_attribute("gen_ai.request.model", "gpt-4o")

    with tracer.start_as_current_span("keep-me") as span:
        span.set_attribute("gen_ai.request.model", "gpt-4o-mini")

    provider.force_flush()

    exported_spans = exporter.get_finished_spans()

    assert [span.name for span in exported_spans] == ["keep-me"]


def test_mask_otel_spans_invalid_set_value_deletes_attribute():
    exporter = InMemorySpanExporter()
    media_manager, _ = _media_manager()

    def mask_otel_spans(*, params: MaskOtelSpansParams):
        identifier = next(iter(params.spans))

        return MaskOtelSpansResult(
            span_patches={
                identifier: OtelSpanPatch(
                    set_attributes={
                        "secret": object(),
                        "langfuse.masking.applied": True,
                    }
                )
            }
        )

    provider = _tracer_provider(
        exporter=exporter,
        media_manager=media_manager,
        mask_otel_spans=mask_otel_spans,
    )
    tracer = provider.get_tracer("openinference.instrumentation.openai")

    with tracer.start_as_current_span("third-party-span") as span:
        span.set_attribute("gen_ai.request.model", "gpt-4o")
        span.set_attribute("secret", "raw")

    provider.force_flush()

    exported_span = exporter.get_finished_spans()[0]

    assert "secret" not in exported_span.attributes
    assert exported_span.attributes["gen_ai.request.model"] == "gpt-4o"
    assert exported_span.attributes["langfuse.masking.applied"] is True


def test_mask_otel_spans_set_wins_when_key_is_deleted_and_set():
    exporter = InMemorySpanExporter()
    media_manager, _ = _media_manager()

    def mask_otel_spans(*, params: MaskOtelSpansParams):
        identifier = next(iter(params.spans))

        return MaskOtelSpansResult(
            span_patches={
                identifier: OtelSpanPatch(
                    delete_attributes=["secret"],
                    set_attributes={"secret": "masked"},
                )
            }
        )

    provider = _tracer_provider(
        exporter=exporter,
        media_manager=media_manager,
        mask_otel_spans=mask_otel_spans,
    )
    tracer = provider.get_tracer("openinference.instrumentation.openai")

    with tracer.start_as_current_span("third-party-span") as span:
        span.set_attribute("gen_ai.request.model", "gpt-4o")
        span.set_attribute("secret", "raw")

    provider.force_flush()

    exported_span = exporter.get_finished_spans()[0]

    assert exported_span.attributes["secret"] == "masked"


def _find_identifier_by_attribute(
    params: MaskOtelSpansParams, attribute_key: str
) -> OtelSpanIdentifier:
    return next(
        identifier
        for identifier, span_data in params.spans.items()
        if attribute_key in span_data.attributes
    )


def _find_identifier_by_name(
    params: MaskOtelSpansParams, name: str
) -> OtelSpanIdentifier:
    return next(
        identifier
        for identifier, span_data in params.spans.items()
        if span_data.name == name
    )
