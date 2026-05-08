import base64
import json
import logging
from queue import Queue
from types import SimpleNamespace
from typing import Sequence
from unittest.mock import Mock

import pytest
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

import langfuse._client.span_exporter as span_exporter_module
from langfuse._client.constants import LANGFUSE_TRACER_NAME
from langfuse._client.span_processor import LangfuseSpanProcessor
from langfuse._task_manager.media_manager import MediaManager
from langfuse._utils.serializer import EventSerializer
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
    resource_attributes=None,
    should_export_span=None,
) -> TracerProvider:
    provider = TracerProvider(
        resource=Resource.create(
            {"service.name": "test", **(resource_attributes or {})}
        )
    )
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
            should_export_span=should_export_span,
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


def test_export_stage_media_processes_string_sequence_attributes():
    exporter = InMemorySpanExporter()
    media_manager, media_queue = _media_manager()
    image_base64 = base64.b64encode(b"image-bytes").decode("utf-8")

    provider = _tracer_provider(exporter=exporter, media_manager=media_manager)
    tracer = provider.get_tracer("openinference.instrumentation.openai")

    inline_data_payload = json.dumps(
        {
            "inline_data": {
                "mime_type": "image/png",
                "data": image_base64,
            }
        }
    )

    with tracer.start_as_current_span("third-party-sequence-media-span") as span:
        span.set_attribute(
            "gen_ai.prompt",
            [
                f"data:image/jpeg;base64,{image_base64}",
                inline_data_payload,
                "plain text",
            ],
        )

    provider.force_flush()

    exported_span = exporter.get_finished_spans()[0]
    exported_sequence = exported_span.attributes["gen_ai.prompt"]
    exported_payload = json.loads(exported_sequence[1])

    assert isinstance(exported_sequence, tuple)
    assert exported_sequence[0].startswith("@@@langfuseMedia:")
    assert exported_payload["inline_data"]["data"].startswith("@@@langfuseMedia:")
    assert exported_sequence[2] == "plain text"
    assert media_queue.qsize() == 2


@pytest.mark.parametrize(
    ("attribute_key", "expected_field"),
    [
        ("langfuse.trace.input", "input"),
        ("langfuse.observation.input", "input"),
        ("ai.prompt.messages", "input"),
        ("gcp.vertex.agent.tool_call_args", "input"),
        ("lk.chat_ctx", "input"),
        ("traceloop.entity.input", "input"),
        ("gen_ai.input.messages", "input"),
        ("gen_ai.prompt.0.content", "input"),
        ("llm.input_messages.0.message.content", "input"),
        ("langfuse.trace.output", "output"),
        ("langfuse.observation.output", "output"),
        ("ai.response.toolCalls", "output"),
        ("gcp.vertex.agent.tool_response", "output"),
        ("lk.response.text", "output"),
        ("mlflow.spanOutputs", "output"),
        ("gen_ai.output.messages", "output"),
        ("gen_ai.completion.0.content", "output"),
        ("llm.output_messages.0.message.content", "output"),
        ("gen_ai.request.model", "metadata"),
        ("http.response.body", "metadata"),
        ("message_bus.payload", "metadata"),
        ("custom.input_payload", "metadata"),
        ("ai.prompt.tools", "metadata"),
    ],
)
def test_media_field_for_attribute_matches_server_input_output_keys(
    attribute_key, expected_field
):
    assert (
        span_exporter_module._media_field_for_attribute(attribute_key) == expected_field
    )


def test_export_stage_media_skips_already_processed_provider_references(caplog):
    exporter = InMemorySpanExporter()
    media_manager, media_queue = _media_manager()
    image_base64 = base64.b64encode(b"image-bytes").decode("utf-8")
    transforming_exporter = span_exporter_module.LangfuseTransformingSpanExporter(
        exporter=exporter,
        media_manager=media_manager,
        mask_otel_spans=None,
    )
    span = SimpleNamespace(
        name="sdk-media-span",
        context=SimpleNamespace(trace_id=1, span_id=2),
    )
    provider_payloads = [
        [{"type": "base64", "media_type": "image/jpeg", "data": image_base64}],
        [{"type": "media", "mime_type": "image/png", "data": image_base64}],
        [{"inline_data": {"mime_type": "image/jpeg", "data": image_base64}}],
        [{"inlineData": {"mimeType": "image/png", "data": image_base64}}],
    ]

    for payload in provider_payloads:
        processed_payload = media_manager._find_and_process_media(
            data=payload,
            trace_id="trace-id",
            observation_id="observation-id",
            field="input",
        )
        serialized_payload = json.dumps(processed_payload, cls=EventSerializer)
        queue_size_before_export_processing = media_queue.qsize()

        caplog.clear()
        with caplog.at_level(logging.ERROR, logger="langfuse"):
            result = transforming_exporter._process_media_string(
                span=span,
                attribute_key="langfuse.observation.input",
                value=serialized_payload,
            )

        assert result == serialized_payload
        assert media_queue.qsize() == queue_size_before_export_processing
        assert not any(
            "Error parsing base64 data URI" in record.message
            for record in caplog.records
        )


def test_export_stage_media_replaces_invalid_media_attribute_with_failure_marker():
    exporter = InMemorySpanExporter()
    media_manager, media_queue = _media_manager()
    invalid_data_uri = "data:image/jpeg;base64,"

    provider = _tracer_provider(exporter=exporter, media_manager=media_manager)
    tracer = provider.get_tracer("openinference.instrumentation.openai")

    with tracer.start_as_current_span("third-party-invalid-media-span") as span:
        span.set_attribute("gen_ai.prompt", invalid_data_uri)

    provider.force_flush()

    exported_span = exporter.get_finished_spans()[0]

    assert exported_span.attributes["gen_ai.prompt"] == (
        "<Upload handling failed for LangfuseMedia of type None>"
    )
    assert media_queue.empty()


def test_mask_otel_spans_receives_whole_span_snapshot():
    exporter = InMemorySpanExporter()
    media_manager, _ = _media_manager()
    seen_params: list[MaskOtelSpansParams] = []

    def mask_otel_spans(*, params: MaskOtelSpansParams):
        seen_params.append(params)

        return None

    provider = _tracer_provider(
        exporter=exporter,
        media_manager=media_manager,
        mask_otel_spans=mask_otel_spans,
        resource_attributes={"deployment.environment.name": "ci"},
        should_export_span=lambda span: True,
    )
    tracer = provider.get_tracer("snapshot.scope", "1.2.3")

    with tracer.start_as_current_span("parent-span") as parent_span:
        parent_span.set_attribute("parent.attr", "visible")

        with tracer.start_as_current_span("child-span") as child_span:
            child_span.set_attribute("secret", "raw")

    provider.force_flush()

    params = seen_params[0]
    parent_identifier = _find_identifier_by_name(params, "parent-span")
    child_identifier = _find_identifier_by_name(params, "child-span")
    child_data = params.spans[child_identifier]

    assert len(params.spans) == 2
    assert child_data.trace_id == child_identifier.trace_id
    assert child_data.span_id == child_identifier.span_id
    assert child_data.parent_span_id == parent_identifier.span_id
    assert child_data.name == "child-span"
    assert child_data.instrumentation_scope_name == "snapshot.scope"
    assert child_data.instrumentation_scope_version == "1.2.3"
    assert child_data.attributes["secret"] == "raw"
    assert child_data.resource_attributes["service.name"] == "test"
    assert child_data.resource_attributes["deployment.environment.name"] == "ci"

    with pytest.raises(TypeError):
        params.spans[child_identifier] = child_data

    with pytest.raises(TypeError):
        child_data.attributes["secret"] = "changed"


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


def test_mask_otel_spans_none_result_leaves_batch_unchanged():
    exporter = InMemorySpanExporter()
    media_manager, _ = _media_manager()

    def mask_otel_spans(*, params: MaskOtelSpansParams):
        return None

    provider = _tracer_provider(
        exporter=exporter,
        media_manager=media_manager,
        mask_otel_spans=mask_otel_spans,
    )
    tracer = provider.get_tracer("openinference.instrumentation.openai")

    with tracer.start_as_current_span("third-party-span") as span:
        span.set_attribute("secret", "raw")

    provider.force_flush()

    exported_span = exporter.get_finished_spans()[0]

    assert exported_span.attributes["secret"] == "raw"


def test_mask_otel_spans_none_patch_leaves_span_unchanged():
    exporter = InMemorySpanExporter()
    media_manager, _ = _media_manager()

    def mask_otel_spans(*, params: MaskOtelSpansParams):
        identifier = next(iter(params.spans))

        return MaskOtelSpansResult(span_patches={identifier: None})

    provider = _tracer_provider(
        exporter=exporter,
        media_manager=media_manager,
        mask_otel_spans=mask_otel_spans,
    )
    tracer = provider.get_tracer("openinference.instrumentation.openai")

    with tracer.start_as_current_span("third-party-span") as span:
        span.set_attribute("secret", "raw")

    provider.force_flush()

    exported_span = exporter.get_finished_spans()[0]

    assert exported_span.attributes["secret"] == "raw"


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


def test_mask_otel_spans_invalid_span_patches_container_drops_batch():
    exporter = InMemorySpanExporter()
    media_manager, _ = _media_manager()
    invalid_result = MaskOtelSpansResult()
    object.__setattr__(invalid_result, "span_patches", [])

    def mask_otel_spans(*, params: MaskOtelSpansParams):
        return invalid_result

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


@pytest.mark.parametrize("invalid_field", ["set_attributes", "delete_attributes"])
def test_mask_otel_spans_invalid_patch_containers_drop_only_that_span(invalid_field):
    exporter = InMemorySpanExporter()
    media_manager, _ = _media_manager()

    def mask_otel_spans(*, params: MaskOtelSpansParams):
        target_identifier = _find_identifier_by_name(params, "drop-me")
        patch = OtelSpanPatch()

        if invalid_field == "set_attributes":
            object.__setattr__(patch, "set_attributes", ["secret"])
        else:
            object.__setattr__(patch, "delete_attributes", "secret")

        return MaskOtelSpansResult(span_patches={target_identifier: patch})

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


def test_mask_otel_spans_invalid_patch_keys_are_ignored():
    exporter = InMemorySpanExporter()
    media_manager, _ = _media_manager()

    def mask_otel_spans(*, params: MaskOtelSpansParams):
        identifier = next(iter(params.spans))

        return MaskOtelSpansResult(
            span_patches={
                identifier: OtelSpanPatch(
                    delete_attributes=[None, "secret"],
                    set_attributes={
                        None: "ignored",
                        "masked": "value",
                    },
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
        span.set_attribute("secret", "raw")
        span.set_attribute("kept", "value")

    provider.force_flush()

    exported_span = exporter.get_finished_spans()[0]

    assert "secret" not in exported_span.attributes
    assert None not in exported_span.attributes
    assert exported_span.attributes["kept"] == "value"
    assert exported_span.attributes["masked"] == "value"


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


def test_mask_otel_spans_runs_after_should_export_span_filter():
    exporter = InMemorySpanExporter()
    media_manager, media_queue = _media_manager()
    seen_params: list[MaskOtelSpansParams] = []
    image_base64 = base64.b64encode(b"image-bytes").decode("utf-8")

    def mask_otel_spans(*, params: MaskOtelSpansParams):
        seen_params.append(params)

        return MaskOtelSpansResult()

    provider = _tracer_provider(
        exporter=exporter,
        media_manager=media_manager,
        mask_otel_spans=mask_otel_spans,
        should_export_span=lambda span: span.name == "keep-me",
    )
    tracer = provider.get_tracer("openinference.instrumentation.openai")

    with tracer.start_as_current_span("drop-me") as span:
        span.set_attribute("gen_ai.prompt", f"data:image/jpeg;base64,{image_base64}")

    with tracer.start_as_current_span("keep-me") as span:
        span.set_attribute("gen_ai.request.model", "gpt-4o")

    provider.force_flush()

    exported_spans = exporter.get_finished_spans()

    assert len(seen_params) == 1
    assert [span_data.name for span_data in seen_params[0].spans.values()] == [
        "keep-me"
    ]
    assert [span.name for span in exported_spans] == ["keep-me"]
    assert media_queue.empty()


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
