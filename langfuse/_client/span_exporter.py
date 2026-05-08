import json
from collections.abc import Mapping as MappingCollection
from collections.abc import Sequence as SequenceCollection
from types import MappingProxyType
from typing import Any, Dict, Optional, Sequence, cast

from opentelemetry.attributes import BoundedAttributes
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
from opentelemetry.trace import format_span_id, format_trace_id
from opentelemetry.util.types import AttributeValue

from langfuse._client.attributes import LangfuseOtelSpanAttributes
from langfuse._task_manager.media_manager import MediaManager
from langfuse._utils.serializer import EventSerializer
from langfuse.logger import langfuse_logger
from langfuse.media import LangfuseMedia
from langfuse.types import (
    MaskOtelSpansFunction,
    MaskOtelSpansParams,
    MaskOtelSpansResult,
    OtelSpanData,
    OtelSpanIdentifier,
    OtelSpanPatch,
)

_INPUT_MEDIA_ATTRIBUTE_KEYS = frozenset(
    {
        LangfuseOtelSpanAttributes.TRACE_INPUT,
        LangfuseOtelSpanAttributes.OBSERVATION_INPUT,
        "ai.prompt.messages",
        "ai.prompt",
        "ai.toolCall.args",
        "gcp.vertex.agent.llm_request",
        "gcp.vertex.agent.tool_call_args",
        "prompt",
        "lk.input_text",
        "lk.user_transcript",
        "lk.chat_ctx",
        "lk.user_input",
        "mlflow.spanInputs",
        "traceloop.entity.input",
        "input.value",
        "pydantic_ai.all_messages",
        "gen_ai.system_instructions",
        "input",
        "gen_ai.input.messages",
        "gen_ai.tool.call.arguments",
        "genkit:input",
        "tool_arguments",
    }
)

_OUTPUT_MEDIA_ATTRIBUTE_KEYS = frozenset(
    {
        LangfuseOtelSpanAttributes.TRACE_OUTPUT,
        LangfuseOtelSpanAttributes.OBSERVATION_OUTPUT,
        "ai.response.text",
        "ai.result.text",
        "ai.toolCall.result",
        "ai.response.object",
        "ai.result.object",
        "ai.response.toolCalls",
        "ai.result.toolCalls",
        "gcp.vertex.agent.llm_response",
        "gcp.vertex.agent.tool_response",
        "all_messages_events",
        "lk.function_tool.output",
        "lk.response.text",
        "mlflow.spanOutputs",
        "traceloop.entity.output",
        "output.value",
        "final_result",
        "output",
        "gen_ai.output.messages",
        "gen_ai.tool.call.result",
        "genkit:output",
        "tool_response",
    }
)

_INPUT_MEDIA_ATTRIBUTE_PREFIXES = (
    "gen_ai.prompt",
    "llm.input_messages",
)

_OUTPUT_MEDIA_ATTRIBUTE_PREFIXES = (
    "gen_ai.completion",
    "llm.output_messages",
)


class LangfuseTransformingSpanExporter(SpanExporter):
    """Apply Langfuse export-stage transformations before delegating export."""

    def __init__(
        self,
        *,
        exporter: SpanExporter,
        media_manager: Optional[MediaManager],
        mask_otel_spans: Optional[MaskOtelSpansFunction],
    ) -> None:
        self._exporter = exporter
        self._media_manager = media_manager
        self._mask_otel_spans = mask_otel_spans

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        span_attributes = [
            (
                span,
                self._process_media_attributes(
                    span=span,
                    attributes=dict(span.attributes or {}),
                ),
            )
            for span in spans
        ]

        if self._mask_otel_spans is not None:
            masked_span_attributes = self._apply_mask_otel_spans(
                span_attributes=span_attributes
            )

            if masked_span_attributes is None:
                return SpanExportResult.SUCCESS

            span_attributes = masked_span_attributes

        transformed_spans = [
            self._clone_span(span=span, attributes=attributes)
            for span, attributes in span_attributes
        ]

        if not transformed_spans:
            return SpanExportResult.SUCCESS

        return self._exporter.export(transformed_spans)

    def shutdown(self) -> None:
        self._exporter.shutdown()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return self._exporter.force_flush(timeout_millis=timeout_millis)

    def _process_media_attributes(
        self, *, span: ReadableSpan, attributes: Dict[str, AttributeValue]
    ) -> Dict[str, AttributeValue]:
        if self._media_manager is None:
            return attributes

        processed_attributes: Dict[str, AttributeValue] = {}

        for key, value in attributes.items():
            try:
                processed_attributes[key] = self._process_media_attribute_value(
                    span=span,
                    attribute_key=key,
                    value=value,
                )
            except Exception as error:
                langfuse_logger.warning(
                    "Media processing error: Failed to process span attribute before export. "
                    f"Leaving attribute unchanged. span_name='{span.name}' "
                    f"trace_id='{_get_trace_id(span)}' span_id='{_get_span_id(span)}' "
                    f"attribute_key='{key}' error='{error}'"
                )
                processed_attributes[key] = value

        return processed_attributes

    def _process_media_attribute_value(
        self,
        *,
        span: ReadableSpan,
        attribute_key: str,
        value: AttributeValue,
    ) -> AttributeValue:
        if isinstance(value, str):
            return self._process_media_string(
                span=span,
                attribute_key=attribute_key,
                value=value,
            )

        if _is_attribute_sequence(value):
            sequence_value = cast(Sequence[Any], value)

            return cast(
                AttributeValue,
                tuple(
                    self._process_media_string(
                        span=span,
                        attribute_key=attribute_key,
                        value=item,
                    )
                    if isinstance(item, str)
                    else item
                    for item in sequence_value
                ),
            )

        return value

    def _process_media_string(
        self,
        *,
        span: ReadableSpan,
        attribute_key: str,
        value: str,
    ) -> str:
        media_manager = cast(MediaManager, self._media_manager)
        trace_id = _get_trace_id(span)
        observation_id = _get_span_id(span)
        field = _media_field_for_attribute(attribute_key)

        if _is_base64_data_uri(value):
            processed_direct_value = media_manager._find_and_process_media(
                data=value,
                trace_id=trace_id,
                observation_id=observation_id,
                field=field,
            )

            direct_reference = _media_reference_string(processed_direct_value)

            if direct_reference is not None:
                return direct_reference

            if processed_direct_value != value:
                return _serialize_media_value(processed_direct_value, fallback=value)

        stripped_value = value.lstrip()

        if not stripped_value.startswith(("{", "[")):
            return value

        if not _may_contain_serialized_media(value):
            return value

        try:
            parsed_value = json.loads(value)
        except Exception:
            return value

        if not isinstance(parsed_value, (dict, list)):
            return value

        processed_json_value = media_manager._find_and_process_media(
            data=parsed_value,
            trace_id=trace_id,
            observation_id=observation_id,
            field=field,
        )

        if processed_json_value == parsed_value:
            return value

        return _serialize_media_value(processed_json_value, fallback=value)

    def _apply_mask_otel_spans(
        self,
        *,
        span_attributes: Sequence[tuple[ReadableSpan, Dict[str, AttributeValue]]],
    ) -> Optional[list[tuple[ReadableSpan, Dict[str, AttributeValue]]]]:
        mask_otel_spans = cast(MaskOtelSpansFunction, self._mask_otel_spans)
        span_data_by_identifier: Dict[OtelSpanIdentifier, OtelSpanData] = {}

        for span, attributes in span_attributes:
            identifier = _create_otel_span_identifier(span)

            if identifier in span_data_by_identifier:
                langfuse_logger.error(
                    "Masking error: mask_otel_spans received duplicate span identifiers. "
                    "Dropping export batch. "
                    f"trace_id='{identifier.trace_id}' span_id='{identifier.span_id}'"
                )
                return None

            span_data_by_identifier[identifier] = _create_otel_span_data(
                span=span, attributes=attributes, identifier=identifier
            )

        try:
            result: Any = mask_otel_spans(
                params=MaskOtelSpansParams(
                    spans=MappingProxyType(dict(span_data_by_identifier))
                )
            )
        except Exception as error:
            langfuse_logger.error(
                "Masking error: mask_otel_spans raised an exception. "
                f"Dropping export batch. span_count={len(span_attributes)} "
                f"error='{error}'"
            )
            return None

        if result is None:
            return list(span_attributes)

        if not isinstance(result, MaskOtelSpansResult):
            langfuse_logger.error(
                "Masking error: mask_otel_spans returned an invalid result. "
                f"Dropping export batch. result_type='{type(result).__name__}'"
            )
            return None

        span_patches: Any = result.span_patches

        if not isinstance(span_patches, MappingCollection):
            langfuse_logger.error(
                "Masking error: mask_otel_spans returned invalid span_patches. "
                f"Dropping export batch. "
                f"span_patches_type='{type(span_patches).__name__}'"
            )
            return None

        span_identifiers = set(span_data_by_identifier)

        for identifier in span_patches:
            if identifier not in span_identifiers:
                langfuse_logger.error(
                    "Masking error: mask_otel_spans returned a patch for an unknown "
                    "span identifier. Dropping export batch. "
                    f"identifier_type='{type(identifier).__name__}'"
                )
                return None

        masked_span_attributes: list[
            tuple[ReadableSpan, Dict[str, AttributeValue]]
        ] = []

        for span, attributes in span_attributes:
            identifier = _create_otel_span_identifier(span)
            patch = span_patches.get(identifier)

            if patch is None:
                masked_span_attributes.append((span, attributes))
                continue

            patched_attributes = self._apply_otel_span_patch(
                span=span,
                attributes=attributes,
                patch=patch,
            )

            if patched_attributes is not None:
                masked_span_attributes.append((span, patched_attributes))

        return masked_span_attributes

    def _apply_otel_span_patch(
        self,
        *,
        span: ReadableSpan,
        attributes: Dict[str, AttributeValue],
        patch: Any,
    ) -> Optional[Dict[str, AttributeValue]]:
        if not isinstance(patch, OtelSpanPatch):
            langfuse_logger.error(
                "Masking error: mask_otel_spans returned an invalid span patch. "
                "Dropping span. "
                f"span_name='{span.name}' trace_id='{_get_trace_id(span)}' "
                f"span_id='{_get_span_id(span)}' patch_type='{type(patch).__name__}'"
            )
            return None

        set_attributes: Any = patch.set_attributes
        delete_attributes: Any = patch.delete_attributes

        if not isinstance(set_attributes, MappingCollection):
            langfuse_logger.error(
                "Masking error: mask_otel_spans returned invalid set_attributes. "
                "Dropping span. "
                f"span_name='{span.name}' trace_id='{_get_trace_id(span)}' "
                f"span_id='{_get_span_id(span)}' "
                f"set_attributes_type='{type(set_attributes).__name__}'"
            )
            return None

        if isinstance(delete_attributes, (str, bytes)) or not isinstance(
            delete_attributes, SequenceCollection
        ):
            langfuse_logger.error(
                "Masking error: mask_otel_spans returned invalid delete_attributes. "
                "Dropping span. "
                f"span_name='{span.name}' trace_id='{_get_trace_id(span)}' "
                f"span_id='{_get_span_id(span)}' "
                f"delete_attributes_type='{type(delete_attributes).__name__}'"
            )
            return None

        masked_attributes = dict(attributes)

        for key in delete_attributes:
            if not _is_valid_attribute_key(key):
                langfuse_logger.warning(
                    "Masking error: mask_otel_spans requested deletion with an invalid attribute key. "
                    f"Ignoring delete entry. span_name='{span.name}' "
                    f"trace_id='{_get_trace_id(span)}' span_id='{_get_span_id(span)}' "
                    f"delete_key_type='{type(key).__name__}'"
                )
                continue

            masked_attributes.pop(key, None)

        for key, value in set_attributes.items():
            if not _is_valid_attribute_key(key):
                langfuse_logger.warning(
                    "Masking error: mask_otel_spans returned an invalid set_attributes key. "
                    f"Ignoring set entry. span_name='{span.name}' "
                    f"trace_id='{_get_trace_id(span)}' span_id='{_get_span_id(span)}' "
                    f"attribute_key_type='{type(key).__name__}'"
                )
                continue

            cleaned_attribute = _clean_attribute_value(key=key, value=value)

            if cleaned_attribute is None:
                masked_attributes.pop(key, None)
                langfuse_logger.warning(
                    "Masking error: mask_otel_spans returned an invalid attribute value. "
                    f"Deleting attribute from export. span_name='{span.name}' "
                    f"trace_id='{_get_trace_id(span)}' span_id='{_get_span_id(span)}' "
                    f"attribute_key='{key}' value_type='{type(value).__name__}'"
                )
                continue

            masked_attributes[key] = cleaned_attribute

        return masked_attributes

    @staticmethod
    def _clone_span(
        *, span: ReadableSpan, attributes: Dict[str, AttributeValue]
    ) -> ReadableSpan:
        return ReadableSpan(
            name=span.name,
            context=span.context,
            parent=span.parent,
            resource=span.resource,
            attributes=attributes,
            events=span.events,
            links=span.links,
            kind=span.kind,
            status=span.status,
            start_time=span.start_time,
            end_time=span.end_time,
            instrumentation_scope=span.instrumentation_scope,
        )


def _create_otel_span_identifier(span: ReadableSpan) -> OtelSpanIdentifier:
    return OtelSpanIdentifier(
        trace_id=_get_trace_id(span),
        span_id=_get_span_id(span),
    )


def _create_otel_span_data(
    *,
    span: ReadableSpan,
    attributes: Dict[str, AttributeValue],
    identifier: OtelSpanIdentifier,
) -> OtelSpanData:
    instrumentation_scope = span.instrumentation_scope

    return OtelSpanData(
        trace_id=identifier.trace_id,
        span_id=identifier.span_id,
        parent_span_id=_get_parent_span_id(span),
        name=span.name,
        instrumentation_scope_name=instrumentation_scope.name
        if instrumentation_scope
        else None,
        instrumentation_scope_version=instrumentation_scope.version
        if instrumentation_scope
        else None,
        attributes=MappingProxyType(dict(attributes)),
        resource_attributes=MappingProxyType(dict(span.resource.attributes or {})),
    )


def _clean_attribute_value(*, key: str, value: Any) -> Optional[AttributeValue]:
    cleaned_attributes = BoundedAttributes(maxlen=1, immutable=False)

    try:
        cleaned_attributes[key] = value
    except Exception:
        return None

    if key not in cleaned_attributes:
        return None

    return cast(AttributeValue, cleaned_attributes[key])


def _is_valid_attribute_key(key: Any) -> bool:
    return isinstance(key, str) and bool(key)


def _is_attribute_sequence(value: AttributeValue) -> bool:
    return isinstance(value, SequenceCollection) and not isinstance(value, (str, bytes))


def _is_base64_data_uri(value: str) -> bool:
    if not value.startswith("data:") or "," not in value:
        return False

    header, _ = value.split(",", 1)

    return header.endswith(";base64")


def _may_contain_serialized_media(value: str) -> bool:
    if "data:" in value or "inline_data" in value or "inlineData" in value:
        return True

    has_media_type_key = (
        '"media_type"' in value or '"mime_type"' in value or '"mimeType"' in value
    )

    return has_media_type_key and '"data"' in value


def _media_reference_string(value: Any) -> Optional[str]:
    if not isinstance(value, LangfuseMedia):
        return None

    return value._reference_string


def _serialize_media_value(value: Any, *, fallback: str) -> str:
    if isinstance(value, str):
        return value

    if isinstance(value, LangfuseMedia):
        return (
            _media_reference_string(value)
            or f"<Upload handling failed for LangfuseMedia of type {value._content_type}>"
        )

    try:
        return json.dumps(value, cls=EventSerializer)
    except Exception:
        return fallback


def _media_field_for_attribute(
    attribute_key: str,
) -> str:
    if attribute_key in _INPUT_MEDIA_ATTRIBUTE_KEYS or attribute_key.startswith(
        _INPUT_MEDIA_ATTRIBUTE_PREFIXES
    ):
        return "input"

    if attribute_key in _OUTPUT_MEDIA_ATTRIBUTE_KEYS or attribute_key.startswith(
        _OUTPUT_MEDIA_ATTRIBUTE_PREFIXES
    ):
        return "output"

    return "metadata"


def _get_trace_id(span: ReadableSpan) -> str:
    if span.context is None:
        return ""

    return format_trace_id(span.context.trace_id)


def _get_span_id(span: ReadableSpan) -> str:
    if span.context is None:
        return ""

    return format_span_id(span.context.span_id)


def _get_parent_span_id(span: ReadableSpan) -> Optional[str]:
    if span.parent is None:
        return None

    return format_span_id(span.parent.span_id)
