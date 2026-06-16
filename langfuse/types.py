"""Public API for all Langfuse types.

This module provides a centralized location for importing commonly used types
from the Langfuse SDK, making them easily accessible without requiring nested imports.

Example:
    ```python
    from langfuse.types import Evaluation, LocalExperimentItem, TaskFunction

    # Define your task function
    def my_task(*, item: LocalExperimentItem, **kwargs) -> str:
        return f"Processed: {item['input']}"

    # Define your evaluator
    def my_evaluator(*, output: str, **kwargs) -> Evaluation:
        return {"name": "length", "value": len(output)}
    ```
"""

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import (
    Any,
    Dict,
    Literal,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    TypedDict,
)

from opentelemetry.util.types import AttributeValue

try:
    from typing import NotRequired  # type: ignore
except ImportError:
    from typing_extensions import NotRequired


from langfuse.api import MediaContentType

SpanLevel = Literal["DEBUG", "DEFAULT", "WARNING", "ERROR"]

ScoreDataType = Literal["NUMERIC", "CATEGORICAL", "BOOLEAN", "TEXT", "CORRECTION"]

# Text scores are not supported for evals and experiments
ExperimentScoreType = Literal["NUMERIC", "CATEGORICAL", "BOOLEAN"]


class MaskFunction(Protocol):
    """A function that masks data.

    Keyword Args:
        data: The data to mask.

    Returns:
        The masked data that must be serializable to JSON.
    """

    def __call__(self, *, data: Any, **kwargs: Dict[str, Any]) -> Any: ...


@dataclass(frozen=True)
class OtelSpanIdentifier:
    """Stable key for one OpenTelemetry span in a masking batch.

    Use this object as the key when returning a patch for a span. It is a
    frozen, hashable dataclass, so the safest pattern is to reuse the exact
    identifier object from `MaskOtelSpansParams.spans` instead of rebuilding it.

    Attributes:
        trace_id: Lowercase 32-character hexadecimal OpenTelemetry trace ID.
        span_id: Lowercase 16-character hexadecimal OpenTelemetry span ID.
    """

    trace_id: str
    span_id: str


@dataclass(frozen=True)
class OtelSpanData:
    """Read-only OpenTelemetry span snapshot passed to `mask_otel_spans`.

    The snapshot contains the span data that Langfuse is about to export after
    the SDK has applied `should_export_span` filtering and export-stage media
    processing. The mappings are immutable views and mutating them is not
    supported; return an `OtelSpanPatch` to change exported attributes.

    `mask_otel_spans` can only change span attributes. It cannot change the
    span name, IDs, parent relationship, resource attributes, events, links, or
    instrumentation scope.

    Attributes:
        trace_id: Lowercase 32-character hexadecimal OpenTelemetry trace ID.
        span_id: Lowercase 16-character hexadecimal OpenTelemetry span ID.
        parent_span_id: Lowercase hexadecimal parent span ID, or `None` for a
            root span or when the parent is not available.
        name: OpenTelemetry span name.
        instrumentation_scope_name: Name of the instrumentation scope that
            emitted the span, for example `openai` or `langfuse`.
        instrumentation_scope_version: Version of the instrumentation scope, if
            the instrumentation library provided one.
        attributes: Read-only attributes that will be exported unless patched.
            Values use OpenTelemetry `AttributeValue` types: strings, booleans,
            numbers, or homogeneous sequences of those scalar values.
        resource_attributes: Read-only resource attributes from the span's
            OpenTelemetry resource. These are available for decisions only and
            cannot be patched through `mask_otel_spans`.
    """

    trace_id: str
    span_id: str
    parent_span_id: Optional[str]
    name: str
    instrumentation_scope_name: Optional[str]
    instrumentation_scope_version: Optional[str]
    attributes: Mapping[str, AttributeValue]
    resource_attributes: Mapping[str, AttributeValue]


@dataclass(frozen=True)
class MaskOtelSpansParams:
    """Input passed to an export-stage OpenTelemetry span masking function.

    A single call receives one OpenTelemetry export batch, not necessarily a
    complete trace, request, or Langfuse observation tree. Batch contents depend
    on OpenTelemetry span processor settings such as `flush_at`,
    `flush_interval`, explicit `flush()`, and shutdown.

    Example:
        ```python
        from typing import Optional

        from langfuse.types import (
            MaskOtelSpansParams,
            MaskOtelSpansResult,
            OtelSpanPatch,
        )

        def mask_otel_spans(
            *, params: MaskOtelSpansParams
        ) -> Optional[MaskOtelSpansResult]:
            patches = {}

            for identifier, span in params.spans.items():
                if "http.request.header.authorization" in span.attributes:
                    patches[identifier] = OtelSpanPatch(
                        delete_attributes=("http.request.header.authorization",),
                        set_attributes={"security.redacted": True},
                    )

            return MaskOtelSpansResult(span_patches=patches)
        ```

    Attributes:
        spans: Read-only mapping from stable span identifiers to span snapshots.
            Return patches using keys from this mapping.
    """

    spans: Mapping[OtelSpanIdentifier, OtelSpanData]


@dataclass(frozen=True)
class OtelSpanPatch:
    """Attribute changes to apply to one OpenTelemetry span before export.

    Patches are sparse: include only the attributes that should change. Langfuse
    deletes `delete_attributes` first and then applies `set_attributes`, so a key
    present in both fields is exported with the value from `set_attributes`.

    Attribute values must be valid OpenTelemetry attributes: strings, booleans,
    integers, floats, or homogeneous sequences of those scalar types. If one
    value is not valid for OpenTelemetry, Langfuse removes that attribute from
    the export rather than sending an invalid span.

    Example:
        ```python
        OtelSpanPatch(
            delete_attributes=("gen_ai.prompt.0.content",),
            set_attributes={
                "gen_ai.prompt.redacted": True,
                "app.masking.rule": "drop_prompt_text",
            },
        )
        ```

    Attributes:
        set_attributes: Attribute values to add or replace on the exported span.
        delete_attributes: Attribute keys to remove from the exported span.
    """

    set_attributes: Mapping[str, AttributeValue] = field(
        default_factory=lambda: MappingProxyType({})
    )
    delete_attributes: Sequence[str] = field(default_factory=tuple)


@dataclass(frozen=True)
class MaskOtelSpansResult:
    """Patches returned by a `mask_otel_spans` function.

    Omit spans that do not need changes. A mapping value of `None` also leaves
    that span unchanged. Returning an invalid patch to drop a span is not a
    supported API; use `should_export_span` when you need span-level export
    filtering.

    If `mask_otel_spans` raises or returns an object that is not a
    `MaskOtelSpansResult`, Langfuse drops the whole export batch. If one
    individual `OtelSpanPatch` is invalid, Langfuse drops only that span from
    the export batch.

    Attributes:
        span_patches: Mapping from identifiers in `MaskOtelSpansParams.spans` to
            sparse attribute patches.
    """

    span_patches: Mapping[OtelSpanIdentifier, Optional[OtelSpanPatch]] = field(
        default_factory=lambda: MappingProxyType({})
    )


class MaskOtelSpansFunction(Protocol):
    """Function protocol for export-stage OpenTelemetry span masking.

    `mask_otel_spans` runs after Langfuse decides which spans this client should
    export and after export-stage media handling has converted supported media
    payloads into Langfuse media references. It affects only the spans exported
    by this Langfuse client. If the same OpenTelemetry spans are sent to another
    exporter, that exporter receives its own unmodified copy.

    The function is synchronous. It usually runs on the OpenTelemetry batch span
    processor worker thread; during `flush()` and shutdown it may run on the
    caller thread. Keep it deterministic and fast, and avoid relying on request
    locals, the current active span, or async I/O.

    Return `None` to leave the whole batch unchanged, or return
    `MaskOtelSpansResult` with sparse patches for the spans that should change.

    Example:
        ```python
        from typing import Optional

        from langfuse import Langfuse
        from langfuse.types import (
            MaskOtelSpansParams,
            MaskOtelSpansResult,
            OtelSpanPatch,
        )

        def mask_otel_spans(
            *, params: MaskOtelSpansParams
        ) -> Optional[MaskOtelSpansResult]:
            patches = {}

            for identifier, span in params.spans.items():
                if span.instrumentation_scope_name == "openai":
                    patches[identifier] = OtelSpanPatch(
                        delete_attributes=(
                            "gen_ai.prompt.0.content",
                            "gen_ai.completion.0.content",
                        ),
                        set_attributes={"masking.applied": True},
                    )

            return MaskOtelSpansResult(span_patches=patches)

        langfuse = Langfuse(mask_otel_spans=mask_otel_spans)
        ```
    """

    def __call__(
        self, *, params: MaskOtelSpansParams
    ) -> Optional[MaskOtelSpansResult]: ...


class ParsedMediaReference(TypedDict):
    """A parsed media reference.

    Attributes:
        media_id: The media ID.
        source: The original source of the media, e.g. a file path, bytes, base64 data URI, etc.
        content_type: The content type of the media.
    """

    media_id: str
    source: str
    content_type: MediaContentType


class TraceContext(TypedDict):
    trace_id: str
    parent_span_id: NotRequired[str]


__all__ = [
    "SpanLevel",
    "ScoreDataType",
    "ExperimentScoreType",
    "MaskFunction",
    "MaskOtelSpansFunction",
    "MaskOtelSpansParams",
    "MaskOtelSpansResult",
    "OtelSpanData",
    "OtelSpanIdentifier",
    "OtelSpanPatch",
    "ParsedMediaReference",
    "TraceContext",
]
