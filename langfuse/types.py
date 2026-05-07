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

ScoreDataType = Literal["NUMERIC", "CATEGORICAL", "BOOLEAN", "TEXT"]

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
    """Stable identifier for an OpenTelemetry span in a masking batch."""

    trace_id: str
    span_id: str


@dataclass(frozen=True)
class OtelSpanData:
    """Read-only OpenTelemetry span snapshot passed to a masking function."""

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
    """Read-only batch passed to an export-stage OpenTelemetry span mask function."""

    spans: Mapping[OtelSpanIdentifier, OtelSpanData]


@dataclass(frozen=True)
class OtelSpanPatch:
    """Attribute mutations for one OpenTelemetry span before export."""

    set_attributes: Mapping[str, AttributeValue] = field(
        default_factory=lambda: MappingProxyType({})
    )
    delete_attributes: Sequence[str] = field(default_factory=tuple)


@dataclass(frozen=True)
class MaskOtelSpansResult:
    """Sparse attribute patches returned by an OpenTelemetry span mask function."""

    span_patches: Mapping[OtelSpanIdentifier, Optional[OtelSpanPatch]] = field(
        default_factory=lambda: MappingProxyType({})
    )


class MaskOtelSpansFunction(Protocol):
    """A synchronous function that masks OpenTelemetry span attributes before export."""

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
