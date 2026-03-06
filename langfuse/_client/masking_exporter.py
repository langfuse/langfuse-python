"""Masking span exporter for batch masking prior to OTLP export.

When batch_mask is configured, this exporter wraps the real OTLPSpanExporter.
It collects maskable span attributes (input/output/metadata), calls the
batch mask function, then wraps each span with MaskedAttributeSpanWrapper
so the OTLP exporter serializes masked data.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

from langfuse._client.attributes import LangfuseOtelSpanAttributes
from langfuse.logger import langfuse_logger
from langfuse.types import BatchMaskFunction


class MaskedAttributeSpanWrapper:
    """Wraps a ReadableSpan and overlays masked attribute values.

    When using batch masking, the exporter wraps each ReadableSpan in this
    class so that attribute reads (e.g. by the OTLP exporter) see masked
    input/output/metadata instead of the original values.

    Delegates all span properties and methods to the underlying span except
    for attributes: both .attributes and ._attributes return a merge of the
    original span attributes and the provided masked_attributes (masked
    values take precedence).
    """

    def __init__(self, span: ReadableSpan, masked_attributes: Dict[str, Any]) -> None:
        self._span = span
        self._masked_attributes = masked_attributes
        base = dict(span.attributes) if span.attributes else {}
        self._merged_attributes = {**base, **masked_attributes}

    @property
    def attributes(self) -> Dict[str, Any]:
        """Return span attributes with masked values overlaid."""
        return self._merged_attributes

    @property
    def _attributes(self) -> Dict[str, Any]:
        """Return span attributes with masked values overlaid (private API)."""
        return self._merged_attributes

    def __getattr__(self, name: str) -> Any:
        """Delegate all other attribute access to the underlying span."""
        return getattr(self._span, name)

# Attribute keys that may contain PII and are passed to batch_mask.
_MASKABLE_ATTR_KEYS = (
    LangfuseOtelSpanAttributes.TRACE_INPUT,
    LangfuseOtelSpanAttributes.TRACE_OUTPUT,
    LangfuseOtelSpanAttributes.OBSERVATION_INPUT,
    LangfuseOtelSpanAttributes.OBSERVATION_OUTPUT,
    LangfuseOtelSpanAttributes.OBSERVATION_METADATA,
)


def _collect_maskable_items(
    spans: Sequence[ReadableSpan],
) -> Tuple[List[Any], List[Tuple[int, str]]]:
    """Collect maskable attribute values from spans in deterministic order.

    Returns:
        (items, backref): items is the list to pass to batch_mask;
        backref[i] is (span_index, attr_key) for items[i].
    """
    items: List[Any] = []
    backref: List[Tuple[int, str]] = []
    for span_idx, span in enumerate(spans):
        attrs = span.attributes or {}
        for key in _MASKABLE_ATTR_KEYS:
            if key in attrs:
                items.append(attrs[key])
                backref.append((span_idx, key))
    return items, backref


def _apply_batch_mask(
    batch_mask: BatchMaskFunction,
    items: List[Any],
    backref: List[Tuple[int, str]],
    mask_batch_size: Optional[int],
) -> List[Tuple[int, str, Any]]:
    """Call batch_mask (in chunks if mask_batch_size set), return (span_idx, key, masked_value)."""
    if not items:
        return []
    if mask_batch_size is None or mask_batch_size <= 0:
        try:
            masked = batch_mask(items)
        except Exception as e:
            langfuse_logger.error(
                "Batch masking failed; exporting with fallback masking. Error: %s", e
            )
            masked = ["<masking failed>"] * len(items)
        if len(masked) != len(items):
            langfuse_logger.error(
                "Batch mask returned length %s, expected %s; using fallback.",
                len(masked),
                len(items),
            )
            masked = ["<masking failed>"] * len(items)
    else:
        masked: List[Any] = []
        for i in range(0, len(items), mask_batch_size):
            chunk = items[i : i + mask_batch_size]
            chunk_backref = backref[i : i + mask_batch_size]
            try:
                chunk_masked = batch_mask(chunk)
            except Exception as e:
                langfuse_logger.error(
                    "Batch masking chunk failed; using fallback. Error: %s", e
                )
                chunk_masked = ["<masking failed>"] * len(chunk)
            if len(chunk_masked) != len(chunk):
                langfuse_logger.error(
                    "Batch mask returned length %s, expected %s; using fallback.",
                    len(chunk_masked),
                    len(chunk),
                )
                chunk_masked = ["<masking failed>"] * len(chunk)
            masked.extend(chunk_masked)
    return [(backref[j][0], backref[j][1], masked[j]) for j in range(len(masked))]


def _build_masked_attributes_per_span(
    num_spans: int,
    masked_triples: List[Tuple[int, str, Any]],
) -> List[Dict[str, Any]]:
    """Build a masked_attributes dict for each span."""
    result: List[Dict[str, Any]] = [{} for _ in range(num_spans)]
    for span_idx, key, value in masked_triples:
        result[span_idx][key] = value
    return result


class MaskingSpanExporter(SpanExporter):
    """SpanExporter that runs batch masking before delegating to the real exporter."""

    def __init__(
        self,
        *,
        span_exporter: SpanExporter,
        batch_mask: BatchMaskFunction,
        mask_batch_size: Optional[int] = None,
    ) -> None:
        self._span_exporter = span_exporter
        self._batch_mask = batch_mask
        self._mask_batch_size = mask_batch_size

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        if not spans:
            return self._span_exporter.export(spans)

        items, backref = _collect_maskable_items(spans)
        if not items:
            return self._span_exporter.export(spans)

        masked_triples = _apply_batch_mask(
            self._batch_mask, items, backref, self._mask_batch_size
        )
        masked_per_span = _build_masked_attributes_per_span(len(spans), masked_triples)

        wrapped = [
            MaskedAttributeSpanWrapper(span, masked_per_span[i])
            for i, span in enumerate(spans)
        ]
        return self._span_exporter.export(wrapped)

    def shutdown(self) -> None:
        if hasattr(self._span_exporter, "shutdown"):
            self._span_exporter.shutdown()

    def force_flush(self, timeout_millis: Optional[int] = None) -> bool:
        if hasattr(self._span_exporter, "force_flush"):
            return self._span_exporter.force_flush(timeout_millis)
        return True
