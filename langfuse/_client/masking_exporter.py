"""Masking span exporter for batch masking prior to OTLP export.

When batch_mask is configured, this exporter wraps the real OTLPSpanExporter.
It collects maskable span attributes (input/output/metadata), calls the
batch mask function, then wraps each span with MaskedAttributeSpanWrapper
so the OTLP exporter serializes masked data.

When use_async_masking is True, batch masking and export run in a background
thread so the main thread is not blocked.
"""

import threading
from queue import Empty, Full, Queue
from typing import Any, Dict, List, Optional, Sequence, Tuple

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

from langfuse._client.attributes import LangfuseOtelSpanAttributes
from langfuse.logger import langfuse_logger
from langfuse.types import BatchMaskFunction

_SHUTDOWN_SENTINEL: Any = object()
_FLUSH_SENTINEL: Any = object()


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


def _export_batch_sync(
    span_exporter: SpanExporter,
    batch_mask: BatchMaskFunction,
    mask_batch_size: Optional[int],
    spans: Sequence[ReadableSpan],
) -> SpanExportResult:
    """Run batch mask and export on the given spans; used by sync and worker."""
    if not spans:
        return span_exporter.export(spans)

    items, backref = _collect_maskable_items(spans)
    if not items:
        return span_exporter.export(spans)

    masked_triples = _apply_batch_mask(
        batch_mask, items, backref, mask_batch_size
    )
    masked_per_span = _build_masked_attributes_per_span(len(spans), masked_triples)

    wrapped = [
        MaskedAttributeSpanWrapper(span, masked_per_span[i])
        for i, span in enumerate(spans)
    ]
    return span_exporter.export(wrapped)


class MaskingSpanExporter(SpanExporter):
    """SpanExporter that runs batch masking before delegating to the real exporter."""

    _QUEUE_MAX_SIZE = 1000
    _QUEUE_PUT_TIMEOUT_SEC = 5.0

    def __init__(
        self,
        *,
        span_exporter: SpanExporter,
        batch_mask: BatchMaskFunction,
        mask_batch_size: Optional[int] = None,
        use_async_masking: bool = False,
    ) -> None:
        self._span_exporter = span_exporter
        self._batch_mask = batch_mask
        self._mask_batch_size = mask_batch_size
        self._use_async_masking = use_async_masking
        self._queue: Optional[Queue] = None
        self._worker: Optional[threading.Thread] = None
        self._flush_event: Optional[threading.Event] = None
        self._closed = False

        if use_async_masking:
            self._queue = Queue(maxsize=self._QUEUE_MAX_SIZE)
            self._flush_event = threading.Event()
            self._worker = threading.Thread(
                target=self._worker_loop,
                name="langfuse-masking-exporter",
                daemon=True,
            )
            self._worker.start()

    def _worker_loop(self) -> None:
        if self._queue is None or self._flush_event is None:
            return
        while True:
            try:
                item = self._queue.get(timeout=0.5)
            except Empty:
                continue
            if item is _SHUTDOWN_SENTINEL:
                break
            if item is _FLUSH_SENTINEL:
                while True:
                    try:
                        extra = self._queue.get_nowait()
                    except Empty:
                        break
                    if extra is _SHUTDOWN_SENTINEL:
                        self._queue.put(extra)
                        self._flush_event.set()
                        break
                    if extra is not _FLUSH_SENTINEL:
                        try:
                            _export_batch_sync(
                                self._span_exporter,
                                self._batch_mask,
                                self._mask_batch_size,
                                extra,
                            )
                        except Exception as e:
                            langfuse_logger.error(
                                "Async batch masking/export failed: %s",
                                e,
                            )
                self._flush_event.set()
                continue
            try:
                _export_batch_sync(
                    self._span_exporter,
                    self._batch_mask,
                    self._mask_batch_size,
                    item,
                )
            except Exception as e:
                langfuse_logger.error(
                    "Async batch masking/export failed: %s",
                    e,
                )
        self._flush_event.set()

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        if not spans:
            return self._span_exporter.export(spans)

        if not self._use_async_masking:
            return _export_batch_sync(
                self._span_exporter,
                self._batch_mask,
                self._mask_batch_size,
                spans,
            )

        if self._queue is None or self._closed:
            return SpanExportResult.FAILURE

        try:
            self._queue.put(list(spans), timeout=self._QUEUE_PUT_TIMEOUT_SEC)
        except Full:
            langfuse_logger.error(
                "Masking exporter queue full; dropping batch of %s spans",
                len(spans),
            )
            return SpanExportResult.FAILURE
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        self._closed = True
        if self._worker is not None and self._worker.is_alive() and self._queue is not None:
            self._queue.put(_SHUTDOWN_SENTINEL)
            self._worker.join(timeout=10.0)
            if self._worker.is_alive():
                langfuse_logger.warning(
                    "Masking exporter worker did not stop within timeout"
                )
        if hasattr(self._span_exporter, "shutdown"):
            self._span_exporter.shutdown()

    def force_flush(self, timeout_millis: Optional[int] = None) -> bool:
        if not self._use_async_masking:
            if hasattr(self._span_exporter, "force_flush"):
                return self._span_exporter.force_flush(timeout_millis)
            return True

        if self._queue is None or self._flush_event is None or self._closed:
            return True

        self._flush_event.clear()
        try:
            self._queue.put(_FLUSH_SENTINEL, timeout=self._QUEUE_PUT_TIMEOUT_SEC)
        except Full:
            return False
        timeout_sec = (timeout_millis / 1000.0) if timeout_millis is not None else 10.0
        if not self._flush_event.wait(timeout=timeout_sec):
            return False
        if hasattr(self._span_exporter, "force_flush"):
            return self._span_exporter.force_flush(timeout_millis)
        return True
