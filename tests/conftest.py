import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import pytest
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

from langfuse._client.client import Langfuse
from langfuse._client.resource_manager import LangfuseResourceManager

CORE_E2E_FILENAMES = {
    "test_core_sdk.py",
    "test_decorators.py",
    "test_media.py",
}

SERIAL_E2E_NODEIDS = {
    "tests/e2e/test_core_sdk.py::test_create_trace",
    "tests/e2e/test_core_sdk.py::test_create_boolean_score",
    "tests/e2e/test_core_sdk.py::test_create_categorical_score",
    "tests/e2e/test_core_sdk.py::test_create_score_with_custom_timestamp",
    "tests/e2e/test_decorators.py::test_return_dict_for_output",
    "tests/e2e/test_decorators.py::test_media",
    "tests/e2e/test_decorators.py::test_merge_metadata_and_tags",
    "tests/e2e/test_experiments.py::test_boolean_score_types",
    "tests/e2e/test_media.py::test_replace_media_reference_string_in_object",
}


class InMemorySpanExporter(SpanExporter):
    """Simple in-memory exporter to collect spans for deterministic tests."""

    def __init__(self) -> None:
        self._finished_spans: list[ReadableSpan] = []
        self._stopped = False

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        if self._stopped:
            return SpanExportResult.FAILURE

        self._finished_spans.extend(spans)
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        self._stopped = True

    def get_finished_spans(self) -> list[ReadableSpan]:
        return list(self._finished_spans)

    def clear(self) -> None:
        self._finished_spans.clear()


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        file_path = Path(str(item.fspath))
        test_group = file_path.parent.name

        if test_group == "unit":
            item.add_marker(pytest.mark.unit)
            continue

        if test_group == "e2e":
            item.add_marker(pytest.mark.e2e)
            # Keep the data shard as the default so new tests under tests/e2e
            # are picked up automatically unless we explicitly promote them.
            if file_path.name in CORE_E2E_FILENAMES:
                item.add_marker(pytest.mark.e2e_core)
            else:
                item.add_marker(pytest.mark.e2e_data)
            if item.nodeid in SERIAL_E2E_NODEIDS:
                item.add_marker(pytest.mark.serial_e2e)
            continue

        if test_group == "live_provider":
            item.add_marker(pytest.mark.e2e)
            item.add_marker(pytest.mark.live_provider)


@pytest.fixture(autouse=True)
def reset_langfuse_state() -> Iterable[None]:
    LangfuseResourceManager.reset()
    yield
    LangfuseResourceManager.reset()


@pytest.fixture
def memory_exporter() -> Iterable[InMemorySpanExporter]:
    exporter = InMemorySpanExporter()
    yield exporter
    exporter.shutdown()


@pytest.fixture
def langfuse_memory_client(
    monkeypatch: pytest.MonkeyPatch, memory_exporter: InMemorySpanExporter
) -> Iterable[Langfuse]:
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "test-public-key")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "test-secret-key")
    monkeypatch.setenv("LANGFUSE_BASE_URL", "http://test-host")

    tracer_provider = TracerProvider(resource=Resource.create({"service.name": "test"}))

    def mock_init(self: Any, **kwargs: Any) -> None:
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        from langfuse._client.span_filter import is_default_export_span

        self.public_key = kwargs.get("public_key", "test-public-key")
        blocked_scopes = kwargs.get("blocked_instrumentation_scopes")
        self.blocked_instrumentation_scopes = (
            blocked_scopes if blocked_scopes is not None else []
        )
        self._should_export_span = (
            kwargs.get("should_export_span") or is_default_export_span
        )
        BatchSpanProcessor.__init__(
            self,
            span_exporter=memory_exporter,
            max_export_batch_size=512,
            schedule_delay_millis=1,
        )

    monkeypatch.setattr(
        "langfuse._client.span_processor.LangfuseSpanProcessor.__init__",
        mock_init,
    )

    client = Langfuse(
        public_key="test-public-key",
        secret_key="test-secret-key",
        base_url="http://test-host",
        tracing_enabled=True,
        tracer_provider=tracer_provider,
    )

    yield client
    client.flush()


@pytest.fixture
def get_span(memory_exporter: InMemorySpanExporter):
    def _get_span(name: str) -> ReadableSpan:
        for span in memory_exporter.get_finished_spans():
            if span.name == name:
                return span

        raise AssertionError(
            f"Span {name!r} not found in {[span.name for span in memory_exporter.get_finished_spans()]}"
        )

    return _get_span


@pytest.fixture
def find_spans(memory_exporter: InMemorySpanExporter):
    def _find_spans(name: str) -> list[ReadableSpan]:
        return [
            span for span in memory_exporter.get_finished_spans() if span.name == name
        ]

    return _find_spans


@pytest.fixture
def json_attr():
    def _json_attr(span: ReadableSpan, attribute: str) -> Any:
        value = span.attributes[attribute]
        if not isinstance(value, str):
            return value

        return json.loads(value)

    return _json_attr
