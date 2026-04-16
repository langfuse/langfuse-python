import sys

import pytest

from langfuse import observe
from langfuse._client.attributes import LangfuseOtelSpanAttributes


def _finished_spans_by_name(memory_exporter, name: str):
    return [span for span in memory_exporter.get_finished_spans() if span.name == name]


def test_sync_generator_preserves_context_without_output_capture(
    langfuse_memory_client, memory_exporter
):
    @observe(name="child_step")
    def child_step(index: int) -> str:
        return f"item_{index}"

    @observe(name="root", capture_output=False)
    def root():
        def body():
            for index in range(2):
                yield child_step(index)

        return body()

    generator = root()

    assert memory_exporter.get_finished_spans() == []

    assert list(generator) == ["item_0", "item_1"]

    langfuse_memory_client.flush()

    root_span = _finished_spans_by_name(memory_exporter, "root")[0]
    child_spans = _finished_spans_by_name(memory_exporter, "child_step")

    assert len(child_spans) == 2
    assert all(child.parent is not None for child in child_spans)
    assert all(
        child.parent.span_id == root_span.context.span_id for child in child_spans
    )
    assert all(
        child.context.trace_id == root_span.context.trace_id for child in child_spans
    )
    assert LangfuseOtelSpanAttributes.OBSERVATION_OUTPUT not in root_span.attributes


@pytest.mark.asyncio
@pytest.mark.skipif(sys.version_info < (3, 11), reason="requires python3.11 or higher")
async def test_streaming_response_preserves_context_without_output_capture(
    langfuse_memory_client, memory_exporter
):
    class StreamingResponse:
        def __init__(self, body_iterator):
            self.body_iterator = body_iterator

    @observe(name="stream_step")
    async def stream_step(index: int) -> str:
        return f"chunk_{index}"

    async def body():
        for index in range(2):
            yield await stream_step(index)

    @observe(name="endpoint", capture_output=False)
    async def endpoint():
        return StreamingResponse(body())

    response = await endpoint()

    assert memory_exporter.get_finished_spans() == []

    assert [item async for item in response.body_iterator] == ["chunk_0", "chunk_1"]

    langfuse_memory_client.flush()

    endpoint_span = _finished_spans_by_name(memory_exporter, "endpoint")[0]
    step_spans = _finished_spans_by_name(memory_exporter, "stream_step")

    assert len(step_spans) == 2
    assert all(step.parent is not None for step in step_spans)
    assert all(
        step.parent.span_id == endpoint_span.context.span_id for step in step_spans
    )
    assert all(
        step.context.trace_id == endpoint_span.context.trace_id for step in step_spans
    )
    assert LangfuseOtelSpanAttributes.OBSERVATION_OUTPUT not in endpoint_span.attributes
