from contextlib import nullcontext
from typing import Any, AsyncGenerator, Generator, Iterable

import pytest

from langfuse import observe
from langfuse._client.attributes import LangfuseOtelSpanAttributes


@pytest.mark.asyncio
@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.parametrize("close_early", [False, True])
@pytest.mark.parametrize("fails", [False, True])
async def test_generator_transform_settles_span_once(
    langfuse_memory_client: Any,
    memory_exporter: Any,
    is_async: bool,
    close_early: bool,
    fails: bool,
) -> None:
    should_fail = fails
    calls = 0
    failure = ValueError("cannot format streamed output")

    def transform(chunks: Iterable[str]) -> str:
        nonlocal calls
        calls += 1
        if should_fail:
            raise failure
        return "".join(chunks)

    @observe(name="transformed-stream", transform_to_string=transform)
    def sync_stream() -> Generator[str, None, None]:
        yield "first"
        yield "second"

    @observe(name="transformed-stream", transform_to_string=transform)
    async def async_stream() -> AsyncGenerator[str, None]:
        yield "first"
        yield "second"

    stream: Any = async_stream() if is_async else sync_stream()
    try:
        assert (await anext(stream) if is_async else next(stream)) == "first"
        with (
            pytest.raises(ValueError, match="cannot format streamed output")
            if fails
            else nullcontext()
        ) as raised:
            if close_early:
                if is_async:
                    await stream.aclose()
                else:
                    stream.close()
            elif is_async:
                assert [chunk async for chunk in stream] == ["second"]
            else:
                assert list(stream) == ["second"]
        if fails:
            assert raised is not None and raised.value is failure

        langfuse_memory_client.flush()
        spans = [
            span
            for span in memory_exporter.get_finished_spans()
            if span.name == "transformed-stream"
        ]
        assert len(spans) == 1
        attributes = spans[0].attributes
        if fails:
            assert attributes[LangfuseOtelSpanAttributes.OBSERVATION_LEVEL] == "ERROR"
            assert attributes[
                LangfuseOtelSpanAttributes.OBSERVATION_STATUS_MESSAGE
            ] == str(failure)
        else:
            assert attributes[LangfuseOtelSpanAttributes.OBSERVATION_OUTPUT] == (
                "first" if close_early else "firstsecond"
            )
        # A second close must not run a failed formatter or end/export again.
        if is_async:
            await stream.aclose()
        else:
            stream.close()
        langfuse_memory_client.flush()
        assert calls == 1
        assert len(memory_exporter.get_finished_spans()) == 1
    finally:
        # Keep the unchanged baseline from retrying the failing formatter at GC.
        should_fail = False
        if is_async:
            await stream.aclose()
        else:
            stream.close()
