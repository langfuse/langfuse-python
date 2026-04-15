from types import SimpleNamespace
from unittest.mock import patch

import pytest

from langfuse._client.attributes import LangfuseOtelSpanAttributes
from langfuse.openai import openai as lf_openai


class DummySyncResponse:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class DummyAsyncResponse:
    def __init__(self) -> None:
        self.closed = False

    async def aclose(self) -> None:
        self.closed = True


class DummyOpenAIStream(lf_openai.Stream):
    def __init__(self, items, response) -> None:
        self.response = response
        self._iterator = iter(items)


class DummyOpenAIAsyncStream(lf_openai.AsyncStream):
    def __init__(self, items, response) -> None:
        self.response = response
        self._iterator = self._stream(items)

    async def _stream(self, items):
        for item in items:
            yield item


def _make_chat_stream_chunks():
    usage = SimpleNamespace(prompt_tokens=3, completion_tokens=1, total_tokens=4)

    return [
        SimpleNamespace(
            model="gpt-4o-mini",
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(
                        role="assistant",
                        content="2",
                        function_call=None,
                        tool_calls=None,
                    ),
                    finish_reason=None,
                )
            ],
            usage=None,
        ),
        SimpleNamespace(
            model="gpt-4o-mini",
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(
                        role=None,
                        content=None,
                        function_call=None,
                        tool_calls=None,
                    ),
                    finish_reason="stop",
                )
            ],
            usage=usage,
        ),
    ]


def test_chat_completion_exports_generation_span(
    langfuse_memory_client, get_span, json_attr
):
    openai_client = lf_openai.OpenAI(api_key="test")
    response = SimpleNamespace(
        model="gpt-4o-mini",
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    role="assistant",
                    content="2",
                    function_call=None,
                    tool_calls=None,
                    audio=None,
                )
            )
        ],
        usage=SimpleNamespace(prompt_tokens=3, completion_tokens=1, total_tokens=4),
    )

    with patch.object(openai_client.chat.completions, "_post", return_value=response):
        result = openai_client.chat.completions.create(
            name="unit-openai-chat",
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "1 + 1 = ?"}],
            temperature=0,
            metadata={"suite": "unit"},
        )

    assert result is response

    langfuse_memory_client.flush()
    span = get_span("unit-openai-chat")

    assert span.attributes[LangfuseOtelSpanAttributes.OBSERVATION_TYPE] == "generation"
    assert (
        span.attributes[LangfuseOtelSpanAttributes.OBSERVATION_MODEL] == "gpt-4o-mini"
    )
    assert span.attributes["langfuse.observation.metadata.suite"] == "unit"
    assert json_attr(span, LangfuseOtelSpanAttributes.OBSERVATION_INPUT) == [
        {"role": "user", "content": "1 + 1 = ?"}
    ]
    assert json_attr(span, LangfuseOtelSpanAttributes.OBSERVATION_OUTPUT) == {
        "role": "assistant",
        "content": "2",
    }
    assert json_attr(span, LangfuseOtelSpanAttributes.OBSERVATION_MODEL_PARAMETERS) == {
        "temperature": 0,
        "max_tokens": "Infinity",
        "top_p": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0,
    }
    assert json_attr(span, LangfuseOtelSpanAttributes.OBSERVATION_USAGE_DETAILS) == {
        "prompt_tokens": 3,
        "completion_tokens": 1,
        "total_tokens": 4,
    }


def test_streaming_chat_completion_exports_ttft(
    langfuse_memory_client, get_span, json_attr
):
    openai_client = lf_openai.OpenAI(api_key="test")
    usage = SimpleNamespace(prompt_tokens=3, completion_tokens=1, total_tokens=4)

    def fake_stream():
        yield SimpleNamespace(
            model="gpt-4o-mini",
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(
                        role="assistant",
                        content="2",
                        function_call=None,
                        tool_calls=None,
                    ),
                    finish_reason=None,
                )
            ],
            usage=None,
        )
        yield SimpleNamespace(
            model="gpt-4o-mini",
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(
                        role=None,
                        content=None,
                        function_call=None,
                        tool_calls=None,
                    ),
                    finish_reason="stop",
                )
            ],
            usage=usage,
        )

    with patch.object(
        openai_client.chat.completions, "_post", return_value=fake_stream()
    ):
        stream = openai_client.chat.completions.create(
            name="unit-openai-stream",
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "1 + 1 = ?"}],
            temperature=0,
            stream=True,
        )
        chunks = list(stream)

    assert len(chunks) == 2

    langfuse_memory_client.flush()
    span = get_span("unit-openai-stream")

    assert span.attributes[LangfuseOtelSpanAttributes.OBSERVATION_OUTPUT] == "2"
    assert (
        span.attributes[LangfuseOtelSpanAttributes.OBSERVATION_COMPLETION_START_TIME]
        is not None
    )
    assert span.attributes["langfuse.observation.metadata.finish_reason"] == "stop"
    assert json_attr(span, LangfuseOtelSpanAttributes.OBSERVATION_USAGE_DETAILS) == {
        "prompt_tokens": 3,
        "completion_tokens": 1,
        "total_tokens": 4,
    }


def test_chat_completion_error_marks_generation_error(langfuse_memory_client, get_span):
    openai_client = lf_openai.OpenAI(api_key="test")

    with patch.object(
        openai_client.chat.completions,
        "_post",
        side_effect=RuntimeError("boom"),
    ):
        with pytest.raises(RuntimeError, match="boom"):
            openai_client.chat.completions.create(
                name="unit-openai-error",
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "explode"}],
                temperature=0,
            )

    langfuse_memory_client.flush()
    span = get_span("unit-openai-error")

    assert span.attributes[LangfuseOtelSpanAttributes.OBSERVATION_LEVEL] == "ERROR"
    assert (
        "boom" in span.attributes[LangfuseOtelSpanAttributes.OBSERVATION_STATUS_MESSAGE]
    )
    assert LangfuseOtelSpanAttributes.OBSERVATION_OUTPUT not in span.attributes


def test_openai_stream_preserves_original_stream_contract(
    langfuse_memory_client, get_span, json_attr
):
    openai_client = lf_openai.OpenAI(api_key="test")
    raw_response = DummySyncResponse()
    raw_stream = DummyOpenAIStream(_make_chat_stream_chunks(), raw_response)

    with patch.object(openai_client.chat.completions, "_post", return_value=raw_stream):
        stream = openai_client.chat.completions.create(
            name="unit-openai-native-stream",
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "1 + 1 = ?"}],
            temperature=0,
            stream=True,
        )

    assert stream is raw_stream
    assert isinstance(stream, lf_openai.Stream)
    assert stream.response is raw_response

    chunks = list(stream)
    stream.close()

    assert len(chunks) == 2
    assert raw_response.closed is True

    langfuse_memory_client.flush()
    span = get_span("unit-openai-native-stream")

    assert span.attributes[LangfuseOtelSpanAttributes.OBSERVATION_OUTPUT] == "2"
    assert (
        span.attributes[LangfuseOtelSpanAttributes.OBSERVATION_COMPLETION_START_TIME]
        is not None
    )
    assert span.attributes["langfuse.observation.metadata.finish_reason"] == "stop"
    assert json_attr(span, LangfuseOtelSpanAttributes.OBSERVATION_USAGE_DETAILS) == {
        "prompt_tokens": 3,
        "completion_tokens": 1,
        "total_tokens": 4,
    }


@pytest.mark.asyncio
async def test_async_chat_completion_exports_generation_span(
    langfuse_memory_client, get_span, json_attr
):
    openai_client = lf_openai.AsyncOpenAI(api_key="test")
    response = SimpleNamespace(
        model="gpt-4o-mini",
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    role="assistant",
                    content="async result",
                    function_call=None,
                    tool_calls=None,
                    audio=None,
                )
            )
        ],
        usage=SimpleNamespace(prompt_tokens=5, completion_tokens=2, total_tokens=7),
    )

    with patch.object(openai_client.chat.completions, "_post", return_value=response):
        result = await openai_client.chat.completions.create(
            name="unit-openai-async",
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "hello"}],
            temperature=0,
        )

    assert result is response

    langfuse_memory_client.flush()
    span = get_span("unit-openai-async")

    assert json_attr(span, LangfuseOtelSpanAttributes.OBSERVATION_OUTPUT) == {
        "role": "assistant",
        "content": "async result",
    }
    assert json_attr(span, LangfuseOtelSpanAttributes.OBSERVATION_USAGE_DETAILS) == {
        "prompt_tokens": 5,
        "completion_tokens": 2,
        "total_tokens": 7,
    }


@pytest.mark.asyncio
async def test_openai_async_stream_preserves_original_stream_contract(
    langfuse_memory_client, get_span, json_attr
):
    openai_client = lf_openai.AsyncOpenAI(api_key="test")
    raw_response = DummyAsyncResponse()
    raw_stream = DummyOpenAIAsyncStream(_make_chat_stream_chunks(), raw_response)

    with patch.object(openai_client.chat.completions, "_post", return_value=raw_stream):
        stream = await openai_client.chat.completions.create(
            name="unit-openai-native-async-stream",
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "1 + 1 = ?"}],
            temperature=0,
            stream=True,
        )

    assert stream is raw_stream
    assert isinstance(stream, lf_openai.AsyncStream)
    assert stream.response is raw_response
    assert hasattr(stream, "aclose")

    chunks = []
    async for chunk in stream:
        chunks.append(chunk)

    await stream.aclose()

    assert len(chunks) == 2
    assert raw_response.closed is True

    langfuse_memory_client.flush()
    span = get_span("unit-openai-native-async-stream")

    assert span.attributes[LangfuseOtelSpanAttributes.OBSERVATION_OUTPUT] == "2"
    assert (
        span.attributes[LangfuseOtelSpanAttributes.OBSERVATION_COMPLETION_START_TIME]
        is not None
    )
    assert span.attributes["langfuse.observation.metadata.finish_reason"] == "stop"
    assert json_attr(span, LangfuseOtelSpanAttributes.OBSERVATION_USAGE_DETAILS) == {
        "prompt_tokens": 3,
        "completion_tokens": 1,
        "total_tokens": 4,
    }


@pytest.mark.asyncio
async def test_openai_async_stream_supports_anext(
    langfuse_memory_client, get_span, json_attr
):
    openai_client = lf_openai.AsyncOpenAI(api_key="test")
    raw_stream = DummyOpenAIAsyncStream(
        _make_chat_stream_chunks(), DummyAsyncResponse()
    )

    with patch.object(openai_client.chat.completions, "_post", return_value=raw_stream):
        stream = await openai_client.chat.completions.create(
            name="unit-openai-native-async-anext",
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "1 + 1 = ?"}],
            temperature=0,
            stream=True,
        )

    first = await stream.__anext__()
    second = await stream.__anext__()

    assert first.choices[0].delta.content == "2"
    assert second.choices[0].finish_reason == "stop"

    with pytest.raises(StopAsyncIteration):
        await stream.__anext__()

    langfuse_memory_client.flush()
    span = get_span("unit-openai-native-async-anext")

    assert span.attributes[LangfuseOtelSpanAttributes.OBSERVATION_OUTPUT] == "2"
    assert (
        span.attributes[LangfuseOtelSpanAttributes.OBSERVATION_COMPLETION_START_TIME]
        is not None
    )
    assert span.attributes["langfuse.observation.metadata.finish_reason"] == "stop"
    assert json_attr(span, LangfuseOtelSpanAttributes.OBSERVATION_USAGE_DETAILS) == {
        "prompt_tokens": 3,
        "completion_tokens": 1,
        "total_tokens": 4,
    }


def test_embedding_exports_dimensions_and_count(
    langfuse_memory_client, get_span, json_attr
):
    openai_client = lf_openai.OpenAI(api_key="test")
    response = SimpleNamespace(
        model="text-embedding-3-small",
        data=[SimpleNamespace(embedding=[0.1, 0.2, 0.3])],
        usage=SimpleNamespace(prompt_tokens=2, total_tokens=2),
    )

    with patch.object(openai_client.embeddings, "_post", return_value=response):
        result = openai_client.embeddings.create(
            name="unit-openai-embedding",
            model="text-embedding-3-small",
            input="hello world",
        )

    assert result is response

    langfuse_memory_client.flush()
    span = get_span("unit-openai-embedding")

    assert span.attributes[LangfuseOtelSpanAttributes.OBSERVATION_TYPE] == "embedding"
    assert json_attr(span, LangfuseOtelSpanAttributes.OBSERVATION_OUTPUT) == {
        "dimensions": 3,
        "count": 1,
    }
    assert json_attr(span, LangfuseOtelSpanAttributes.OBSERVATION_USAGE_DETAILS) == {
        "input": 2
    }
