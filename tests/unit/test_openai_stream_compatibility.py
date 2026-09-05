import httpx
import pytest
from packaging.version import Version

from langfuse.openai import openai as lf_openai

_OPENAI_VERSION = Version(lf_openai.__version__)
_STREAM_RESOURCES = [
    "chat",
    pytest.param(
        "responses",
        marks=pytest.mark.skipif(
            _OPENAI_VERSION < Version("1.66.0"),
            reason="Responses stream helpers require OpenAI 1.66 or newer",
        ),
    ),
]


def _stream_response(_request):
    return httpx.Response(
        200,
        content=b"data: [DONE]\n\n",
        headers={"content-type": "text/event-stream"},
    )


def _chat_completions(client):
    if _OPENAI_VERSION < Version("1.92.0"):
        return client.beta.chat.completions

    return client.chat.completions


def _stream_request(client, resource):
    if resource == "chat":
        return _chat_completions(client), {
            "messages": [{"role": "user", "content": "Hello"}]
        }

    return client.responses, {"input": "Hello"}


@pytest.mark.parametrize("resource", _STREAM_RESOURCES)
def test_stream_helper_compatibility(resource, langfuse_memory_client, get_span):
    client = lf_openai.OpenAI(
        api_key="test",
        http_client=httpx.Client(transport=httpx.MockTransport(_stream_response)),
    )
    endpoint, request = _stream_request(client, resource)
    name = f"unit-openai-{resource}-stream-compatibility"

    with endpoint.stream(
        name=name,
        metadata={"compatibility": lf_openai.__version__},
        model="sample-model",
        **request,
    ):
        pass

    client.close()
    langfuse_memory_client.flush()
    span = get_span(name)
    assert (
        span.attributes["langfuse.observation.metadata.compatibility"]
        == lf_openai.__version__
    )


@pytest.mark.parametrize("resource", _STREAM_RESOURCES)
@pytest.mark.asyncio
async def test_async_stream_helper_compatibility(
    resource, langfuse_memory_client, get_span
):
    client = lf_openai.AsyncOpenAI(
        api_key="test",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(_stream_response)),
    )
    endpoint, request = _stream_request(client, resource)
    name = f"unit-openai-async-{resource}-stream-compatibility"

    async with endpoint.stream(
        name=name,
        metadata={"compatibility": lf_openai.__version__},
        model="sample-model",
        **request,
    ):
        pass

    await client.close()
    langfuse_memory_client.flush()
    span = get_span(name)
    assert (
        span.attributes["langfuse.observation.metadata.compatibility"]
        == lf_openai.__version__
    )
