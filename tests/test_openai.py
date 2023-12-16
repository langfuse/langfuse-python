import os
import pytest
from langfuse.client import Langfuse
from langfuse.openai import (
    _is_openai_v1,
    _is_streaming_response,
    openai,
    AsyncOpenAI,
    AzureOpenAI,
    AsyncAzureOpenAI,
)
from openai import APIConnectionError

from tests.utils import create_uuid, get_api


chat_func = (
    openai.chat.completions.create if _is_openai_v1() else openai.ChatCompletion.create
)
completion_func = (
    openai.completions.create if _is_openai_v1() else openai.Completion.create
)
expected_err = openai.APIError if _is_openai_v1() else openai.error.AuthenticationError
expected_err_msg = (
    "Connection error." if _is_openai_v1() else "You didn't provide an API key."
)


def test_openai_chat_completion():
    api = get_api()
    generation_name = create_uuid()
    completion = chat_func(
        name=generation_name,
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "1 + 1 = "}],
        temperature=0,
        metadata={"someKey": "someResponse"},
    )

    openai.flush_langfuse()

    generation = api.observations.get_many(name=generation_name, type="GENERATION")

    assert len(generation.data) != 0
    assert generation.data[0].name == generation_name
    assert generation.data[0].metadata == {"someKey": "someResponse"}
    assert len(completion.choices) != 0
    assert generation.data[0].input == [{"content": "1 + 1 = ", "role": "user"}]
    assert generation.data[0].type == "GENERATION"
    assert generation.data[0].model == "gpt-3.5-turbo-0613"
    assert generation.data[0].start_time is not None
    assert generation.data[0].end_time is not None
    assert generation.data[0].start_time < generation.data[0].end_time
    assert generation.data[0].model_parameters == {
        "temperature": 0,
        "top_p": 1,
        "frequency_penalty": 0,
        "maxTokens": "inf",
        "presence_penalty": 0,
    }
    assert generation.data[0].usage.input is not None
    assert generation.data[0].usage.output is not None
    assert generation.data[0].usage.total is not None
    assert "2" in generation.data[0].output


def test_openai_chat_completion_stream():
    api = get_api()
    generation_name = create_uuid()
    completion = chat_func(
        name=generation_name,
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "1 + 1 = "}],
        temperature=0,
        metadata={"someKey": "someResponse"},
        stream=True,
    )

    assert _is_streaming_response(completion)
    for i in completion:
        print(i)

    openai.flush_langfuse()

    generation = api.observations.get_many(name=generation_name, type="GENERATION")

    assert len(generation.data) != 0
    assert generation.data[0].name == generation_name
    assert generation.data[0].metadata == {"someKey": "someResponse"}

    assert generation.data[0].input == [{"content": "1 + 1 = ", "role": "user"}]
    assert generation.data[0].type == "GENERATION"
    assert generation.data[0].model == "gpt-3.5-turbo-0613"
    assert generation.data[0].start_time is not None
    assert generation.data[0].end_time is not None
    assert generation.data[0].start_time < generation.data[0].end_time
    assert generation.data[0].model_parameters == {
        "temperature": 0,
        "top_p": 1,
        "frequency_penalty": 0,
        "maxTokens": "inf",
        "presence_penalty": 0,
    }
    assert generation.data[0].usage.input is not None
    assert generation.data[0].usage.output is not None
    assert generation.data[0].usage.total is not None
    assert generation.data[0].output == "2"
    assert generation.data[0].completion_start_time is not None


def test_openai_chat_completion_stream_fail():
    api = get_api()
    generation_name = create_uuid()
    openai.api_key = ""

    with pytest.raises(expected_err, match=expected_err_msg):
        chat_func(
            name=generation_name,
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "1 + 1 = "}],
            temperature=0,
            metadata={"someKey": "someResponse"},
            stream=True,
        )

    openai.flush_langfuse()

    generation = api.observations.get_many(name=generation_name, type="GENERATION")

    assert len(generation.data) != 0
    assert generation.data[0].name == generation_name
    assert generation.data[0].metadata == {"someKey": "someResponse"}

    assert generation.data[0].input == [{"content": "1 + 1 = ", "role": "user"}]
    assert generation.data[0].type == "GENERATION"
    assert generation.data[0].model == "gpt-3.5-turbo"
    assert generation.data[0].start_time is not None
    assert generation.data[0].end_time is not None
    assert generation.data[0].start_time < generation.data[0].end_time
    assert generation.data[0].model_parameters == {
        "temperature": 0,
        "top_p": 1,
        "frequency_penalty": 0,
        "maxTokens": "inf",
        "presence_penalty": 0,
    }
    assert generation.data[0].usage.input is not None
    assert generation.data[0].usage.output is not None
    assert generation.data[0].usage.total is not None
    assert generation.data[0].level == "ERROR"
    assert expected_err_msg in generation.data[0].status_message
    assert generation.data[0].output is None

    openai.api_key = os.environ["OPENAI_API_KEY"]


def test_openai_chat_completion_with_trace():
    api = get_api()
    generation_name = create_uuid()
    trace_id = create_uuid()
    langfuse = Langfuse()

    langfuse.trace(id=trace_id)

    chat_func(
        name=generation_name,
        model="gpt-3.5-turbo",
        trace_id=trace_id,
        messages=[{"role": "user", "content": "1 + 1 = "}],
        temperature=0,
        metadata={"someKey": "someResponse"},
    )

    openai.flush_langfuse()

    generation = api.observations.get_many(name=generation_name, type="GENERATION")

    assert len(generation.data) != 0
    assert generation.data[0].name == generation_name
    assert generation.data[0].trace_id == trace_id


def test_openai_chat_completion_fail():
    api = get_api()
    generation_name = create_uuid()

    openai.api_key = ""

    with pytest.raises(expected_err, match=expected_err_msg):
        chat_func(
            name=generation_name,
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "1 + 1 = "}],
            temperature=0,
            metadata={"someKey": "someResponse"},
        )

    openai.flush_langfuse()

    generation = api.observations.get_many(name=generation_name, type="GENERATION")

    assert len(generation.data) != 0
    assert generation.data[0].name == generation_name
    assert generation.data[0].metadata == {"someKey": "someResponse"}
    assert generation.data[0].input == [{"content": "1 + 1 = ", "role": "user"}]
    assert generation.data[0].type == "GENERATION"
    assert generation.data[0].model == "gpt-3.5-turbo"
    assert generation.data[0].level == "ERROR"
    assert generation.data[0].start_time is not None
    assert generation.data[0].end_time is not None
    assert expected_err_msg in generation.data[0].status_message
    assert generation.data[0].start_time < generation.data[0].end_time
    assert generation.data[0].model_parameters == {
        "temperature": 0,
        "top_p": 1,
        "frequency_penalty": 0,
        "maxTokens": "inf",
        "presence_penalty": 0,
    }
    assert generation.data[0].output is None

    openai.api_key = os.environ["OPENAI_API_KEY"]


def test_openai_chat_completion_without_extra_param():
    completion = chat_func(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "1 + 1 = "}],
        temperature=0,
        metadata={"someKey": "someResponse"},
    )

    assert len(completion.choices) != 0


def test_openai_chat_completion_two_calls():
    api = get_api()
    generation_name = create_uuid()
    completion = chat_func(
        name=generation_name,
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "1 + 1 = "}],
        temperature=0,
        metadata={"someKey": "someResponse"},
    )

    generation_name_2 = create_uuid()

    completion_2 = chat_func(
        name=generation_name_2,
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "2 + 2 = "}],
        temperature=0,
        metadata={"someKey": "someResponse"},
    )

    openai.flush_langfuse()

    generation = api.observations.get_many(name=generation_name, type="GENERATION")

    assert len(generation.data) != 0
    assert generation.data[0].name == generation_name
    assert len(completion.choices) != 0

    assert generation.data[0].input == [{"content": "1 + 1 = ", "role": "user"}]

    generation_2 = api.observations.get_many(name=generation_name_2, type="GENERATION")

    assert len(generation_2.data) != 0
    assert generation_2.data[0].name == generation_name_2
    assert len(completion_2.choices) != 0

    assert generation_2.data[0].input == [{"content": "2 + 2 = ", "role": "user"}]


def test_openai_completion():
    api = get_api()
    generation_name = create_uuid()
    completion = completion_func(
        name=generation_name,
        model="gpt-3.5-turbo-instruct",
        prompt="1 + 1 = ",
        temperature=0,
        metadata={"someKey": "someResponse"},
    )

    openai.flush_langfuse()

    generation = api.observations.get_many(name=generation_name, type="GENERATION")

    assert len(generation.data) != 0
    assert generation.data[0].name == generation_name
    assert generation.data[0].metadata == {"someKey": "someResponse"}
    assert len(completion.choices) != 0
    assert completion.choices[0].text == generation.data[0].output
    assert generation.data[0].input == "1 + 1 = "
    assert generation.data[0].type == "GENERATION"
    assert generation.data[0].model == "gpt-3.5-turbo-instruct"
    assert generation.data[0].start_time is not None
    assert generation.data[0].end_time is not None
    assert generation.data[0].start_time < generation.data[0].end_time
    assert generation.data[0].model_parameters == {
        "temperature": 0,
        "top_p": 1,
        "frequency_penalty": 0,
        "maxTokens": "inf",
        "presence_penalty": 0,
    }
    assert generation.data[0].usage.input is not None
    assert generation.data[0].usage.output is not None
    assert generation.data[0].usage.total is not None
    assert generation.data[0].output == "2\n\n1 + 2 = 3\n\n2 + 3 = "


def test_openai_completion_stream():
    api = get_api()
    generation_name = create_uuid()
    completion = completion_func(
        name=generation_name,
        model="gpt-3.5-turbo-instruct",
        prompt="1 + 1 = ",
        temperature=0,
        metadata={"someKey": "someResponse"},
        stream=True,
    )

    assert _is_streaming_response(completion)
    for i in completion:
        print(i)

    openai.flush_langfuse()

    generation = api.observations.get_many(name=generation_name, type="GENERATION")

    assert len(generation.data) != 0
    assert generation.data[0].name == generation_name
    assert generation.data[0].metadata == {"someKey": "someResponse"}

    assert generation.data[0].input == "1 + 1 = "
    assert generation.data[0].type == "GENERATION"
    assert generation.data[0].model == "gpt-3.5-turbo-instruct"
    assert generation.data[0].start_time is not None
    assert generation.data[0].end_time is not None
    assert generation.data[0].start_time < generation.data[0].end_time
    assert generation.data[0].model_parameters == {
        "temperature": 0,
        "top_p": 1,
        "frequency_penalty": 0,
        "maxTokens": "inf",
        "presence_penalty": 0,
    }
    assert generation.data[0].usage.input is not None
    assert generation.data[0].usage.output is not None
    assert generation.data[0].usage.total is not None
    assert generation.data[0].output == "2\n\n1 + 2 = 3\n\n2 + 3 = "
    assert generation.data[0].completion_start_time is not None


def test_openai_completion_fail():
    api = get_api()
    generation_name = create_uuid()

    openai.api_key = ""

    with pytest.raises(expected_err, match=expected_err_msg):
        completion_func(
            name=generation_name,
            model="gpt-3.5-turbo-instruct",
            prompt="1 + 1 = ",
            temperature=0,
            metadata={"someKey": "someResponse"},
        )

    openai.flush_langfuse()

    generation = api.observations.get_many(name=generation_name, type="GENERATION")

    assert len(generation.data) != 0
    assert generation.data[0].name == generation_name
    assert generation.data[0].metadata == {"someKey": "someResponse"}
    assert generation.data[0].input == "1 + 1 = "
    assert generation.data[0].type == "GENERATION"
    assert generation.data[0].model == "gpt-3.5-turbo-instruct"
    assert generation.data[0].level == "ERROR"
    assert generation.data[0].start_time is not None
    assert generation.data[0].end_time is not None
    assert expected_err_msg in generation.data[0].status_message
    assert generation.data[0].start_time < generation.data[0].end_time
    assert generation.data[0].model_parameters == {
        "temperature": 0,
        "top_p": 1,
        "frequency_penalty": 0,
        "maxTokens": "inf",
        "presence_penalty": 0,
    }
    assert generation.data[0].output is None

    openai.api_key = os.environ["OPENAI_API_KEY"]


def test_openai_completion_stream_fail():
    api = get_api()
    generation_name = create_uuid()
    openai.api_key = ""

    with pytest.raises(expected_err, match=expected_err_msg):
        completion_func(
            name=generation_name,
            model="gpt-3.5-turbo",
            prompt="1 + 1 = ",
            temperature=0,
            metadata={"someKey": "someResponse"},
            stream=True,
        )

    openai.flush_langfuse()

    generation = api.observations.get_many(name=generation_name, type="GENERATION")

    assert len(generation.data) != 0
    assert generation.data[0].name == generation_name
    assert generation.data[0].metadata == {"someKey": "someResponse"}

    assert generation.data[0].input == "1 + 1 = "
    assert generation.data[0].type == "GENERATION"
    assert generation.data[0].model == "gpt-3.5-turbo"
    assert generation.data[0].start_time is not None
    assert generation.data[0].end_time is not None
    assert generation.data[0].start_time < generation.data[0].end_time
    assert generation.data[0].model_parameters == {
        "temperature": 0,
        "top_p": 1,
        "frequency_penalty": 0,
        "maxTokens": "inf",
        "presence_penalty": 0,
    }
    assert generation.data[0].usage.input is not None
    assert generation.data[0].usage.output is not None
    assert generation.data[0].usage.total is not None
    assert generation.data[0].level == "ERROR"
    assert expected_err_msg in generation.data[0].status_message
    assert generation.data[0].output is None

    openai.api_key = os.environ["OPENAI_API_KEY"]


def test_fails_wrong_name():
    with pytest.raises(TypeError, match="name must be a string"):
        completion_func(
            name={"key": "generation_name"},
            model="gpt-3.5-turbo-instruct",
            prompt="1 + 1 = ",
            temperature=0,
        )


def test_fails_wrong_metadata():
    with pytest.raises(TypeError, match="metadata must be a dictionary"):
        completion_func(
            metadata="metadata",
            model="gpt-3.5-turbo-instruct",
            prompt="1 + 1 = ",
            temperature=0,
        )


def test_fails_wrong_trace_id():
    with pytest.raises(TypeError, match="trace_id must be a string"):
        completion_func(
            trace_id={"trace_id": "metadata"},
            model="gpt-3.5-turbo-instruct",
            prompt="1 + 1 = ",
            temperature=0,
        )


@pytest.mark.asyncio
async def test_async_chat():
    api = get_api()
    client = AsyncOpenAI()
    generation_name = create_uuid()

    completion = await client.chat.completions.create(
        messages=[{"role": "user", "content": "1 + 1 = "}],
        model="gpt-3.5-turbo",
        name=generation_name,
    )

    openai.flush_langfuse()
    print(completion)

    generation = api.observations.get_many(name=generation_name, type="GENERATION")

    assert len(generation.data) != 0
    assert generation.data[0].name == generation_name
    assert len(completion.choices) != 0

    assert generation.data[0].input == [{"content": "1 + 1 = ", "role": "user"}]
    assert generation.data[0].type == "GENERATION"
    assert generation.data[0].model == "gpt-3.5-turbo-0613"
    assert generation.data[0].start_time is not None
    assert generation.data[0].end_time is not None
    assert generation.data[0].start_time < generation.data[0].end_time
    assert generation.data[0].model_parameters == {
        "temperature": 1,
        "top_p": 1,
        "frequency_penalty": 0,
        "maxTokens": "inf",
        "presence_penalty": 0,
    }
    assert generation.data[0].usage.input is not None
    assert generation.data[0].usage.output is not None
    assert generation.data[0].usage.total is not None
    assert "2" in generation.data[0].output


@pytest.mark.asyncio
async def test_async_chat_stream():
    api = get_api()
    client = AsyncOpenAI()

    generation_name = create_uuid()

    completion = await client.chat.completions.create(
        messages=[{"role": "user", "content": "1 + 1 = "}],
        model="gpt-3.5-turbo",
        name=generation_name,
        stream=True,
    )

    async for c in completion:
        print(c)

    openai.flush_langfuse()
    print(completion)

    generation = api.observations.get_many(name=generation_name, type="GENERATION")

    assert len(generation.data) != 0
    assert generation.data[0].name == generation_name
    assert generation.data[0].input == [{"content": "1 + 1 = ", "role": "user"}]
    assert generation.data[0].type == "GENERATION"
    assert generation.data[0].model == "gpt-3.5-turbo-0613"
    assert generation.data[0].start_time is not None
    assert generation.data[0].end_time is not None
    assert generation.data[0].start_time < generation.data[0].end_time
    assert generation.data[0].model_parameters == {
        "temperature": 1,
        "top_p": 1,
        "frequency_penalty": 0,
        "maxTokens": "inf",
        "presence_penalty": 0,
    }
    assert generation.data[0].usage.input is not None
    assert generation.data[0].usage.output is not None
    assert generation.data[0].usage.total is not None
    assert "2" in generation.data[0].output


def test_openai_function_call():
    from typing import List

    from pydantic import BaseModel

    api = get_api()
    generation_name = create_uuid()

    class StepByStepAIResponse(BaseModel):
        title: str
        steps: List[str]

    import json

    response = openai.chat.completions.create(
        name=generation_name,
        model="gpt-3.5-turbo-0613",
        messages=[{"role": "user", "content": "Explain how to assemble a PC"}],
        functions=[
            {
                "name": "get_answer_for_user_query",
                "description": "Get user answer in series of steps",
                "parameters": StepByStepAIResponse.schema(),
            }
        ],
        function_call={"name": "get_answer_for_user_query"},
    )

    output = json.loads(response.choices[0].message.function_call.arguments)

    openai.flush_langfuse()

    generation = api.observations.get_many(name=generation_name, type="GENERATION")

    assert len(generation.data) != 0
    assert generation.data[0].name == generation_name
    assert generation.data[0].output is not None
    assert "function_call" in generation.data[0].output

    assert output["title"] is not None


def test_azure():
    api = get_api()
    generation_name = create_uuid()
    azure = AzureOpenAI(
        api_key="missing",
        api_version="2020-07-01-preview",
        base_url="https://api.labs.azure.com",
    )

    with pytest.raises(APIConnectionError):
        azure.chat.completions.create(
            name=generation_name,
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "1 + 1 = "}],
            temperature=0,
            metadata={"someKey": "someResponse"},
        )

    openai.flush_langfuse()

    generation = api.observations.get_many(name=generation_name, type="GENERATION")

    assert len(generation.data) != 0
    assert generation.data[0].name == generation_name
    assert generation.data[0].metadata == {"someKey": "someResponse"}
    assert generation.data[0].input == [{"content": "1 + 1 = ", "role": "user"}]
    assert generation.data[0].type == "GENERATION"
    assert generation.data[0].model == "gpt-3.5-turbo"
    assert generation.data[0].start_time is not None
    assert generation.data[0].end_time is not None
    assert generation.data[0].start_time < generation.data[0].end_time
    assert generation.data[0].model_parameters == {
        "temperature": 0,
        "top_p": 1,
        "frequency_penalty": 0,
        "maxTokens": "inf",
        "presence_penalty": 0,
    }
    assert generation.data[0].usage.input is not None
    assert generation.data[0].usage.output is not None
    assert generation.data[0].usage.total is not None
    assert generation.data[0].level == "ERROR"


@pytest.mark.asyncio
async def test_async_azure():
    api = get_api()
    generation_name = create_uuid()
    azure = AsyncAzureOpenAI(
        api_key="missing",
        api_version="2020-07-01-preview",
        base_url="https://api.labs.azure.com",
    )

    with pytest.raises(APIConnectionError):
        await azure.chat.completions.create(
            name=generation_name,
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "1 + 1 = "}],
            temperature=0,
            metadata={"someKey": "someResponse"},
        )

    openai.flush_langfuse()

    generation = api.observations.get_many(name=generation_name, type="GENERATION")

    assert len(generation.data) != 0
    assert generation.data[0].name == generation_name
    assert generation.data[0].metadata == {"someKey": "someResponse"}
    assert generation.data[0].input == [{"content": "1 + 1 = ", "role": "user"}]
    assert generation.data[0].type == "GENERATION"
    assert generation.data[0].model == "gpt-3.5-turbo"
    assert generation.data[0].start_time is not None
    assert generation.data[0].end_time is not None
    assert generation.data[0].start_time < generation.data[0].end_time
    assert generation.data[0].model_parameters == {
        "temperature": 0,
        "top_p": 1,
        "frequency_penalty": 0,
        "maxTokens": "inf",
        "presence_penalty": 0,
    }
    assert generation.data[0].usage.input is not None
    assert generation.data[0].usage.output is not None
    assert generation.data[0].usage.total is not None
    assert generation.data[0].level == "ERROR"
