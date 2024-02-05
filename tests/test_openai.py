import os

import pytest
import time
from openai import APIConnectionError

from langfuse.client import Langfuse
from langfuse.openai import (
    AsyncAzureOpenAI,
    AsyncOpenAI,
    AzureOpenAI,
    _is_openai_v1,
    _is_streaming_response,
    filter_image_data,
    openai,
)
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


# def _wait_on_run(run, thread, openai):
#     while run.status == "queued" or run.status == "in_progress":
#         run = client.beta.threads.runs.retrieve(
#             thread_id=thread.id,
#             run_id=run.id,
#         )
#         time.sleep(0.5)
#     return run


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
        "max_tokens": "inf",
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
        "max_tokens": "inf",
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
        "max_tokens": "inf",
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
        "max_tokens": "inf",
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
        "max_tokens": "inf",
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
        "max_tokens": "inf",
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
        "max_tokens": "inf",
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
        "max_tokens": "inf",
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
        "max_tokens": "inf",
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
        "max_tokens": "inf",
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
        "max_tokens": "inf",
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
        "max_tokens": "inf",
        "presence_penalty": 0,
    }
    assert generation.data[0].usage.input is not None
    assert generation.data[0].usage.output is not None
    assert generation.data[0].usage.total is not None
    assert generation.data[0].level == "ERROR"


def test_image_data_filtered():
    api = get_api()
    generation_name = create_uuid()

    openai.chat.completions.create(
        name=generation_name,
        model="gpt-4-vision-preview",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is in this image?"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEASABIAAD/2wBDAAMCAgMCAgMDAwMEAwMEBQgFBQQEBQoHBwYIDAoMDAsKCwsNDhIQDQ4RDgsLEBYQERMUFRUVDA8XGBYUGBIUFRT/wAALCAABAAEBAREA/8QAFAABAAAAAAAAAAAAAAAAAAAACf/EABQQAQAAAAAAAAAAAAAAAAAAAAD/2gAIAQEAAD8AKp//2Q=="
                        },
                    },
                ],
            }
        ],
        max_tokens=300,
    )

    generation = api.observations.get_many(name=generation_name, type="GENERATION")

    assert len(generation.data) == 1
    assert "data:image/jpeg;base64" not in generation.data[0].input


def test_image_filter_base64():
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What’s in this image?"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/jpeg;base64,base64_image"},
                },
            ],
        }
    ]
    result = filter_image_data(messages)

    print(result)

    assert result == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What’s in this image?"},
                {"type": "image_url"},
            ],
        }
    ]


def test_image_filter_url():
    result = filter_image_data(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What’s in this image?"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
                        },
                    },
                ],
            }
        ]
    )
    assert result == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What’s in this image?"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
                    },
                },
            ],
        }
    ]


def test_openai_assistant_creation():
    api = get_api()
    event_name = create_uuid()
    assistant_description = "Hello world, I am a test assistant"

    assistant = openai.beta.assistants.create(
        name=event_name,  # TODO: Assistant name is meant here, not event name
        model="gpt-3.5-turbo",
        description=assistant_description,
    )
    openai.flush_langfuse()

    # event = api.observations.get(observation_id=event_name) # TODO: returns 404
    observations = api.observations.get_many(name=event_name, type="EVENT")

    assert len(observations.data) != 0

    event = observations.data[0]
    assert event.name == event_name
    assert event.type == "EVENT"
    assert event.input["description"] == assistant_description
    assert event.output["description"] == assistant_description
    assert event.output["object"] == "assistant"
    assert event.output["id"] is not None
    assert event.output["instructions"] is None


def test_openai_assistant_creation_with_trace():
    api = get_api()
    event_name = create_uuid()
    assistant_description = "Hello world, I am a test assistant"
    trace_id = create_uuid()
    langfuse = Langfuse()
    langfuse.trace(id=trace_id)

    assistant = openai.beta.assistants.create(
        name=event_name,  # TODO: Assistent name is meant here, not event name
        model="gpt-3.5-turbo",
        description=assistant_description,
        trace_id=trace_id,
    )
    openai.flush_langfuse()

    # event = api.observations.get(observation_id=event_name) # TODO: returns 404
    observations = api.observations.get_many(name=event_name, type="EVENT")

    assert len(observations.data) != 0

    event = observations.data[0]
    assert event.trace_id == trace_id
    assert event.name == event_name
    assert event.type == "EVENT"
    assert event.input["description"] == assistant_description
    assert event.output["description"] == assistant_description
    assert event.output["object"] == "assistant"
    assert event.output["id"] is not None
    assert event.output["instructions"] is None


def test_openai_assistant_creation_all_attributes():
    api = get_api()
    event_name = create_uuid()
    trace_id = create_uuid()
    langfuse = Langfuse()
    langfuse.trace(id=trace_id)

    assistant_description = "Hello world, I am a test assistant"
    assistant_name = "My awesome assistant"
    assistant_instructions = "Please help me with this"
    assistant_tools = [{"type": "code_interpreter"}]
    assistant_metadata = {"my_key": "my_value", "my_other_key": 1, "my_third_key": True}
    assistant_file_ids = []  # TODO: Add file ids

    assistant = openai.beta.assistants.create(
        name=event_name,  # TODO: Assistent name is meant here, not event name
        model="gpt-3.5-turbo",
        instructions=assistant_instructions,
        tools=assistant_tools,
        metadata=assistant_metadata,  # TODO: Assistant metadata is meant here, not event metadata
        description=assistant_description,
        trace_id=trace_id,
    )
    openai.flush_langfuse()

    # event = api.observations.get(observation_id=event_name) # TODO: returns 404
    observations = api.observations.get_many(name=event_name, type="EVENT")

    assert len(observations.data) != 0

    event = observations.data[0]
    assert event.trace_id == trace_id
    assert event.name == event_name
    assert event.type == "EVENT"
    assert event.output["object"] == "assistant"
    assert event.output["id"] == assistant.id
    assert (
        event.input["description"]
        == event.output["description"]
        == assistant_description
        == assistant.description
    )
    assert (
        event.input["instructions"]
        == event.output["instructions"]
        == assistant_instructions
        == assistant.instructions
    )
    assert (
        event.input["tools"]
        == event.output["tools"]
        == assistant_tools
        == [{"type": a.type} for a in assistant.tools]
    )
    assert (
        event.input["file_ids"]
        == event.output["file_ids"]
        == assistant_file_ids
        == assistant.file_ids
    )
    # assert event["metadata"] == # TODO
    # assert event.input["name"] == assistant_name == assistant.name # TODO


def test_openai_thread_creation():
    api = get_api()
    trace_id = create_uuid()

    thread = openai.beta.threads.create(trace_id=trace_id)
    openai.flush_langfuse()

    observation = api.observations.get_many(name=thread.id)

    assert len(observation.data) == 1
    observation = observation.data[0]
    assert observation.name == thread.id


@pytest.fixture(scope="session")
def thread_and_observation():
    api = get_api()
    trace_id = create_uuid()

    thread = openai.beta.threads.create(trace_id=trace_id)
    openai.flush_langfuse()

    observations = api.observations.get_many(name=thread.id, type="EVENT")

    assert len(observations.data) == 1
    observation = observations.data[0]
    assert observation.trace_id == trace_id

    yield thread, observation


# test creation only possible if a thread exists
def test_openai_message_creation(thread_and_observation):
    thread, thread_creation_event = thread_and_observation
    api = get_api()
    observation_name = create_uuid()

    msg_content = "You are a hello world bot. You say hello world."
    msg_role = "user"
    message = openai.beta.threads.messages.create(
        thread_id=thread.id,  # needs to be created beforehand
        role=msg_role,
        content=msg_content,
        name=observation_name,
    )

    openai.flush_langfuse()
    observations = api.observations.get_many(name=observation_name, type="EVENT")

    assert len(observations.data) != 0

    observation = observations.data[0]
    assert observation.trace_id == thread_creation_event.trace_id
    assert observation.name == observation_name
    assert observation.type == "EVENT"
    assert observation.input["content"] == msg_content
    assert observation.input["role"] == msg_role == message.role
    assert (
        observation.input["thread_id"] == thread.id == observation.output["thread_id"]
    )
    assert observation.output["object"] == "thread.message"
    assert observation.output["id"] == message.id
    # TODO: test for full message object


def test_openai_run_creation():
    api = get_api()
    event_name = create_uuid()
    assistant_description = "Hello world, I am a test assistant"
    trace_id = create_uuid()

    assistant = openai.beta.assistants.create(
        name=event_name,  # TODO: Assistent name is meant here, not event name
        model="gpt-3.5-turbo",
        description=assistant_description,
        trace_id=trace_id,
    )

    thread = openai.beta.threads.create(trace_id=trace_id)

    # this return immediately
    run = openai.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant.id,
        instructions="Say hello world.",
        trace_id=trace_id,
    )

    openai.flush_langfuse()
    # user polling the status of the run
    while run.status == "queued" or run.status == "in_progress":
        run = openai.beta.threads.runs.retrieve(
            thread_id=thread.id,
            run_id=run.id,
            trace_id=trace_id,
        )
        time.sleep(0.05)

    openai.flush_langfuse()
    # TODO: Continue here
