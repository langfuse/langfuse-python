import importlib
import os
from time import sleep

import pytest
from pydantic import BaseModel

from langfuse._client.client import Langfuse
from tests.utils import create_uuid, encode_file_to_base64, get_api

langfuse = Langfuse()


@pytest.fixture(scope="module")
def openai():
    import openai

    from langfuse.openai import openai as _openai

    yield _openai

    importlib.reload(openai)


def test_openai_chat_completion(openai):
    generation_name = create_uuid()
    completion = openai.OpenAI().chat.completions.create(
        name=generation_name,
        model="gpt-3.5-turbo",
        messages=[
            {"role": "assistant", "content": "You are an expert mathematician"},
            {"role": "user", "content": "1 + 1 = "},
        ],
        temperature=0,
        metadata={"someKey": "someResponse"},
    )

    langfuse.flush()

    sleep(1)

    generation = get_api().observations.get_many(
        name=generation_name, type="GENERATION"
    )

    assert len(generation.data) != 0
    assert generation.data[0].name == generation_name
    assert generation.data[0].metadata["someKey"] == "someResponse"
    assert len(completion.choices) != 0
    assert generation.data[0].input == [
        {
            "content": "You are an expert mathematician",
            "role": "assistant",
        },
        {"content": "1 + 1 = ", "role": "user"},
    ]
    assert generation.data[0].type == "GENERATION"
    assert "gpt-3.5-turbo-0125" in generation.data[0].model
    assert generation.data[0].start_time is not None
    assert generation.data[0].end_time is not None
    assert generation.data[0].start_time < generation.data[0].end_time
    assert generation.data[0].model_parameters == {
        "temperature": 0,
        "top_p": 1,
        "frequency_penalty": 0,
        "max_tokens": "Infinity",
        "presence_penalty": 0,
    }
    assert generation.data[0].usage.input is not None
    assert generation.data[0].usage.output is not None
    assert generation.data[0].usage.total is not None
    assert "2" in generation.data[0].output["content"]
    assert generation.data[0].output["role"] == "assistant"


def test_openai_chat_completion_stream(openai):
    generation_name = create_uuid()
    completion = openai.OpenAI().chat.completions.create(
        name=generation_name,
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "1 + 1 = "}],
        temperature=0,
        metadata={"someKey": "someResponse"},
        stream=True,
    )

    assert iter(completion)

    chat_content = ""
    for i in completion:
        print("\n", i)
        chat_content += (i.choices[0].delta.content or "") if i.choices else ""

    assert len(chat_content) > 0

    langfuse.flush()
    sleep(3)

    generation = get_api().observations.get_many(
        name=generation_name, type="GENERATION"
    )

    assert len(generation.data) != 0
    assert generation.data[0].name == generation_name
    assert generation.data[0].metadata["someKey"] == "someResponse"

    assert generation.data[0].input == [{"content": "1 + 1 = ", "role": "user"}]
    assert generation.data[0].type == "GENERATION"
    assert "gpt-3.5-turbo-0125" in generation.data[0].model
    assert generation.data[0].start_time is not None
    assert generation.data[0].end_time is not None
    assert generation.data[0].start_time < generation.data[0].end_time
    assert generation.data[0].model_parameters == {
        "temperature": 0,
        "top_p": 1,
        "frequency_penalty": 0,
        "max_tokens": "Infinity",
        "presence_penalty": 0,
    }
    assert generation.data[0].usage.input is not None
    assert generation.data[0].usage.output is not None
    assert generation.data[0].usage.total is not None
    assert generation.data[0].output == 2
    assert generation.data[0].completion_start_time is not None

    # Completion start time for time-to-first-token
    assert generation.data[0].completion_start_time is not None
    assert generation.data[0].completion_start_time >= generation.data[0].start_time
    assert generation.data[0].completion_start_time <= generation.data[0].end_time


def test_openai_chat_completion_stream_with_next_iteration(openai):
    generation_name = create_uuid()
    completion = openai.OpenAI().chat.completions.create(
        name=generation_name,
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "1 + 1 = "}],
        temperature=0,
        metadata={"someKey": "someResponse"},
        stream=True,
    )

    assert iter(completion)

    chat_content = ""

    while True:
        try:
            c = next(completion)
            chat_content += (c.choices[0].delta.content or "") if c.choices else ""

        except StopIteration:
            break

    assert len(chat_content) > 0

    langfuse.flush()

    generation = get_api().observations.get_many(
        name=generation_name, type="GENERATION"
    )

    assert len(generation.data) != 0
    assert generation.data[0].name == generation_name
    assert generation.data[0].metadata["someKey"] == "someResponse"

    assert generation.data[0].input == [{"content": "1 + 1 = ", "role": "user"}]
    assert generation.data[0].type == "GENERATION"
    assert generation.data[0].model == "gpt-3.5-turbo-0125"
    assert generation.data[0].start_time is not None
    assert generation.data[0].end_time is not None
    assert generation.data[0].start_time < generation.data[0].end_time
    assert generation.data[0].model_parameters == {
        "temperature": 0,
        "top_p": 1,
        "frequency_penalty": 0,
        "max_tokens": "Infinity",
        "presence_penalty": 0,
    }
    assert generation.data[0].usage.input is not None
    assert generation.data[0].usage.output is not None
    assert generation.data[0].usage.total is not None
    assert generation.data[0].output == 2
    assert generation.data[0].completion_start_time is not None

    # Completion start time for time-to-first-token
    assert generation.data[0].completion_start_time is not None
    assert generation.data[0].completion_start_time >= generation.data[0].start_time
    assert generation.data[0].completion_start_time <= generation.data[0].end_time


def test_openai_chat_completion_stream_fail(openai):
    generation_name = create_uuid()
    openai.api_key = ""

    with pytest.raises(Exception):
        openai.OpenAI().chat.completions.create(
            name=generation_name,
            model="fake",
            messages=[{"role": "user", "content": "1 + 1 = "}],
            temperature=0,
            metadata={"someKey": "someResponse"},
            stream=True,
        )

    langfuse.flush()

    generation = get_api().observations.get_many(
        name=generation_name, type="GENERATION"
    )

    assert len(generation.data) != 0
    assert generation.data[0].name == generation_name
    assert generation.data[0].metadata["someKey"] == "someResponse"

    assert generation.data[0].input == [{"content": "1 + 1 = ", "role": "user"}]
    assert generation.data[0].type == "GENERATION"
    assert generation.data[0].model == "fake"
    assert generation.data[0].start_time is not None
    assert generation.data[0].end_time is not None
    assert generation.data[0].start_time < generation.data[0].end_time
    assert generation.data[0].model_parameters == {
        "temperature": 0,
        "top_p": 1,
        "frequency_penalty": 0,
        "max_tokens": "Infinity",
        "presence_penalty": 0,
    }
    assert generation.data[0].usage.input is not None
    assert generation.data[0].usage.output is not None
    assert generation.data[0].usage.total is not None
    assert generation.data[0].level == "ERROR"
    assert generation.data[0].status_message is not None
    assert generation.data[0].output is None

    openai.api_key = os.environ["OPENAI_API_KEY"]


def test_openai_chat_completion_with_langfuse_prompt(openai):
    generation_name = create_uuid()
    langfuse = Langfuse()
    prompt_name = create_uuid()
    langfuse.create_prompt(
        name=prompt_name, prompt="test prompt", labels=["production"]
    )

    prompt_client = langfuse.get_prompt(name=prompt_name)

    openai.OpenAI().chat.completions.create(
        name=generation_name,
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Make me laugh"}],
        langfuse_prompt=prompt_client,
    )

    langfuse.flush()

    generation = get_api().observations.get_many(
        name=generation_name, type="GENERATION"
    )

    assert len(generation.data) != 0
    assert generation.data[0].name == generation_name
    assert isinstance(generation.data[0].prompt_id, str)


def test_openai_chat_completion_fail(openai):
    generation_name = create_uuid()

    with pytest.raises(Exception):
        openai.OpenAI().chat.completions.create(
            name=generation_name,
            model="fake",
            messages=[{"role": "user", "content": "1 + 1 = "}],
            temperature=0,
            metadata={"someKey": "someResponse"},
        )

    langfuse.flush()

    generation = get_api().observations.get_many(
        name=generation_name, type="GENERATION"
    )

    assert len(generation.data) != 0
    assert generation.data[0].name == generation_name
    assert generation.data[0].metadata["someKey"] == "someResponse"
    assert generation.data[0].input == [{"content": "1 + 1 = ", "role": "user"}]
    assert generation.data[0].type == "GENERATION"
    assert generation.data[0].model == "fake"
    assert generation.data[0].level == "ERROR"
    assert generation.data[0].start_time is not None
    assert generation.data[0].end_time is not None
    assert generation.data[0].status_message is not None
    assert generation.data[0].start_time < generation.data[0].end_time
    assert generation.data[0].model_parameters == {
        "temperature": 0,
        "top_p": 1,
        "frequency_penalty": 0,
        "max_tokens": "Infinity",
        "presence_penalty": 0,
    }
    assert generation.data[0].output is None

    openai.api_key = os.environ["OPENAI_API_KEY"]


def test_openai_chat_completion_without_extra_param(openai):
    completion = openai.OpenAI().chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "1 + 1 = "}],
        temperature=0,
        metadata={"someKey": "someResponse"},
    )

    assert len(completion.choices) != 0


def test_openai_chat_completion_two_calls(openai):
    generation_name = create_uuid()
    completion = openai.OpenAI().chat.completions.create(
        name=generation_name,
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "1 + 1 = "}],
        temperature=0,
        metadata={"someKey": "someResponse"},
    )

    generation_name_2 = create_uuid()

    completion_2 = openai.OpenAI().chat.completions.create(
        name=generation_name_2,
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "2 + 2 = "}],
        temperature=0,
        metadata={"someKey": "someResponse"},
    )

    langfuse.flush()

    generation = get_api().observations.get_many(
        name=generation_name, type="GENERATION"
    )

    assert len(generation.data) != 0
    assert generation.data[0].name == generation_name
    assert len(completion.choices) != 0

    assert generation.data[0].input == [{"content": "1 + 1 = ", "role": "user"}]

    generation_2 = get_api().observations.get_many(
        name=generation_name_2, type="GENERATION"
    )

    assert len(generation_2.data) != 0
    assert generation_2.data[0].name == generation_name_2
    assert len(completion_2.choices) != 0

    assert generation_2.data[0].input == [{"content": "2 + 2 = ", "role": "user"}]


def test_openai_chat_completion_with_seed(openai):
    generation_name = create_uuid()
    completion = openai.OpenAI().chat.completions.create(
        name=generation_name,
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "1 + 1 = "}],
        temperature=0,
        seed=123,
        metadata={"someKey": "someResponse"},
    )

    langfuse.flush()

    generation = get_api().observations.get_many(
        name=generation_name, type="GENERATION"
    )

    assert generation.data[0].model_parameters == {
        "temperature": 0,
        "top_p": 1,
        "frequency_penalty": 0,
        "max_tokens": "Infinity",
        "presence_penalty": 0,
        "seed": 123,
    }
    assert len(completion.choices) != 0


def test_openai_completion(openai):
    generation_name = create_uuid()
    completion = openai.OpenAI().completions.create(
        name=generation_name,
        model="gpt-3.5-turbo-instruct",
        prompt="1 + 1 = ",
        temperature=0,
        metadata={"someKey": "someResponse"},
    )

    langfuse.flush()

    generation = get_api().observations.get_many(
        name=generation_name, type="GENERATION"
    )

    assert len(generation.data) != 0
    assert generation.data[0].name == generation_name
    assert generation.data[0].metadata["someKey"] == "someResponse"
    assert len(completion.choices) != 0
    assert completion.choices[0].text == generation.data[0].output
    assert generation.data[0].input == "1 + 1 = "
    assert generation.data[0].type == "GENERATION"
    assert "gpt-3.5-turbo-instruct" in generation.data[0].model
    assert generation.data[0].start_time is not None
    assert generation.data[0].end_time is not None
    assert generation.data[0].start_time < generation.data[0].end_time
    assert generation.data[0].model_parameters == {
        "temperature": 0,
        "top_p": 1,
        "frequency_penalty": 0,
        "max_tokens": "Infinity",
        "presence_penalty": 0,
    }
    assert generation.data[0].usage.input is not None
    assert generation.data[0].usage.output is not None
    assert generation.data[0].usage.total is not None
    assert generation.data[0].output == "2\n\n1 + 2 = 3\n\n2 + 3 = "


def test_openai_completion_stream(openai):
    generation_name = create_uuid()
    completion = openai.OpenAI().completions.create(
        name=generation_name,
        model="gpt-3.5-turbo-instruct",
        prompt="1 + 1 = ",
        temperature=0,
        metadata={"someKey": "someResponse"},
        stream=True,
    )

    assert iter(completion)
    content = ""
    for i in completion:
        content += (i.choices[0].text or "") if i.choices else ""

    langfuse.flush()

    assert len(content) > 0

    generation = get_api().observations.get_many(
        name=generation_name, type="GENERATION"
    )

    assert len(generation.data) != 0
    assert generation.data[0].name == generation_name
    assert generation.data[0].metadata["someKey"] == "someResponse"

    assert generation.data[0].input == "1 + 1 = "
    assert generation.data[0].type == "GENERATION"
    assert "gpt-3.5-turbo-instruct" in generation.data[0].model
    assert generation.data[0].start_time is not None
    assert generation.data[0].end_time is not None
    assert generation.data[0].start_time < generation.data[0].end_time
    assert generation.data[0].model_parameters == {
        "temperature": 0,
        "top_p": 1,
        "frequency_penalty": 0,
        "max_tokens": "Infinity",
        "presence_penalty": 0,
    }
    assert generation.data[0].usage.input is not None
    assert generation.data[0].usage.output is not None
    assert generation.data[0].usage.total is not None
    assert generation.data[0].output == "2\n\n1 + 2 = 3\n\n2 + 3 = "
    assert generation.data[0].completion_start_time is not None

    # Completion start time for time-to-first-token
    assert generation.data[0].completion_start_time is not None
    assert generation.data[0].completion_start_time >= generation.data[0].start_time
    assert generation.data[0].completion_start_time <= generation.data[0].end_time


def test_openai_completion_fail(openai):
    generation_name = create_uuid()

    openai.api_key = ""

    with pytest.raises(Exception):
        openai.OpenAI().completions.create(
            name=generation_name,
            model="fake",
            prompt="1 + 1 = ",
            temperature=0,
            metadata={"someKey": "someResponse"},
        )

    langfuse.flush()

    generation = get_api().observations.get_many(
        name=generation_name, type="GENERATION"
    )

    assert len(generation.data) != 0
    assert generation.data[0].name == generation_name
    assert generation.data[0].metadata["someKey"] == "someResponse"
    assert generation.data[0].input == "1 + 1 = "
    assert generation.data[0].type == "GENERATION"
    assert generation.data[0].model == "fake"
    assert generation.data[0].level == "ERROR"
    assert generation.data[0].start_time is not None
    assert generation.data[0].end_time is not None
    assert generation.data[0].status_message is not None
    assert generation.data[0].start_time < generation.data[0].end_time
    assert generation.data[0].model_parameters == {
        "temperature": 0,
        "top_p": 1,
        "frequency_penalty": 0,
        "max_tokens": "Infinity",
        "presence_penalty": 0,
    }
    assert generation.data[0].output is None

    openai.api_key = os.environ["OPENAI_API_KEY"]


def test_openai_completion_stream_fail(openai):
    generation_name = create_uuid()
    openai.api_key = ""

    with pytest.raises(Exception):
        openai.OpenAI().completions.create(
            name=generation_name,
            model="gpt-3.5-turbo",
            prompt="1 + 1 = ",
            temperature=0,
            metadata={"someKey": "someResponse"},
            stream=True,
        )

    langfuse.flush()

    generation = get_api().observations.get_many(
        name=generation_name, type="GENERATION"
    )

    assert len(generation.data) != 0
    assert generation.data[0].name == generation_name
    assert generation.data[0].metadata["someKey"] == "someResponse"

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
        "max_tokens": "Infinity",
        "presence_penalty": 0,
    }
    assert generation.data[0].usage.input is not None
    assert generation.data[0].usage.output is not None
    assert generation.data[0].usage.total is not None
    assert generation.data[0].level == "ERROR"
    assert generation.data[0].status_message is not None
    assert generation.data[0].output is None

    openai.api_key = os.environ["OPENAI_API_KEY"]


def test_openai_completion_with_langfuse_prompt(openai):
    generation_name = create_uuid()
    langfuse = Langfuse()
    prompt_name = create_uuid()
    prompt_client = langfuse.create_prompt(
        name=prompt_name, prompt="test prompt", labels=["production"]
    )
    openai.OpenAI().completions.create(
        name=generation_name,
        model="gpt-3.5-turbo-instruct",
        prompt="1 + 1 = ",
        temperature=0,
        metadata={"someKey": "someResponse"},
        langfuse_prompt=prompt_client,
    )

    langfuse.flush()

    generation = get_api().observations.get_many(
        name=generation_name, type="GENERATION"
    )

    assert len(generation.data) != 0
    assert generation.data[0].name == generation_name
    assert isinstance(generation.data[0].prompt_id, str)


def test_fails_wrong_name(openai):
    with pytest.raises(TypeError, match="name must be a string"):
        openai.OpenAI().completions.create(
            name={"key": "generation_name"},
            model="gpt-3.5-turbo-instruct",
            prompt="1 + 1 = ",
            temperature=0,
        )


def test_fails_wrong_metadata(openai):
    with pytest.raises(TypeError, match="metadata must be a dictionary"):
        openai.OpenAI().completions.create(
            metadata="metadata",
            model="gpt-3.5-turbo-instruct",
            prompt="1 + 1 = ",
            temperature=0,
        )


def test_fails_wrong_trace_id(openai):
    with pytest.raises(TypeError, match="trace_id must be a string"):
        openai.OpenAI().completions.create(
            trace_id={"trace_id": "metadata"},
            model="gpt-3.5-turbo-instruct",
            prompt="1 + 1 = ",
            temperature=0,
        )


@pytest.mark.asyncio
async def test_async_chat(openai):
    client = openai.AsyncOpenAI()
    generation_name = create_uuid()

    completion = await client.chat.completions.create(
        messages=[{"role": "user", "content": "1 + 1 = "}],
        model="gpt-3.5-turbo",
        name=generation_name,
    )

    langfuse.flush()

    generation = get_api().observations.get_many(
        name=generation_name, type="GENERATION"
    )

    assert len(generation.data) != 0
    assert generation.data[0].name == generation_name
    assert len(completion.choices) != 0

    assert generation.data[0].input == [{"content": "1 + 1 = ", "role": "user"}]
    assert generation.data[0].type == "GENERATION"
    assert generation.data[0].model == "gpt-3.5-turbo-0125"
    assert generation.data[0].start_time is not None
    assert generation.data[0].end_time is not None
    assert generation.data[0].start_time < generation.data[0].end_time
    assert generation.data[0].model_parameters == {
        "temperature": 1,
        "top_p": 1,
        "frequency_penalty": 0,
        "max_tokens": "Infinity",
        "presence_penalty": 0,
    }
    assert generation.data[0].usage.input is not None
    assert generation.data[0].usage.output is not None
    assert generation.data[0].usage.total is not None
    assert "2" in generation.data[0].output["content"]
    assert generation.data[0].output["role"] == "assistant"


@pytest.mark.asyncio
async def test_async_chat_stream(openai):
    client = openai.AsyncOpenAI()

    generation_name = create_uuid()

    completion = await client.chat.completions.create(
        messages=[{"role": "user", "content": "1 + 1 = "}],
        model="gpt-3.5-turbo",
        name=generation_name,
        stream=True,
    )

    async for c in completion:
        print(c)

    langfuse.flush()

    generation = get_api().observations.get_many(
        name=generation_name, type="GENERATION"
    )

    assert len(generation.data) != 0
    assert generation.data[0].name == generation_name
    assert generation.data[0].input == [{"content": "1 + 1 = ", "role": "user"}]
    assert generation.data[0].type == "GENERATION"
    assert generation.data[0].model == "gpt-3.5-turbo-0125"
    assert generation.data[0].start_time is not None
    assert generation.data[0].end_time is not None
    assert generation.data[0].start_time < generation.data[0].end_time
    assert generation.data[0].model_parameters == {
        "temperature": 1,
        "top_p": 1,
        "frequency_penalty": 0,
        "max_tokens": "Infinity",
        "presence_penalty": 0,
    }
    assert generation.data[0].usage.input is not None
    assert generation.data[0].usage.output is not None
    assert generation.data[0].usage.total is not None
    assert "2" in str(generation.data[0].output)

    # Completion start time for time-to-first-token
    assert generation.data[0].completion_start_time is not None
    assert generation.data[0].completion_start_time >= generation.data[0].start_time
    assert generation.data[0].completion_start_time <= generation.data[0].end_time


@pytest.mark.asyncio
async def test_async_chat_stream_with_anext(openai):
    client = openai.AsyncOpenAI()

    generation_name = create_uuid()

    completion = await client.chat.completions.create(
        messages=[{"role": "user", "content": "Give me a one-liner joke"}],
        model="gpt-3.5-turbo",
        name=generation_name,
        stream=True,
    )

    result = ""

    while True:
        try:
            c = await completion.__anext__()

            result += (c.choices[0].delta.content or "") if c.choices else ""

        except StopAsyncIteration:
            break

    langfuse.flush()

    print(result)

    generation = get_api().observations.get_many(
        name=generation_name, type="GENERATION"
    )

    assert len(generation.data) != 0
    assert generation.data[0].name == generation_name
    assert generation.data[0].input == [
        {"content": "Give me a one-liner joke", "role": "user"}
    ]
    assert generation.data[0].type == "GENERATION"
    assert generation.data[0].model == "gpt-3.5-turbo-0125"
    assert generation.data[0].start_time is not None
    assert generation.data[0].end_time is not None
    assert generation.data[0].start_time < generation.data[0].end_time
    assert generation.data[0].model_parameters == {
        "temperature": 1,
        "top_p": 1,
        "frequency_penalty": 0,
        "max_tokens": "Infinity",
        "presence_penalty": 0,
    }
    assert generation.data[0].usage.input is not None
    assert generation.data[0].usage.output is not None
    assert generation.data[0].usage.total is not None

    # Completion start time for time-to-first-token
    assert generation.data[0].completion_start_time is not None
    assert generation.data[0].completion_start_time >= generation.data[0].start_time
    assert generation.data[0].completion_start_time <= generation.data[0].end_time


def test_openai_function_call(openai):
    from typing import List

    from pydantic import BaseModel

    generation_name = create_uuid()

    class StepByStepAIResponse(BaseModel):
        title: str
        steps: List[str]

    import json

    response = openai.OpenAI().chat.completions.create(
        name=generation_name,
        model="gpt-3.5-turbo",
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

    langfuse.flush()

    generation = get_api().observations.get_many(
        name=generation_name, type="GENERATION"
    )

    assert len(generation.data) != 0
    assert generation.data[0].name == generation_name
    assert generation.data[0].output is not None
    assert "function_call" in generation.data[0].output

    assert output["title"] is not None


def test_openai_function_call_streamed(openai):
    from typing import List

    from pydantic import BaseModel

    generation_name = create_uuid()

    class StepByStepAIResponse(BaseModel):
        title: str
        steps: List[str]

    response = openai.OpenAI().chat.completions.create(
        name=generation_name,
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Explain how to assemble a PC"}],
        functions=[
            {
                "name": "get_answer_for_user_query",
                "description": "Get user answer in series of steps",
                "parameters": StepByStepAIResponse.schema(),
            }
        ],
        function_call={"name": "get_answer_for_user_query"},
        stream=True,
    )

    # Consume the stream
    for _ in response:
        pass

    langfuse.flush()

    generation = get_api().observations.get_many(
        name=generation_name, type="GENERATION"
    )

    assert len(generation.data) != 0
    assert generation.data[0].name == generation_name
    assert generation.data[0].output is not None
    assert "function_call" in generation.data[0].output


def test_openai_tool_call(openai):
    generation_name = create_uuid()

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                    "required": ["location"],
                },
            },
        }
    ]
    messages = [{"role": "user", "content": "What's the weather like in Boston today?"}]
    openai.OpenAI().chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        tools=tools,
        tool_choice="auto",
        name=generation_name,
    )

    langfuse.flush()

    generation = get_api().observations.get_many(
        name=generation_name, type="GENERATION"
    )

    assert len(generation.data) != 0
    assert generation.data[0].name == generation_name
    assert (
        generation.data[0].output["tool_calls"][0]["function"]["name"]
        == "get_current_weather"
    )
    assert (
        generation.data[0].output["tool_calls"][0]["function"]["arguments"] is not None
    )
    assert generation.data[0].input["tools"] == tools
    assert generation.data[0].input["messages"] == messages


def test_openai_tool_call_streamed(openai):
    generation_name = create_uuid()

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                    "required": ["location"],
                },
            },
        }
    ]
    messages = [{"role": "user", "content": "What's the weather like in Boston today?"}]
    response = openai.OpenAI().chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        tools=tools,
        tool_choice="required",
        name=generation_name,
        stream=True,
    )

    # Consume the stream
    for _ in response:
        pass

    langfuse.flush()

    generation = get_api().observations.get_many(
        name=generation_name, type="GENERATION"
    )

    assert len(generation.data) != 0
    assert generation.data[0].name == generation_name

    assert (
        generation.data[0].output["tool_calls"][0]["function"]["name"]
        == "get_current_weather"
    )
    assert (
        generation.data[0].output["tool_calls"][0]["function"]["arguments"] is not None
    )
    assert generation.data[0].input["tools"] == tools
    assert generation.data[0].input["messages"] == messages


def test_langchain_integration(openai):
    from langchain_openai import ChatOpenAI

    chat = ChatOpenAI(model="gpt-4o")

    result = ""

    for chunk in chat.stream("Hello, how are you?"):
        result += chunk.content

    print(result)
    assert result != ""


def test_structured_output_response_format_kwarg(openai):
    generation_name = (
        "test_structured_output_response_format_kwarg" + create_uuid()[0:10]
    )

    json_schema = {
        "name": "math_response",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "steps": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "explanation": {"type": "string"},
                            "output": {"type": "string"},
                        },
                        "required": ["explanation", "output"],
                        "additionalProperties": False,
                    },
                },
                "final_answer": {"type": "string"},
            },
            "required": ["steps", "final_answer"],
            "additionalProperties": False,
        },
    }

    openai.OpenAI().chat.completions.create(
        name=generation_name,
        model="gpt-4o-2024-08-06",
        messages=[
            {"role": "system", "content": "You are a helpful math tutor."},
            {"role": "user", "content": "solve 8x + 31 = 2"},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": json_schema,
        },
        metadata={"someKey": "someResponse"},
    )

    langfuse.flush()

    generation = get_api().observations.get_many(
        name=generation_name, type="GENERATION"
    )

    assert len(generation.data) != 0
    assert generation.data[0].name == generation_name
    assert generation.data[0].metadata["someKey"] == "someResponse"
    assert generation.data[0].metadata["response_format"] == {
        "type": "json_schema",
        "json_schema": json_schema,
    }

    assert generation.data[0].input == [
        {"role": "system", "content": "You are a helpful math tutor."},
        {"content": "solve 8x + 31 = 2", "role": "user"},
    ]
    assert generation.data[0].type == "GENERATION"
    assert generation.data[0].model == "gpt-4o-2024-08-06"
    assert generation.data[0].start_time is not None
    assert generation.data[0].end_time is not None
    assert generation.data[0].start_time < generation.data[0].end_time
    assert generation.data[0].model_parameters == {
        "temperature": 1,
        "top_p": 1,
        "frequency_penalty": 0,
        "max_tokens": "Infinity",
        "presence_penalty": 0,
    }
    assert generation.data[0].usage.input is not None
    assert generation.data[0].usage.output is not None
    assert generation.data[0].usage.total is not None
    assert generation.data[0].output["role"] == "assistant"


def test_structured_output_beta_completions_parse(openai):
    from typing import List

    from packaging.version import Version

    class CalendarEvent(BaseModel):
        name: str
        date: str
        participants: List[str]

    generation_name = create_uuid()

    params = {
        "model": "gpt-4o",
        "messages": [
            {"role": "system", "content": "Extract the event information."},
            {
                "role": "user",
                "content": "Alice and Bob are going to a science fair on Friday.",
            },
        ],
        "response_format": CalendarEvent,
        "name": generation_name,
    }

    # The beta API is only wrapped for this version range. prior to that, implicitly another wrapped method was called
    if Version(openai.__version__) < Version("1.50.0"):
        params.pop("name")

    openai.OpenAI().chat.completions.parse(**params)

    langfuse.flush()

    if Version(openai.__version__) >= Version("1.50.0"):
        # Check the trace and observation properties
        generation = get_api().observations.get_many(
            name=generation_name, type="GENERATION"
        )

        assert len(generation.data) == 1
        assert generation.data[0].name == generation_name
        assert generation.data[0].type == "GENERATION"
        assert "gpt-4o" in generation.data[0].model
        assert generation.data[0].start_time is not None
        assert generation.data[0].end_time is not None
        assert generation.data[0].start_time < generation.data[0].end_time

        # Check input and output
        assert len(generation.data[0].input) == 2
        assert generation.data[0].input[0]["role"] == "system"
        assert generation.data[0].input[1]["role"] == "user"
        assert isinstance(generation.data[0].output, dict)
        assert "name" in generation.data[0].output["content"]
        assert "date" in generation.data[0].output["content"]
        assert "participants" in generation.data[0].output["content"]

        # Check usage
        assert generation.data[0].usage.input is not None
        assert generation.data[0].usage.output is not None
        assert generation.data[0].usage.total is not None


@pytest.mark.asyncio
async def test_close_async_stream(openai):
    client = openai.AsyncOpenAI()
    generation_name = create_uuid()

    stream = await client.chat.completions.create(
        messages=[{"role": "user", "content": "1 + 1 = "}],
        model="gpt-3.5-turbo",
        name=generation_name,
        stream=True,
    )

    async for token in stream:
        print(token)

    await stream.close()

    langfuse.flush()

    generation = get_api().observations.get_many(
        name=generation_name, type="GENERATION"
    )

    assert len(generation.data) != 0
    assert generation.data[0].name == generation_name
    assert generation.data[0].input == [{"content": "1 + 1 = ", "role": "user"}]
    assert generation.data[0].type == "GENERATION"
    assert generation.data[0].model == "gpt-3.5-turbo-0125"
    assert generation.data[0].start_time is not None
    assert generation.data[0].end_time is not None
    assert generation.data[0].start_time < generation.data[0].end_time
    assert generation.data[0].model_parameters == {
        "temperature": 1,
        "top_p": 1,
        "frequency_penalty": 0,
        "max_tokens": "Infinity",
        "presence_penalty": 0,
    }
    assert generation.data[0].usage.input is not None
    assert generation.data[0].usage.output is not None
    assert generation.data[0].usage.total is not None
    assert "2" in str(generation.data[0].output)

    # Completion start time for time-to-first-token
    assert generation.data[0].completion_start_time is not None
    assert generation.data[0].completion_start_time >= generation.data[0].start_time
    assert generation.data[0].completion_start_time <= generation.data[0].end_time


def test_base_64_image_input(openai):
    client = openai.OpenAI()
    generation_name = "test_base_64_image_input" + create_uuid()[:8]

    content_path = "static/puton.jpg"
    content_type = "image/jpeg"

    base64_image = encode_file_to_base64(content_path)

    client.chat.completions.create(
        name=generation_name,
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What’s in this image?"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{content_type};base64,{base64_image}"
                        },
                    },
                ],
            }
        ],
        max_tokens=300,
    )

    langfuse.flush()

    generation = get_api().observations.get_many(
        name=generation_name, type="GENERATION"
    )

    assert len(generation.data) != 0
    assert generation.data[0].name == generation_name
    assert generation.data[0].input[0]["content"][0]["text"] == "What’s in this image?"
    assert (
        f"@@@langfuseMedia:type={content_type}|id="
        in generation.data[0].input[0]["content"][1]["image_url"]["url"]
    )
    assert generation.data[0].type == "GENERATION"
    assert "gpt-4o-mini" in generation.data[0].model
    assert generation.data[0].start_time is not None
    assert generation.data[0].end_time is not None
    assert generation.data[0].start_time < generation.data[0].end_time
    assert generation.data[0].usage.input is not None
    assert generation.data[0].usage.output is not None
    assert generation.data[0].usage.total is not None
    assert "dog" in generation.data[0].output["content"]


def test_audio_input_and_output(openai):
    client = openai.OpenAI()
    openai.langfuse_debug = True
    generation_name = "test_audio_input_and_output" + create_uuid()[:8]

    content_path = "static/joke_prompt.wav"
    base64_string = encode_file_to_base64(content_path)

    client.chat.completions.create(
        name=generation_name,
        model="gpt-4o-audio-preview",
        modalities=["text", "audio"],
        audio={"voice": "alloy", "format": "wav"},
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Do what this recording says."},
                    {
                        "type": "input_audio",
                        "input_audio": {"data": base64_string, "format": "wav"},
                    },
                ],
            },
        ],
    )

    langfuse.flush()

    generation = get_api().observations.get_many(
        name=generation_name, type="GENERATION"
    )

    assert len(generation.data) != 0
    assert generation.data[0].name == generation_name
    assert (
        generation.data[0].input[0]["content"][0]["text"]
        == "Do what this recording says."
    )
    assert (
        "@@@langfuseMedia:type=audio/wav|id="
        in generation.data[0].input[0]["content"][1]["input_audio"]["data"]
    )
    assert generation.data[0].type == "GENERATION"
    assert "gpt-4o-audio-preview" in generation.data[0].model
    assert generation.data[0].start_time is not None
    assert generation.data[0].end_time is not None
    assert generation.data[0].start_time < generation.data[0].end_time
    assert generation.data[0].usage.input is not None
    assert generation.data[0].usage.output is not None
    assert generation.data[0].usage.total is not None
    print(generation.data[0].output)
    assert (
        "@@@langfuseMedia:type=audio/wav|id="
        in generation.data[0].output["audio"]["data"]
    )


def test_response_api_text_input(openai):
    client = openai.OpenAI()
    generation_name = "test_response_api_text_input" + create_uuid()[:8]

    client.responses.create(
        name=generation_name,
        model="gpt-4o",
        input="Tell me a three sentence bedtime story about a unicorn.",
    )

    langfuse.flush()
    generation = get_api().observations.get_many(
        name=generation_name, type="GENERATION"
    )

    assert len(generation.data) != 0
    generationData = generation.data[0]
    assert generationData.name == generation_name
    assert (
        generation.data[0].input
        == "Tell me a three sentence bedtime story about a unicorn."
    )
    assert generationData.type == "GENERATION"
    assert "gpt-4o" in generationData.model
    assert generationData.start_time is not None
    assert generationData.end_time is not None
    assert generationData.start_time < generationData.end_time
    assert generationData.usage.input is not None
    assert generationData.usage.output is not None
    assert generationData.usage.total is not None
    assert generationData.output is not None


@pytest.mark.skip("Flaky")
def test_response_api_image_input(openai):
    client = openai.OpenAI()
    generation_name = "test_response_api_image_input" + create_uuid()[:8]

    client.responses.create(
        name=generation_name,
        model="gpt-4o",
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "what is in this image?"},
                    {
                        "type": "input_image",
                        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
                    },
                ],
            }
        ],
    )

    langfuse.flush()

    generation = get_api().observations.get_many(
        name=generation_name, type="GENERATION"
    )

    assert len(generation.data) != 0
    generationData = generation.data[0]
    assert generationData.name == generation_name
    assert generation.data[0].input[0]["content"][0]["text"] == "what is in this image?"
    assert generationData.type == "GENERATION"
    assert "gpt-4o" in generationData.model
    assert generationData.start_time is not None
    assert generationData.end_time is not None
    assert generationData.start_time < generationData.end_time
    assert generationData.usage.input is not None
    assert generationData.usage.output is not None
    assert generationData.usage.total is not None
    assert generationData.output is not None


def test_response_api_web_search(openai):
    client = openai.OpenAI()
    generation_name = "test_response_api_web_search" + create_uuid()[:8]

    client.responses.create(
        name=generation_name,
        model="gpt-4o",
        tools=[{"type": "web_search_preview"}],
        input="What was a positive news story from today?",
    )

    langfuse.flush()

    generation = get_api().observations.get_many(
        name=generation_name, type="GENERATION"
    )

    assert len(generation.data) != 0
    generationData = generation.data[0]
    assert generationData.name == generation_name
    assert generationData.input == "What was a positive news story from today?"
    assert generationData.type == "GENERATION"
    assert "gpt-4o" in generationData.model
    assert generationData.start_time is not None
    assert generationData.end_time is not None
    assert generationData.start_time < generationData.end_time
    assert generationData.usage.input is not None
    assert generationData.usage.output is not None
    assert generationData.usage.total is not None
    assert generationData.output is not None
    assert generationData.metadata is not None


def test_response_api_streaming(openai):
    client = openai.OpenAI()
    generation_name = "test_response_api_streaming" + create_uuid()[:8]

    response = client.responses.create(
        name=generation_name,
        model="gpt-4o",
        instructions="You are a helpful assistant.",
        input="Hello!",
        stream=True,
    )

    for _ in response:
        continue

    langfuse.flush()

    generation = get_api().observations.get_many(
        name=generation_name, type="GENERATION"
    )

    assert len(generation.data) != 0
    generationData = generation.data[0]
    assert generationData.name == generation_name
    assert generation.data[0].input == "Hello!"
    assert generationData.type == "GENERATION"
    assert "gpt-4o" in generationData.model
    assert generationData.start_time is not None
    assert generationData.end_time is not None
    assert generationData.start_time < generationData.end_time
    assert generationData.usage.input is not None
    assert generationData.usage.output is not None
    assert generationData.usage.total is not None
    assert generationData.output is not None
    assert generationData.metadata is not None
    assert generationData.metadata["instructions"] == "You are a helpful assistant."


def test_response_api_functions(openai):
    client = openai.OpenAI()
    generation_name = "test_response_api_functions" + create_uuid()[:8]

    tools = [
        {
            "type": "function",
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location", "unit"],
            },
        }
    ]

    client.responses.create(
        name=generation_name,
        model="gpt-4o",
        tools=tools,
        input="What is the weather like in Boston today?",
        tool_choice="auto",
    )

    langfuse.flush()

    generation = get_api().observations.get_many(
        name=generation_name, type="GENERATION"
    )

    assert len(generation.data) != 0
    generationData = generation.data[0]
    assert generationData.name == generation_name
    assert generation.data[0].input == "What is the weather like in Boston today?"
    assert generationData.type == "GENERATION"
    assert "gpt-4o" in generationData.model
    assert generationData.start_time is not None
    assert generationData.end_time is not None
    assert generationData.start_time < generationData.end_time
    assert generationData.usage.input is not None
    assert generationData.usage.output is not None
    assert generationData.usage.total is not None
    assert generationData.output is not None
    assert generationData.metadata is not None


def test_response_api_reasoning(openai):
    client = openai.OpenAI()
    generation_name = "test_response_api_reasoning" + create_uuid()[:8]

    client.responses.create(
        name=generation_name,
        model="o3-mini",
        input="How much wood would a woodchuck chuck?",
        reasoning={"effort": "high"},
    )
    langfuse.flush()

    generation = get_api().observations.get_many(
        name=generation_name, type="GENERATION"
    )

    assert len(generation.data) != 0
    generationData = generation.data[0]
    assert generationData.name == generation_name
    assert generation.data[0].input == "How much wood would a woodchuck chuck?"
    assert generationData.type == "GENERATION"
    assert "o3-mini" in generationData.model
    assert generationData.start_time is not None
    assert generationData.end_time is not None
    assert generationData.start_time < generationData.end_time
    assert generationData.usage.input is not None
    assert generationData.usage.output is not None
    assert generationData.usage.total is not None
    assert generationData.output is not None
    assert generationData.metadata is not None


def test_openai_embeddings(openai):
    embedding_name = create_uuid()
    openai.OpenAI().embeddings.create(
        name=embedding_name,
        model="text-embedding-ada-002",
        input="The quick brown fox jumps over the lazy dog",
        metadata={"test_key": "test_value"},
    )

    langfuse.flush()
    sleep(1)

    embedding = get_api().observations.get_many(name=embedding_name, type="EMBEDDING")

    assert len(embedding.data) != 0
    embedding_data = embedding.data[0]
    assert embedding_data.name == embedding_name
    assert embedding_data.metadata["test_key"] == "test_value"
    assert embedding_data.input == "The quick brown fox jumps over the lazy dog"
    assert embedding_data.type == "EMBEDDING"
    assert "text-embedding-ada-002" in embedding_data.model
    assert embedding_data.start_time is not None
    assert embedding_data.end_time is not None
    assert embedding_data.start_time < embedding_data.end_time
    assert embedding_data.usage.input is not None
    assert embedding_data.usage.total is not None
    assert embedding_data.output is not None
    assert "dimensions" in embedding_data.output
    assert "count" in embedding_data.output
    assert embedding_data.output["count"] == 1


def test_openai_embeddings_multiple_inputs(openai):
    embedding_name = create_uuid()
    inputs = ["The quick brown fox", "jumps over the lazy dog", "Hello world"]

    openai.OpenAI().embeddings.create(
        name=embedding_name,
        model="text-embedding-ada-002",
        input=inputs,
        metadata={"batch_size": len(inputs)},
    )

    langfuse.flush()
    sleep(1)

    embedding = get_api().observations.get_many(name=embedding_name, type="EMBEDDING")

    assert len(embedding.data) != 0
    embedding_data = embedding.data[0]
    assert embedding_data.name == embedding_name
    assert embedding_data.input == inputs
    assert embedding_data.type == "EMBEDDING"
    assert "text-embedding-ada-002" in embedding_data.model
    assert embedding_data.usage.input is not None
    assert embedding_data.usage.total is not None
    assert embedding_data.output["count"] == len(inputs)


@pytest.mark.asyncio
async def test_async_openai_embeddings(openai):
    client = openai.AsyncOpenAI()
    embedding_name = create_uuid()
    print(embedding_name)

    result = await client.embeddings.create(
        name=embedding_name,
        model="text-embedding-ada-002",
        input="Async embedding test",
        metadata={"async": True},
    )

    print("result:", result.usage)

    langfuse.flush()
    sleep(1)

    embedding = get_api().observations.get_many(name=embedding_name, type="EMBEDDING")

    assert len(embedding.data) != 0
    embedding_data = embedding.data[0]
    assert embedding_data.name == embedding_name
    assert embedding_data.input == "Async embedding test"
    assert embedding_data.type == "EMBEDDING"
    assert "text-embedding-ada-002" in embedding_data.model
    assert embedding_data.metadata["async"] is True
    assert embedding_data.usage.input is not None
    assert embedding_data.usage.total is not None
