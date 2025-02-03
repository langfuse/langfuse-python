import os

import pytest
from openai import APIConnectionError
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from pydantic import BaseModel

from langfuse.client import Langfuse
from langfuse.openai import (
    AsyncAzureOpenAI,
    AsyncOpenAI,
    AzureOpenAI,
    _is_openai_v1,
    openai,
)
from tests.utils import create_uuid, encode_file_to_base64, get_api

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


def test_auth_check():
    auth_check = openai.langfuse_auth_check()

    assert auth_check is True


def test_openai_chat_completion():
    generation_name = create_uuid()
    completion = chat_func(
        name=generation_name,
        model="gpt-3.5-turbo",
        messages=[
            ChatCompletionMessage(
                role="assistant", content="You are an expert mathematician"
            ),
            {"role": "user", "content": "1 + 1 = "},
        ],
        temperature=0,
        metadata={"someKey": "someResponse"},
    )

    openai.flush_langfuse()

    generation = get_api().observations.get_many(
        name=generation_name, type="GENERATION"
    )

    assert len(generation.data) != 0
    assert generation.data[0].name == generation_name
    assert generation.data[0].metadata == {"someKey": "someResponse"}
    assert len(completion.choices) != 0
    assert generation.data[0].input == [
        {
            "content": "You are an expert mathematician",
            "audio": None,
            "function_call": None,
            "refusal": None,
            "role": "assistant",
            "tool_calls": None,
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
        "max_tokens": "inf",
        "presence_penalty": 0,
    }
    assert generation.data[0].usage.input is not None
    assert generation.data[0].usage.output is not None
    assert generation.data[0].usage.total is not None
    assert "2" in generation.data[0].output["content"]
    assert generation.data[0].output["role"] == "assistant"

    trace = get_api().trace.get(generation.data[0].trace_id)
    assert trace.input == [
        {
            "content": "You are an expert mathematician",
            "audio": None,
            "function_call": None,
            "refusal": None,
            "role": "assistant",
            "tool_calls": None,
        },
        {"role": "user", "content": "1 + 1 = "},
    ]
    assert trace.output["content"] == completion.choices[0].message.content
    assert trace.output["role"] == completion.choices[0].message.role


def test_openai_chat_completion_stream():
    generation_name = create_uuid()
    completion = chat_func(
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

    openai.flush_langfuse()

    generation = get_api().observations.get_many(
        name=generation_name, type="GENERATION"
    )

    assert len(generation.data) != 0
    assert generation.data[0].name == generation_name
    assert generation.data[0].metadata == {"someKey": "someResponse"}

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
        "max_tokens": "inf",
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

    trace = get_api().trace.get(generation.data[0].trace_id)
    assert trace.input == [{"role": "user", "content": "1 + 1 = "}]
    assert str(trace.output) == chat_content


def test_openai_chat_completion_stream_with_next_iteration():
    generation_name = create_uuid()
    completion = chat_func(
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

    openai.flush_langfuse()

    generation = get_api().observations.get_many(
        name=generation_name, type="GENERATION"
    )

    assert len(generation.data) != 0
    assert generation.data[0].name == generation_name
    assert generation.data[0].metadata == {"someKey": "someResponse"}

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
        "max_tokens": "inf",
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

    trace = get_api().trace.get(generation.data[0].trace_id)
    assert trace.input == [{"role": "user", "content": "1 + 1 = "}]
    assert str(trace.output) == chat_content


def test_openai_chat_completion_stream_fail():
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

    generation = get_api().observations.get_many(
        name=generation_name, type="GENERATION"
    )

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

    trace = get_api().trace.get(generation.data[0].trace_id)
    assert trace.input == [{"role": "user", "content": "1 + 1 = "}]
    assert trace.output is None


def test_openai_chat_completion_with_trace():
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

    generation = get_api().observations.get_many(
        name=generation_name, type="GENERATION"
    )

    assert len(generation.data) != 0
    assert generation.data[0].name == generation_name
    assert generation.data[0].trace_id == trace_id


def test_openai_chat_completion_with_langfuse_prompt():
    generation_name = create_uuid()
    langfuse = Langfuse()
    prompt_name = create_uuid()
    langfuse.create_prompt(name=prompt_name, prompt="test prompt", is_active=True)

    prompt_client = langfuse.get_prompt(name=prompt_name)

    chat_func(
        name=generation_name,
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Make me laugh"}],
        langfuse_prompt=prompt_client,
    )

    openai.flush_langfuse()

    generation = get_api().observations.get_many(
        name=generation_name, type="GENERATION"
    )

    assert len(generation.data) != 0
    assert generation.data[0].name == generation_name
    assert isinstance(generation.data[0].prompt_id, str)


def test_openai_chat_completion_with_parent_observation_id():
    generation_name = create_uuid()
    trace_id = create_uuid()
    span_id = create_uuid()
    langfuse = Langfuse()

    trace = langfuse.trace(id=trace_id)
    trace.span(id=span_id)

    chat_func(
        name=generation_name,
        model="gpt-3.5-turbo",
        trace_id=trace_id,
        parent_observation_id=span_id,
        messages=[{"role": "user", "content": "1 + 1 = "}],
        temperature=0,
        metadata={"someKey": "someResponse"},
    )

    openai.flush_langfuse()

    generation = get_api().observations.get_many(
        name=generation_name, type="GENERATION"
    )

    assert len(generation.data) != 0
    assert generation.data[0].name == generation_name
    assert generation.data[0].trace_id == trace_id
    assert generation.data[0].parent_observation_id == span_id


def test_openai_chat_completion_fail():
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

    generation = get_api().observations.get_many(
        name=generation_name, type="GENERATION"
    )

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


def test_openai_chat_completion_with_additional_params():
    user_id = create_uuid()
    session_id = create_uuid()
    tags = ["tag1", "tag2"]
    trace_id = create_uuid()
    completion = chat_func(
        name="user-creation",
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "1 + 1 = "}],
        temperature=0,
        metadata={"someKey": "someResponse"},
        user_id=user_id,
        trace_id=trace_id,
        session_id=session_id,
        tags=tags,
    )

    openai.flush_langfuse()

    assert len(completion.choices) != 0
    trace = get_api().trace.get(trace_id)

    assert trace.user_id == user_id
    assert trace.session_id == session_id
    assert trace.tags == tags


def test_openai_chat_completion_without_extra_param():
    completion = chat_func(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "1 + 1 = "}],
        temperature=0,
        metadata={"someKey": "someResponse"},
    )

    assert len(completion.choices) != 0


def test_openai_chat_completion_two_calls():
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


def test_openai_chat_completion_with_seed():
    generation_name = create_uuid()
    completion = chat_func(
        name=generation_name,
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "1 + 1 = "}],
        temperature=0,
        seed=123,
        metadata={"someKey": "someResponse"},
    )

    openai.flush_langfuse()

    generation = get_api().observations.get_many(
        name=generation_name, type="GENERATION"
    )

    assert generation.data[0].model_parameters == {
        "temperature": 0,
        "top_p": 1,
        "frequency_penalty": 0,
        "max_tokens": "inf",
        "presence_penalty": 0,
        "seed": 123,
    }
    assert len(completion.choices) != 0


def test_openai_completion():
    generation_name = create_uuid()
    completion = completion_func(
        name=generation_name,
        model="gpt-3.5-turbo-instruct",
        prompt="1 + 1 = ",
        temperature=0,
        metadata={"someKey": "someResponse"},
    )

    openai.flush_langfuse()

    generation = get_api().observations.get_many(
        name=generation_name, type="GENERATION"
    )

    assert len(generation.data) != 0
    assert generation.data[0].name == generation_name
    assert generation.data[0].metadata == {"someKey": "someResponse"}
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
        "max_tokens": "inf",
        "presence_penalty": 0,
    }
    assert generation.data[0].usage.input is not None
    assert generation.data[0].usage.output is not None
    assert generation.data[0].usage.total is not None
    assert generation.data[0].output == "2\n\n1 + 2 = 3\n\n2 + 3 = "

    trace = get_api().trace.get(generation.data[0].trace_id)
    assert trace.input == "1 + 1 = "
    assert trace.output == completion.choices[0].text


def test_openai_completion_stream():
    generation_name = create_uuid()
    completion = completion_func(
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

    openai.flush_langfuse()

    assert len(content) > 0

    generation = get_api().observations.get_many(
        name=generation_name, type="GENERATION"
    )

    assert len(generation.data) != 0
    assert generation.data[0].name == generation_name
    assert generation.data[0].metadata == {"someKey": "someResponse"}

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
        "max_tokens": "inf",
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

    trace = get_api().trace.get(generation.data[0].trace_id)
    assert trace.input == "1 + 1 = "
    assert trace.output == content


def test_openai_completion_fail():
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

    generation = get_api().observations.get_many(
        name=generation_name, type="GENERATION"
    )

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

    generation = get_api().observations.get_many(
        name=generation_name, type="GENERATION"
    )

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


def test_openai_completion_with_languse_prompt():
    generation_name = create_uuid()
    langfuse = Langfuse()
    prompt_name = create_uuid()
    prompt_client = langfuse.create_prompt(
        name=prompt_name, prompt="test prompt", is_active=True
    )
    completion_func(
        name=generation_name,
        model="gpt-3.5-turbo-instruct",
        prompt="1 + 1 = ",
        temperature=0,
        metadata={"someKey": "someResponse"},
        langfuse_prompt=prompt_client,
    )

    openai.flush_langfuse()

    generation = get_api().observations.get_many(
        name=generation_name, type="GENERATION"
    )

    assert len(generation.data) != 0
    assert generation.data[0].name == generation_name
    assert isinstance(generation.data[0].prompt_id, str)


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
    client = AsyncOpenAI()
    generation_name = create_uuid()

    completion = await client.chat.completions.create(
        messages=[{"role": "user", "content": "1 + 1 = "}],
        model="gpt-3.5-turbo",
        name=generation_name,
    )

    openai.flush_langfuse()

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
        "max_tokens": "inf",
        "presence_penalty": 0,
    }
    assert generation.data[0].usage.input is not None
    assert generation.data[0].usage.output is not None
    assert generation.data[0].usage.total is not None
    assert "2" in generation.data[0].output["content"]
    assert generation.data[0].output["role"] == "assistant"


@pytest.mark.asyncio
async def test_async_chat_stream():
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
        "max_tokens": "inf",
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
async def test_async_chat_stream_with_anext():
    client = AsyncOpenAI()

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

    openai.flush_langfuse()

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
        "max_tokens": "inf",
        "presence_penalty": 0,
    }
    assert generation.data[0].usage.input is not None
    assert generation.data[0].usage.output is not None
    assert generation.data[0].usage.total is not None

    # Completion start time for time-to-first-token
    assert generation.data[0].completion_start_time is not None
    assert generation.data[0].completion_start_time >= generation.data[0].start_time
    assert generation.data[0].completion_start_time <= generation.data[0].end_time


def test_openai_function_call():
    from typing import List

    from pydantic import BaseModel

    generation_name = create_uuid()

    class StepByStepAIResponse(BaseModel):
        title: str
        steps: List[str]

    import json

    response = openai.chat.completions.create(
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

    openai.flush_langfuse()

    generation = get_api().observations.get_many(
        name=generation_name, type="GENERATION"
    )

    assert len(generation.data) != 0
    assert generation.data[0].name == generation_name
    assert generation.data[0].output is not None
    assert "function_call" in generation.data[0].output

    assert output["title"] is not None


def test_openai_function_call_streamed():
    from typing import List

    from pydantic import BaseModel

    generation_name = create_uuid()

    class StepByStepAIResponse(BaseModel):
        title: str
        steps: List[str]

    response = openai.chat.completions.create(
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

    openai.flush_langfuse()

    generation = get_api().observations.get_many(
        name=generation_name, type="GENERATION"
    )

    assert len(generation.data) != 0
    assert generation.data[0].name == generation_name
    assert generation.data[0].output is not None
    assert "function_call" in generation.data[0].output


def test_openai_tool_call():
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
    openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        tools=tools,
        tool_choice="auto",
        name=generation_name,
    )

    openai.flush_langfuse()

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


def test_openai_tool_call_streamed():
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
    response = openai.chat.completions.create(
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

    openai.flush_langfuse()

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


def test_azure():
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

    generation = get_api().observations.get_many(
        name=generation_name, type="GENERATION"
    )

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

    generation = get_api().observations.get_many(
        name=generation_name, type="GENERATION"
    )

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


def test_openai_with_existing_trace_id():
    langfuse = Langfuse()
    trace = langfuse.trace(
        name="docs-retrieval",
        user_id="user__935d7d1d-8625-4ef4-8651-544613e7bd22",
        metadata={
            "email": "user@langfuse.com",
        },
        tags=["production"],
        output="This is a standard output",
        input="My custom input",
    )

    langfuse.flush()

    generation_name = create_uuid()
    completion = chat_func(
        name=generation_name,
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "1 + 1 = "}],
        temperature=0,
        metadata={"someKey": "someResponse"},
        trace_id=trace.id,
    )

    openai.flush_langfuse()

    generation = get_api().observations.get_many(
        name=generation_name, type="GENERATION"
    )

    assert len(generation.data) != 0
    assert generation.data[0].name == generation_name
    assert generation.data[0].metadata == {"someKey": "someResponse"}
    assert len(completion.choices) != 0
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
        "max_tokens": "inf",
        "presence_penalty": 0,
    }
    assert generation.data[0].usage.input is not None
    assert generation.data[0].usage.output is not None
    assert generation.data[0].usage.total is not None
    assert "2" in generation.data[0].output["content"]
    assert generation.data[0].output["role"] == "assistant"

    trace = get_api().trace.get(generation.data[0].trace_id)
    assert trace.output == "This is a standard output"
    assert trace.input == "My custom input"


def test_disabled_langfuse():
    # Reimport to reset the state
    from langfuse.openai import openai
    from langfuse.utils.langfuse_singleton import LangfuseSingleton

    LangfuseSingleton().reset()

    openai.langfuse_enabled = False

    generation_name = create_uuid()
    openai.chat.completions.create(
        name=generation_name,
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "1 + 1 = "}],
        temperature=0,
        metadata={"someKey": "someResponse"},
    )

    openai.flush_langfuse()

    generations = get_api().observations.get_many(
        name=generation_name, type="GENERATION"
    )

    assert len(generations.data) == 0

    # Reimport to reset the state
    LangfuseSingleton().reset()
    openai.langfuse_enabled = True

    import importlib

    from langfuse.openai import openai

    importlib.reload(openai)


def test_langchain_integration():
    from langchain_openai import ChatOpenAI

    chat = ChatOpenAI(model="gpt-4o")

    result = ""

    for chunk in chat.stream("Hello, how are you?"):
        result += chunk.content

    print(result)
    assert result != ""


def test_structured_output_response_format_kwarg():
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

    openai.chat.completions.create(
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

    openai.flush_langfuse()

    generation = get_api().observations.get_many(
        name=generation_name, type="GENERATION"
    )

    assert len(generation.data) != 0
    assert generation.data[0].name == generation_name
    assert generation.data[0].metadata == {
        "someKey": "someResponse",
        "response_format": {"type": "json_schema", "json_schema": json_schema},
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
        "max_tokens": "inf",
        "presence_penalty": 0,
    }
    assert generation.data[0].usage.input is not None
    assert generation.data[0].usage.output is not None
    assert generation.data[0].usage.total is not None
    assert generation.data[0].output["role"] == "assistant"

    trace = get_api().trace.get(generation.data[0].trace_id)
    assert trace.output is not None
    assert trace.input is not None


def test_structured_output_beta_completions_parse():
    from typing import List

    from packaging.version import Version

    class CalendarEvent(BaseModel):
        name: str
        date: str
        participants: List[str]

    generation_name = create_uuid()

    params = {
        "model": "gpt-4o-2024-08-06",
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

    openai.beta.chat.completions.parse(**params)

    openai.flush_langfuse()

    if Version(openai.__version__) >= Version("1.50.0"):
        # Check the trace and observation properties
        generation = get_api().observations.get_many(
            name=generation_name, type="GENERATION"
        )

        assert len(generation.data) == 1
        assert generation.data[0].name == generation_name
        assert generation.data[0].type == "GENERATION"
        assert generation.data[0].model == "gpt-4o-2024-08-06"
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

        # Check trace
        trace = get_api().trace.get(generation.data[0].trace_id)

        assert trace.input is not None
        assert trace.output is not None


@pytest.mark.asyncio
async def test_close_async_stream():
    client = AsyncOpenAI()
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

    openai.flush_langfuse()

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
        "max_tokens": "inf",
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


def test_base_64_image_input():
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

    openai.flush_langfuse()

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


def test_audio_input_and_output():
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

    openai.flush_langfuse()

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
