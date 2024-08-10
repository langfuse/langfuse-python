import os

import pytest
from pydantic import BaseModel
from openai import APIConnectionError

from langfuse.client import Langfuse
from langfuse.openai import (
    AsyncAzureOpenAI,
    AsyncOpenAI,
    AzureOpenAI,
    _is_openai_v1,
    _filter_image_data,
    openai,
)
from openai.types.chat.chat_completion import ChatCompletionMessage

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


def test_auth_check():
    auth_check = openai.langfuse_auth_check()

    assert auth_check is True


def test_openai_chat_completion():
    api = get_api()
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

    generation = api.observations.get_many(name=generation_name, type="GENERATION")

    assert len(generation.data) != 0
    assert generation.data[0].name == generation_name
    assert generation.data[0].metadata == {"someKey": "someResponse"}
    assert len(completion.choices) != 0
    assert generation.data[0].input == [
        {
            "content": "You are an expert mathematician",
            "function_call": None,
            "refusal": None,
            "role": "assistant",
            "tool_calls": None,
        },
        {"content": "1 + 1 = ", "role": "user"},
    ]
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

    trace = api.trace.get(generation.data[0].trace_id)
    assert trace.input == [
        {
            "content": "You are an expert mathematician",
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

    assert iter(completion)

    chat_content = ""
    for i in completion:
        chat_content += i.choices[0].delta.content or ""

    assert len(chat_content) > 0

    openai.flush_langfuse()

    generation = api.observations.get_many(name=generation_name, type="GENERATION")

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
    assert generation.data[0].output == "2"
    assert isinstance(generation.data[0].output, str) is True
    assert generation.data[0].completion_start_time is not None

    # Completion start time for time-to-first-token
    assert generation.data[0].completion_start_time is not None
    assert generation.data[0].completion_start_time >= generation.data[0].start_time
    assert generation.data[0].completion_start_time <= generation.data[0].end_time

    trace = api.trace.get(generation.data[0].trace_id)
    assert trace.input == [{"role": "user", "content": "1 + 1 = "}]
    assert trace.output == chat_content


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

    trace = api.trace.get(generation.data[0].trace_id)
    assert trace.input == [{"role": "user", "content": "1 + 1 = "}]
    assert trace.output is None


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


def test_openai_chat_completion_with_langfuse_prompt():
    api = get_api()
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

    generation = api.observations.get_many(name=generation_name, type="GENERATION")

    assert len(generation.data) != 0
    assert generation.data[0].name == generation_name
    assert isinstance(generation.data[0].prompt_id, str)


def test_openai_chat_completion_with_parent_observation_id():
    api = get_api()
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

    generation = api.observations.get_many(name=generation_name, type="GENERATION")

    assert len(generation.data) != 0
    assert generation.data[0].name == generation_name
    assert generation.data[0].trace_id == trace_id
    assert generation.data[0].parent_observation_id == span_id


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


def test_openai_chat_completion_with_additional_params():
    api = get_api()
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
    trace = api.trace.get(trace_id)

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

    trace = api.trace.get(generation.data[0].trace_id)
    assert trace.input == "1 + 1 = "
    assert trace.output == completion.choices[0].text


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

    assert iter(completion)
    content = ""
    for i in completion:
        content += i.choices[0].text

    openai.flush_langfuse()

    assert len(content) > 0

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

    # Completion start time for time-to-first-token
    assert generation.data[0].completion_start_time is not None
    assert generation.data[0].completion_start_time >= generation.data[0].start_time
    assert generation.data[0].completion_start_time <= generation.data[0].end_time

    trace = api.trace.get(generation.data[0].trace_id)
    assert trace.input == "1 + 1 = "
    assert trace.output == content


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


def test_openai_completion_with_languse_prompt():
    api = get_api()
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

    generation = api.observations.get_many(name=generation_name, type="GENERATION")

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
    api = get_api()
    client = AsyncOpenAI()

    generation_name = create_uuid()

    completion = await client.chat.completions.create(
        messages=[{"role": "user", "content": "1 + 1 = "}],
        model="gpt-3.5-turbo",
        name=generation_name,
        stream=True,
    )

    # Make sure that stream is consumed through iterator protocol: __anext__() and __aiter__()
    await anext(completion)
    async for c in completion:
        print(c)

    openai.flush_langfuse()
    print(completion)

    generation = api.observations.get_many(name=generation_name, type="GENERATION")

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
    assert "2" in generation.data[0].output

    # Completion start time for time-to-first-token
    assert generation.data[0].completion_start_time is not None
    assert generation.data[0].completion_start_time >= generation.data[0].start_time
    assert generation.data[0].completion_start_time <= generation.data[0].end_time


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


def test_openai_function_call_streamed():
    from typing import List

    from pydantic import BaseModel

    api = get_api()
    generation_name = create_uuid()

    class StepByStepAIResponse(BaseModel):
        title: str
        steps: List[str]

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
        stream=True,
    )

    # Consume the stream
    for _ in response:
        pass

    openai.flush_langfuse()

    generation = api.observations.get_many(name=generation_name, type="GENERATION")

    assert len(generation.data) != 0
    assert generation.data[0].name == generation_name
    assert generation.data[0].output is not None
    assert "function_call" in generation.data[0].output


def test_openai_tool_call():
    api = get_api()
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
    completion = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        tools=tools,
        tool_choice="auto",
        name=generation_name,
    )

    print(completion)

    openai.flush_langfuse()

    generation = api.observations.get_many(name=generation_name, type="GENERATION")

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
    api = get_api()
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

    generation = api.observations.get_many(name=generation_name, type="GENERATION")

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


def test_image_data_filtered_png():
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
                            "url": "data:image/png;base64,/9j/4AAQSkZJRgABAQEASABIAAD/2wBDAAMCAgMCAgMDAwMEAwMEBQgFBQQEBQoHBwYIDAoMDAsKCwsNDhIQDQ4RDgsLEBYQERMUFRUVDA8XGBYUGBIUFRT/wAALCAABAAEBAREA/8QAFAABAAAAAAAAAAAAAAAAAAAACf/EABQQAQAAAAAAAAAAAAAAAAAAAAD/2gAIAQEAAD8AKp//2Q=="
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
    result = _filter_image_data(messages)

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
    result = _filter_image_data(
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

    api = get_api()
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

    generation = api.observations.get_many(name=generation_name, type="GENERATION")

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

    trace = api.trace.get(generation.data[0].trace_id)
    assert trace.output == "This is a standard output"
    assert trace.input == "My custom input"


def test_disabled_langfuse():
    # Reimport to reset the state
    from langfuse.openai import openai
    from langfuse.utils.langfuse_singleton import LangfuseSingleton

    LangfuseSingleton().reset()

    openai.langfuse_enabled = False

    api = get_api()
    generation_name = create_uuid()
    openai.chat.completions.create(
        name=generation_name,
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "1 + 1 = "}],
        temperature=0,
        metadata={"someKey": "someResponse"},
    )

    openai.flush_langfuse()

    generations = api.observations.get_many(name=generation_name, type="GENERATION")

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
    api = get_api()
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

    generation = api.observations.get_many(name=generation_name, type="GENERATION")

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

    trace = api.trace.get(generation.data[0].trace_id)
    assert trace.output is not None
    assert trace.input is not None


def test_structured_output_beta_completions_parse():
    from typing import List

    class CalendarEvent(BaseModel):
        name: str
        date: str
        participants: List[str]

    openai.beta.chat.completions.parse(
        model="gpt-4o-2024-08-06",
        messages=[
            {"role": "system", "content": "Extract the event information."},
            {
                "role": "user",
                "content": "Alice and Bob are going to a science fair on Friday.",
            },
        ],
        response_format=CalendarEvent,
    )

    openai.flush_langfuse()
