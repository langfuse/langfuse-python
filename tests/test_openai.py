import os
import pytest
from langfuse.client import Langfuse
from langfuse.model import CreateTrace
from langfuse.openai import _is_openai_v1, _is_streaming_response, openai

from tests.utils import create_uuid, get_api


chat_func = openai.chat.completions.create if _is_openai_v1() else openai.ChatCompletion.create
completion_func = openai.completions.create if _is_openai_v1() else openai.Completion.create
expected_err = openai.APIError if _is_openai_v1() else openai.error.AuthenticationError
expected_err_msg = "Connection error." if _is_openai_v1() else "You didn't provide an API key."


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
    assert completion.choices[0].message.content == generation.data[0].output
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
    assert generation.data[0].prompt_tokens is not None
    assert generation.data[0].completion_tokens is not None
    assert generation.data[0].total_tokens is not None
    assert generation.data[0].output == "2"


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
    assert generation.data[0].prompt_tokens is not None
    assert generation.data[0].completion_tokens is not None
    assert generation.data[0].total_tokens is not None
    assert generation.data[0].output == "2"


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
    assert generation.data[0].prompt_tokens is not None
    assert generation.data[0].completion_tokens is not None
    assert generation.data[0].total_tokens is not None
    assert generation.data[0].level == "ERROR"
    assert expected_err_msg in generation.data[0].status_message

    openai.api_key = os.environ["OPENAI_API_KEY"]


def test_openai_chat_completion_with_trace():
    api = get_api()
    generation_name = create_uuid()
    trace_id = create_uuid()
    langfuse = Langfuse()

    langfuse.trace(CreateTrace(id=trace_id))

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
    assert completion.choices[0].message.content == generation.data[0].output
    assert generation.data[0].input == [{"content": "1 + 1 = ", "role": "user"}]

    generation_2 = api.observations.get_many(name=generation_name_2, type="GENERATION")

    assert len(generation_2.data) != 0
    assert generation_2.data[0].name == generation_name_2
    assert len(completion_2.choices) != 0
    assert completion_2.choices[0].message.content == generation_2.data[0].output
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
    assert generation.data[0].prompt_tokens is not None
    assert generation.data[0].completion_tokens is not None
    assert generation.data[0].total_tokens is not None


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
    assert generation.data[0].prompt_tokens is not None
    assert generation.data[0].completion_tokens is not None
    assert generation.data[0].total_tokens is not None


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
    assert generation.data[0].prompt_tokens is not None
    assert generation.data[0].completion_tokens is not None
    assert generation.data[0].total_tokens is not None
    assert generation.data[0].level == "ERROR"
    assert expected_err_msg in generation.data[0].status_message

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
