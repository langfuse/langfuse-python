import pytest
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from langfuse import get_client, observe, propagate_attributes
from langfuse.langchain import CallbackHandler
from tests.support.utils import wait_for_trace_snapshot

mock_metadata = {"key": "metadata"}
mock_deep_metadata = {"key": "mock_deep_metadata"}
mock_session_id = "session-id-1"
mock_args = (1, 2, 3)
mock_kwargs = {"a": 1, "b": 2, "c": 3}


def _get_observation_by_name(trace_data, name):
    return next(
        observation
        for observation in trace_data.observations
        if observation.name == name
    )


def _is_descendant(trace_data, child_id, ancestor_id):
    observations_by_id = {
        observation.id: observation for observation in trace_data.observations
    }
    current_id = child_id

    while current_id in observations_by_id:
        current = observations_by_id[current_id]
        parent_id = current.parent_observation_id
        if parent_id == ancestor_id:
            return True
        if parent_id not in observations_by_id:
            return False
        current_id = parent_id

    return False


def test_decorators_langchain():
    mock_name = "test_decorators_langchain"
    langfuse = get_client()
    mock_trace_id = langfuse.create_trace_id()

    @observe()
    def langchain_operations(*args, **kwargs):
        # Get langfuse callback handler for LangChain
        handler = CallbackHandler()
        prompt = ChatPromptTemplate.from_template("tell me a short joke about {topic}")
        model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

        chain = prompt | model

        return chain.invoke(
            {"topic": kwargs["topic"]},
            config={
                "callbacks": [handler],
            },
        )

    @observe()
    def level_3_function(*args, **kwargs):
        langfuse.update_current_span(metadata=mock_metadata)
        langfuse.update_current_span(metadata=mock_deep_metadata)
        return langchain_operations(*args, **kwargs)

    @observe()
    def level_2_function(*args, **kwargs):
        langfuse.update_current_span(metadata=mock_metadata)

        return level_3_function(*args, **kwargs)

    @observe()
    def level_1_function(*args, **kwargs):
        return level_2_function(*args, **kwargs)

    with propagate_attributes(session_id=mock_session_id, trace_name=mock_name):
        level_1_function(topic="socks", langfuse_trace_id=mock_trace_id)

    langfuse.flush()

    trace_data = wait_for_trace_snapshot(
        mock_trace_id,
        is_result_ready=lambda trace: (
            trace.session_id == mock_session_id
            and trace.name == mock_name
            and {
                "level_1_function",
                "level_2_function",
                "level_3_function",
                "langchain_operations",
                "ChatPromptTemplate",
            }.issubset({observation.name for observation in trace.observations})
        ),
    )
    assert len(trace_data.observations) > 2

    # trace parameters if set anywhere in the call stack
    assert trace_data.session_id == mock_session_id
    assert trace_data.name == mock_name

    level_1_observation = _get_observation_by_name(trace_data, "level_1_function")
    level_2_observation = _get_observation_by_name(trace_data, "level_2_function")
    level_3_observation = _get_observation_by_name(trace_data, "level_3_function")
    langchain_observation = _get_observation_by_name(trace_data, "langchain_operations")
    prompt_observation = _get_observation_by_name(trace_data, "ChatPromptTemplate")

    assert level_1_observation.name == "level_1_function"
    assert _is_descendant(trace_data, level_2_observation.id, level_1_observation.id)
    assert level_2_observation.name == "level_2_function"
    assert level_2_observation.metadata["key"] == mock_metadata["key"]
    assert _is_descendant(trace_data, level_3_observation.id, level_2_observation.id)
    assert level_3_observation.name == "level_3_function"
    assert level_3_observation.metadata["key"] == mock_deep_metadata["key"]
    assert _is_descendant(trace_data, langchain_observation.id, level_3_observation.id)
    assert langchain_observation.name == "langchain_operations"

    # Check that LangChain components are captured
    assert _is_descendant(trace_data, prompt_observation.id, langchain_observation.id)


@pytest.mark.asyncio
async def test_async_nested_openai_chat_stream():
    from langfuse.openai import AsyncOpenAI

    mock_name = "test_async_nested_openai_chat_stream"
    langfuse = get_client()
    mock_trace_id = langfuse.create_trace_id()
    mock_tags = ["tag1", "tag2"]
    mock_session_id = "session-id-1"
    mock_user_id = "user-id-1"

    @observe(capture_output=False)
    async def level_2_function():
        gen = await AsyncOpenAI().chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "1 + 1 = "}],
            temperature=0,
            metadata={"someKey": "someResponse"},
            stream=True,
        )

        with propagate_attributes(
            session_id=mock_session_id,
            user_id=mock_user_id,
            tags=mock_tags,
            trace_name=mock_name,
        ):
            async for c in gen:
                print(c)

            langfuse.update_current_span(metadata=mock_metadata)

        return "level_2"

    @observe()
    async def level_1_function(*args, **kwargs):
        await level_2_function()

        return "level_1"

    result = await level_1_function(
        *mock_args, **mock_kwargs, langfuse_trace_id=mock_trace_id
    )
    langfuse.flush()

    assert result == "level_1"  # Wrapped function returns correctly

    # ID setting for span or trace
    trace_data = wait_for_trace_snapshot(
        mock_trace_id,
        is_result_ready=lambda trace: (
            trace.session_id == mock_session_id
            and trace.name == mock_name
            and {
                "level_1_function",
                "level_2_function",
                "OpenAI-generation",
            }.issubset({observation.name for observation in trace.observations})
        ),
    )
    assert len(trace_data.observations) >= 3

    # trace parameters if set anywhere in the call stack
    assert trace_data.session_id == mock_session_id
    assert trace_data.name == mock_name

    level_1_observation = _get_observation_by_name(trace_data, "level_1_function")
    level_2_observation = _get_observation_by_name(trace_data, "level_2_function")
    level_3_observation = _get_observation_by_name(trace_data, "OpenAI-generation")
    assert _is_descendant(trace_data, level_2_observation.id, level_1_observation.id)
    assert _is_descendant(trace_data, level_3_observation.id, level_2_observation.id)

    assert level_2_observation.metadata["key"] == mock_metadata["key"]

    generation = level_3_observation

    assert generation.name == "OpenAI-generation"
    assert generation.metadata["someKey"] == "someResponse"
    assert generation.input == [{"content": "1 + 1 = ", "role": "user"}]
    assert generation.type == "GENERATION"
    assert "gpt-4o-mini" in generation.model
    assert generation.start_time is not None
    assert generation.end_time is not None
    assert generation.start_time < generation.end_time
    assert generation.model_parameters == {
        "service_tier": "default",
        "temperature": 0,
        "top_p": 1,
        "frequency_penalty": 0,
        "max_tokens": "Infinity",
        "presence_penalty": 0,
    }
    assert generation.usage_details["input"] is not None
    assert generation.usage_details["output"] is not None
    assert generation.usage_details["total"] is not None
    print(generation)
    assert "2" in generation.output
