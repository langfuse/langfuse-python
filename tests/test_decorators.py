import asyncio
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from time import sleep
from typing import Optional

import pytest
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from langfuse import Langfuse, get_client, observe
from langfuse._client.environment_variables import LANGFUSE_PUBLIC_KEY
from langfuse._client.resource_manager import LangfuseResourceManager
from langfuse.langchain import CallbackHandler
from langfuse.media import LangfuseMedia
from tests.utils import get_api

mock_metadata = {"key": "metadata"}
mock_deep_metadata = {"key": "mock_deep_metadata"}
mock_session_id = "session-id-1"
mock_args = (1, 2, 3)
mock_kwargs = {"a": 1, "b": 2, "c": 3}


def removeMockResourceManagerInstances():
    with LangfuseResourceManager._lock:
        for public_key in list(LangfuseResourceManager._instances.keys()):
            if public_key != os.getenv(LANGFUSE_PUBLIC_KEY):
                LangfuseResourceManager._instances.pop(public_key)


def test_nested_observations():
    mock_name = "test_nested_observations"
    langfuse = get_client()
    mock_trace_id = langfuse.create_trace_id()

    @observe(as_type="generation", name="level_3", capture_output=False)
    def level_3_function():
        langfuse.update_current_generation(metadata=mock_metadata)
        langfuse.update_current_generation(
            metadata=mock_deep_metadata,
            usage_details={"input": 150, "output": 50, "total": 300},
            model="gpt-3.5-turbo",
            output="mock_output",
        )
        langfuse.update_current_generation(version="version-1")
        langfuse.update_current_trace(session_id=mock_session_id, name=mock_name)

        langfuse.update_current_trace(
            user_id="user_id",
        )

        return "level_3"

    @observe(name="level_2_manually_set")
    def level_2_function():
        level_3_function()
        langfuse.update_current_span(metadata=mock_metadata)

        return "level_2"

    @observe()
    def level_1_function(*args, **kwargs):
        level_2_function()

        return "level_1"

    result = level_1_function(
        *mock_args, **mock_kwargs, langfuse_trace_id=mock_trace_id
    )
    langfuse.flush()

    assert result == "level_1"  # Wrapped function returns correctly

    # ID setting for span or trace
    trace_data = get_api().trace.get(mock_trace_id)
    assert len(trace_data.observations) == 3

    # trace parameters if set anywhere in the call stack
    assert trace_data.session_id == mock_session_id
    assert trace_data.user_id == "user_id"
    assert trace_data.name == mock_name

    # Check correct nesting
    adjacencies = defaultdict(list)
    for o in trace_data.observations:
        adjacencies[o.parent_observation_id].append(o)

    assert len(adjacencies) == 3

    level_1_observation = next(
        o
        for o in trace_data.observations
        if o.parent_observation_id not in [o.id for o in trace_data.observations]
    )
    level_2_observation = adjacencies[level_1_observation.id][0]
    level_3_observation = adjacencies[level_2_observation.id][0]

    assert level_1_observation.name == "level_1_function"
    assert level_1_observation.input == {"args": list(mock_args), "kwargs": mock_kwargs}
    assert level_1_observation.output == "level_1"

    assert level_2_observation.name == "level_2_manually_set"
    assert level_2_observation.metadata["key"] == mock_metadata["key"]

    assert level_3_observation.name == "level_3"
    assert level_3_observation.metadata["key"] == mock_deep_metadata["key"]
    assert level_3_observation.type == "GENERATION"
    assert level_3_observation.calculated_total_cost > 0
    assert level_3_observation.output == "mock_output"
    assert level_3_observation.version == "version-1"


def test_nested_observations_with_non_parentheses_decorator():
    mock_name = "test_nested_observations"
    langfuse = get_client()
    mock_trace_id = langfuse.create_trace_id()

    @observe(as_type="generation", name="level_3", capture_output=False)
    def level_3_function():
        langfuse.update_current_generation(metadata=mock_metadata)
        langfuse.update_current_generation(
            metadata=mock_deep_metadata,
            usage_details={"input": 150, "output": 50, "total": 300},
            model="gpt-3.5-turbo",
            output="mock_output",
        )
        langfuse.update_current_generation(version="version-1")

        langfuse.update_current_trace(session_id=mock_session_id, name=mock_name)

        langfuse.update_current_trace(
            user_id="user_id",
        )

        return "level_3"

    @observe
    def level_2_function():
        level_3_function()
        langfuse.update_current_span(metadata=mock_metadata)

        return "level_2"

    @observe
    def level_1_function(*args, **kwargs):
        level_2_function()

        return "level_1"

    result = level_1_function(
        *mock_args, **mock_kwargs, langfuse_trace_id=mock_trace_id
    )
    langfuse.flush()

    assert result == "level_1"  # Wrapped function returns correctly

    # ID setting for span or trace
    trace_data = get_api().trace.get(mock_trace_id)
    assert len(trace_data.observations) == 3

    # trace parameters if set anywhere in the call stack
    assert trace_data.session_id == mock_session_id
    assert trace_data.user_id == "user_id"
    assert trace_data.name == mock_name

    # Check correct nesting
    adjacencies = defaultdict(list)
    for o in trace_data.observations:
        adjacencies[o.parent_observation_id or o.trace_id].append(o)

    assert len(adjacencies) == 3

    level_1_observation = next(
        o
        for o in trace_data.observations
        if o.parent_observation_id not in [o.id for o in trace_data.observations]
    )
    level_2_observation = adjacencies[level_1_observation.id][0]
    level_3_observation = adjacencies[level_2_observation.id][0]

    assert level_1_observation.name == "level_1_function"
    assert level_1_observation.input == {"args": list(mock_args), "kwargs": mock_kwargs}
    assert level_1_observation.output == "level_1"

    assert level_2_observation.name == "level_2_function"
    assert level_2_observation.metadata["key"] == mock_metadata["key"]

    assert level_3_observation.name == "level_3"
    assert level_3_observation.metadata["key"] == mock_deep_metadata["key"]
    assert level_3_observation.type == "GENERATION"
    assert level_3_observation.calculated_total_cost > 0
    assert level_3_observation.output == "mock_output"
    assert level_3_observation.version == "version-1"


# behavior on exceptions
def test_exception_in_wrapped_function():
    mock_name = "test_exception_in_wrapped_function"
    langfuse = get_client()
    mock_trace_id = langfuse.create_trace_id()

    @observe(as_type="generation", capture_output=False)
    def level_3_function():
        langfuse.update_current_generation(metadata=mock_metadata)
        langfuse.update_current_generation(
            metadata=mock_deep_metadata,
            usage_details={"input": 150, "output": 50, "total": 300},
            model="gpt-3.5-turbo",
        )
        langfuse.update_current_trace(session_id=mock_session_id, name=mock_name)

        raise ValueError("Mock exception")

    @observe()
    def level_2_function():
        level_3_function()
        langfuse.update_current_generation(metadata=mock_metadata)

        return "level_2"

    @observe()
    def level_1_function(*args, **kwargs):
        sleep(1)
        level_2_function()
        print("hello")

        return "level_1"

    # Check that the exception is raised
    with pytest.raises(ValueError):
        level_1_function(*mock_args, **mock_kwargs, langfuse_trace_id=mock_trace_id)

    langfuse.flush()

    trace_data = get_api().trace.get(mock_trace_id)

    # trace parameters if set anywhere in the call stack
    assert trace_data.session_id == mock_session_id
    assert trace_data.name == mock_name

    adjacencies = defaultdict(list)
    for o in trace_data.observations:
        adjacencies[o.parent_observation_id or o.trace_id].append(o)

    assert len(adjacencies) == 3

    level_1_observation = next(
        o
        for o in trace_data.observations
        if o.parent_observation_id not in [o.id for o in trace_data.observations]
    )
    level_2_observation = adjacencies[level_1_observation.id][0]
    level_3_observation = adjacencies[level_2_observation.id][0]

    assert level_1_observation.name == "level_1_function"
    assert level_1_observation.input == {"args": list(mock_args), "kwargs": mock_kwargs}

    assert level_2_observation.name == "level_2_function"

    assert level_3_observation.name == "level_3_function"
    assert level_3_observation.type == "GENERATION"

    assert level_3_observation.status_message == "Mock exception"
    assert level_3_observation.level == "ERROR"


# behavior on concurrency
def test_concurrent_decorator_executions():
    mock_name = "test_concurrent_decorator_executions"
    langfuse = get_client()
    mock_trace_id_1 = langfuse.create_trace_id()
    mock_trace_id_2 = langfuse.create_trace_id()

    @observe(as_type="generation", capture_output=False)
    def level_3_function():
        langfuse.update_current_generation(metadata=mock_metadata)
        langfuse.update_current_generation(metadata=mock_deep_metadata)
        langfuse.update_current_generation(
            metadata=mock_deep_metadata,
            usage_details={"input": 150, "output": 50, "total": 300},
            model="gpt-3.5-turbo",
        )
        langfuse.update_current_trace(name=mock_name, session_id=mock_session_id)

        return "level_3"

    @observe()
    def level_2_function():
        level_3_function()
        langfuse.update_current_generation(metadata=mock_metadata)

        return "level_2"

    @observe(name=mock_name)
    def level_1_function(*args, **kwargs):
        sleep(1)
        level_2_function()

        return "level_1"

    with ThreadPoolExecutor(max_workers=2) as executor:
        future1 = executor.submit(
            level_1_function,
            *mock_args,
            mock_trace_id_1,
            **mock_kwargs,
            langfuse_trace_id=mock_trace_id_1,
        )
        future2 = executor.submit(
            level_1_function,
            *mock_args,
            mock_trace_id_2,
            **mock_kwargs,
            langfuse_trace_id=mock_trace_id_2,
        )

        future1.result()
        future2.result()

    langfuse.flush()

    for mock_id in [mock_trace_id_1, mock_trace_id_2]:
        trace_data = get_api().trace.get(mock_id)
        assert len(trace_data.observations) == 3

        # ID setting for span or trace
        assert trace_data.session_id == mock_session_id
        assert trace_data.name == mock_name

        # Check correct nesting
        adjacencies = defaultdict(list)
        for o in trace_data.observations:
            adjacencies[o.parent_observation_id].append(o)

        assert len(adjacencies) == 3

        level_1_observation = next(
            o
            for o in trace_data.observations
            if o.parent_observation_id not in [o.id for o in trace_data.observations]
        )
        level_2_observation = adjacencies[level_1_observation.id][0]
        level_3_observation = adjacencies[level_2_observation.id][0]

        assert level_1_observation.name == mock_name
        assert level_1_observation.input == {
            "args": list(mock_args) + [mock_id],
            "kwargs": mock_kwargs,
        }
        assert level_1_observation.output == "level_1"

        assert level_2_observation.metadata["key"] == mock_metadata["key"]

        assert level_3_observation.metadata["key"] == mock_deep_metadata["key"]
        assert level_3_observation.type == "GENERATION"
        assert level_3_observation.calculated_total_cost > 0


def test_decorators_langchain():
    mock_name = "test_decorators_langchain"
    langfuse = get_client()
    mock_trace_id = langfuse.create_trace_id()

    @observe()
    def langchain_operations(*args, **kwargs):
        # Get langfuse callback handler for LangChain
        handler = CallbackHandler()
        prompt = ChatPromptTemplate.from_template("tell me a short joke about {topic}")
        model = ChatOpenAI(temperature=0)

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
        langfuse.update_current_trace(session_id=mock_session_id, name=mock_name)

        return langchain_operations(*args, **kwargs)

    @observe()
    def level_2_function(*args, **kwargs):
        langfuse.update_current_span(metadata=mock_metadata)

        return level_3_function(*args, **kwargs)

    @observe()
    def level_1_function(*args, **kwargs):
        return level_2_function(*args, **kwargs)

    level_1_function(topic="socks", langfuse_trace_id=mock_trace_id)

    langfuse.flush()

    trace_data = get_api().trace.get(mock_trace_id)
    assert len(trace_data.observations) > 2

    # Check correct nesting
    adjacencies = defaultdict(list)
    for o in trace_data.observations:
        adjacencies[o.parent_observation_id].append(o)

    assert len(adjacencies) > 2

    # trace parameters if set anywhere in the call stack
    assert trace_data.session_id == mock_session_id
    assert trace_data.name == mock_name

    # Check that the langchain_operations is at the correct level
    level_1_observation = next(
        o
        for o in trace_data.observations
        if o.parent_observation_id not in [o.id for o in trace_data.observations]
    )
    level_2_observation = adjacencies[level_1_observation.id][0]
    level_3_observation = adjacencies[level_2_observation.id][0]
    langchain_observation = adjacencies[level_3_observation.id][0]

    assert level_1_observation.name == "level_1_function"
    assert level_2_observation.name == "level_2_function"
    assert level_2_observation.metadata["key"] == mock_metadata["key"]
    assert level_3_observation.name == "level_3_function"
    assert level_3_observation.metadata["key"] == mock_deep_metadata["key"]
    assert langchain_observation.name == "langchain_operations"

    # Check that LangChain components are captured
    assert any([o.name == "ChatPromptTemplate" for o in trace_data.observations])


def test_get_current_trace_url():
    langfuse = get_client()
    mock_trace_id = langfuse.create_trace_id()

    @observe()
    def level_3_function():
        return langfuse.get_trace_url(trace_id=langfuse.get_current_trace_id())

    @observe()
    def level_2_function():
        return level_3_function()

    @observe()
    def level_1_function(*args, **kwargs):
        return level_2_function()

    result = level_1_function(
        *mock_args, **mock_kwargs, langfuse_trace_id=mock_trace_id
    )
    langfuse.flush()

    expected_url = f"http://localhost:3000/project/7a88fb47-b4e2-43b8-a06c-a5ce950dc53a/traces/{mock_trace_id}"
    assert result == expected_url


def test_scoring_observations():
    mock_name = "test_scoring_observations"
    langfuse = get_client()
    mock_trace_id = langfuse.create_trace_id()

    @observe(as_type="generation", capture_output=False)
    def level_3_function():
        langfuse.score_current_span(name="test-observation-score", value=1)
        langfuse.score_current_trace(name="another-test-trace-score", value="my_value")

        return "level_3"

    @observe()
    def level_2_function():
        return level_3_function()

    @observe()
    def level_1_function(*args, **kwargs):
        langfuse.score_current_trace(name="test-trace-score", value=3)
        langfuse.update_current_trace(name=mock_name)
        return level_2_function()

    result = level_1_function(
        *mock_args, **mock_kwargs, langfuse_trace_id=mock_trace_id
    )
    langfuse.flush()
    sleep(1)

    assert result == "level_3"  # Wrapped function returns correctly

    # ID setting for span or trace
    trace_data = get_api().trace.get(mock_trace_id)
    assert (
        len(trace_data.observations) == 3
    )  # Top-most function is trace, so it's not an observations
    assert trace_data.name == mock_name

    # Check for correct scoring
    scores = trace_data.scores

    assert len(scores) == 3

    trace_scores = [
        s for s in scores if s.trace_id == mock_trace_id and s.observation_id is None
    ]
    observation_score = [s for s in scores if s.observation_id is not None][0]

    assert any(
        [
            score.name == "another-test-trace-score"
            and score.string_value == "my_value"
            and score.data_type == "CATEGORICAL"
            for score in trace_scores
        ]
    )
    assert any(
        [
            score.name == "test-trace-score"
            and score.value == 3
            and score.data_type == "NUMERIC"
            for score in trace_scores
        ]
    )

    assert observation_score.name == "test-observation-score"
    assert observation_score.value == 1
    assert observation_score.data_type == "NUMERIC"


def test_circular_reference_handling():
    langfuse = get_client()
    mock_trace_id = langfuse.create_trace_id()

    # Define a class that will contain a circular reference
    class CircularRefObject:
        def __init__(self):
            self.reference: Optional[CircularRefObject] = None

    @observe()
    def function_with_circular_arg(circular_obj, *args, **kwargs):
        # This function doesn't need to do anything with circular_obj,
        # the test is simply to see if it can be called without error.
        return "function response"

    # Create an instance of the object and establish a circular reference
    circular_obj = CircularRefObject()
    circular_obj.reference = circular_obj

    # Call the decorated function, passing the circularly-referenced object
    result = function_with_circular_arg(circular_obj, langfuse_trace_id=mock_trace_id)

    langfuse.flush()

    # Validate that the function executed as expected
    assert result == "function response"

    trace_data = get_api().trace.get(mock_trace_id)

    assert (
        trace_data.observations[0].input["args"][0]["reference"] == "CircularRefObject"
    )


def test_disabled_io_capture():
    langfuse = get_client()
    mock_trace_id = langfuse.create_trace_id()

    class Node:
        def __init__(self, value: tuple):
            self.value = value

    @observe(capture_input=False, capture_output=False)
    def nested(*args, **kwargs):
        langfuse.update_current_span(
            input=Node(("manually set tuple", 1)), output="manually set output"
        )
        return "nested response"

    @observe(capture_output=False)
    def main(*args, **kwargs):
        nested(*args, **kwargs)
        return "function response"

    result = main("Hello, World!", name="John", langfuse_trace_id=mock_trace_id)

    langfuse.flush()

    assert result == "function response"

    trace_data = get_api().trace.get(mock_trace_id)

    # Check that disabled capture_io doesn't capture manually set input/output
    assert len(trace_data.observations) == 2
    # Only one of the observations must satisfy this
    found_match = False
    for observation in trace_data.observations:
        if (
            observation.input
            and isinstance(observation.input, dict)
            and "value" in observation.input
            and observation.input["value"] == ["manually set tuple", 1]
            and observation.output == "manually set output"
        ):
            found_match = True
            break
    assert found_match, "No observation found with expected input and output"


def test_decorated_class_and_instance_methods():
    mock_name = "test_decorated_class_and_instance_methods"
    langfuse = get_client()
    mock_trace_id = langfuse.create_trace_id()

    class TestClass:
        @classmethod
        @observe(name="class-method")
        def class_method(cls, *args, **kwargs):
            langfuse.update_current_span()
            return "class_method"

        @observe(as_type="generation", capture_output=False)
        def level_3_function(self):
            langfuse.update_current_generation(metadata=mock_metadata)
            langfuse.update_current_generation(
                metadata=mock_deep_metadata,
                usage_details={"input": 150, "output": 50, "total": 300},
                model="gpt-3.5-turbo",
                output="mock_output",
            )

            langfuse.update_current_trace(session_id=mock_session_id, name=mock_name)

            return "level_3"

        @observe()
        def level_2_function(self):
            TestClass.class_method()

            self.level_3_function()
            langfuse.update_current_span(metadata=mock_metadata)

            return "level_2"

        @observe()
        def level_1_function(self, *args, **kwargs):
            self.level_2_function()

            return "level_1"

    result = TestClass().level_1_function(
        *mock_args, **mock_kwargs, langfuse_trace_id=mock_trace_id
    )

    langfuse.flush()

    assert result == "level_1"  # Wrapped function returns correctly

    # ID setting for span or trace
    trace_data = get_api().trace.get(mock_trace_id)
    assert len(trace_data.observations) == 4

    # trace parameters if set anywhere in the call stack
    assert trace_data.session_id == mock_session_id
    assert trace_data.name == mock_name

    # Check correct nesting
    adjacencies = defaultdict(list)
    for o in trace_data.observations:
        adjacencies[o.parent_observation_id].append(o)

    assert len(adjacencies) == 3

    level_1_observation = next(
        o
        for o in trace_data.observations
        if o.parent_observation_id not in [o.id for o in trace_data.observations]
    )
    level_2_observation = adjacencies[level_1_observation.id][0]

    # Find level_3_observation and class_method_observation in level_2's children
    level_2_children = adjacencies[level_2_observation.id]
    level_3_observation = next(o for o in level_2_children if o.name != "class-method")
    class_method_observation = next(
        o for o in level_2_children if o.name == "class-method"
    )

    assert level_1_observation.name == "level_1_function"
    assert level_1_observation.input == {"args": list(mock_args), "kwargs": mock_kwargs}
    assert level_1_observation.output == "level_1"

    assert level_2_observation.name == "level_2_function"
    assert level_2_observation.metadata["key"] == mock_metadata["key"]

    assert class_method_observation.name == "class-method"
    assert class_method_observation.output == "class_method"

    assert level_3_observation.name == "level_3_function"
    assert level_3_observation.metadata["key"] == mock_deep_metadata["key"]
    assert level_3_observation.type == "GENERATION"
    assert level_3_observation.calculated_total_cost > 0
    assert level_3_observation.output == "mock_output"


def test_generator_as_return_value():
    langfuse = get_client()
    mock_trace_id = langfuse.create_trace_id()
    mock_output = "Hello, World!"

    def custom_transform_to_string(x):
        return "--".join(x)

    def generator_function():
        yield "Hello"
        yield ", "
        yield "World!"

    @observe(transform_to_string=custom_transform_to_string)
    def nested():
        return generator_function()

    @observe()
    def main(**kwargs):
        gen = nested()

        result = ""
        for item in gen:
            result += item

        return result

    result = main(langfuse_trace_id=mock_trace_id)
    langfuse.flush()

    assert result == mock_output

    trace_data = get_api().trace.get(mock_trace_id)

    # Find the main and nested observations
    adjacencies = defaultdict(list)
    for o in trace_data.observations:
        adjacencies[o.parent_observation_id].append(o)

    main_observation = next(
        o
        for o in trace_data.observations
        if o.parent_observation_id not in [o.id for o in trace_data.observations]
    )
    nested_observation = adjacencies[main_observation.id][0]

    assert main_observation.name == "main"
    assert main_observation.output == mock_output

    assert nested_observation.name == "nested"
    assert nested_observation.output == "Hello--, --World!"


@pytest.mark.asyncio
async def test_async_generator_as_return_value():
    langfuse = get_client()
    mock_trace_id = langfuse.create_trace_id()
    mock_output = "Hello, async World!"

    def custom_transform_to_string(x):
        return "--".join(x)

    @observe(transform_to_string=custom_transform_to_string)
    async def async_generator_function():
        await asyncio.sleep(0.1)  # Simulate async operation
        yield "Hello"
        await asyncio.sleep(0.1)
        yield ", async "
        await asyncio.sleep(0.1)
        yield "World!"

    @observe()
    async def main_async(**kwargs):
        gen = async_generator_function()

        result = ""
        async for item in gen:
            result += item

        return result

    result = await main_async(langfuse_trace_id=mock_trace_id)
    langfuse.flush()

    assert result == mock_output

    trace_data = get_api().trace.get(mock_trace_id)

    # Check correct nesting
    adjacencies = defaultdict(list)
    for o in trace_data.observations:
        adjacencies[o.parent_observation_id].append(o)

    main_observation = next(
        o
        for o in trace_data.observations
        if o.parent_observation_id not in [o.id for o in trace_data.observations]
    )
    nested_observation = adjacencies[main_observation.id][0]

    assert main_observation.name == "main_async"
    assert main_observation.output == mock_output

    assert nested_observation.name == "async_generator_function"
    assert nested_observation.output == "Hello--, async --World!"


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
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "1 + 1 = "}],
            temperature=0,
            metadata={"someKey": "someResponse"},
            stream=True,
        )

        langfuse.update_current_trace(
            session_id=mock_session_id,
            user_id=mock_user_id,
            tags=mock_tags,
        )

        async for c in gen:
            print(c)

        langfuse.update_current_span(metadata=mock_metadata)
        langfuse.update_current_trace(name=mock_name)

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
    trace_data = get_api().trace.get(mock_trace_id)
    assert len(trace_data.observations) == 3

    # trace parameters if set anywhere in the call stack
    assert trace_data.session_id == mock_session_id
    assert trace_data.name == mock_name

    # Check correct nesting
    adjacencies = defaultdict(list)
    for o in trace_data.observations:
        adjacencies[o.parent_observation_id or o.trace_id].append(o)

    assert len(adjacencies) == 3

    level_1_observation = next(
        o
        for o in trace_data.observations
        if o.parent_observation_id not in [o.id for o in trace_data.observations]
    )
    level_2_observation = adjacencies[level_1_observation.id][0]
    level_3_observation = adjacencies[level_2_observation.id][0]

    assert level_2_observation.metadata["key"] == mock_metadata["key"]

    generation = level_3_observation

    assert generation.name == "OpenAI-generation"
    assert generation.metadata["someKey"] == "someResponse"
    assert generation.input == [{"content": "1 + 1 = ", "role": "user"}]
    assert generation.type == "GENERATION"
    assert "gpt-3.5-turbo" in generation.model
    assert generation.start_time is not None
    assert generation.end_time is not None
    assert generation.start_time < generation.end_time
    assert generation.model_parameters == {
        "temperature": 0,
        "top_p": 1,
        "frequency_penalty": 0,
        "max_tokens": "Infinity",
        "presence_penalty": 0,
    }
    assert generation.usage.input is not None
    assert generation.usage.output is not None
    assert generation.usage.total is not None
    print(generation)
    assert generation.output == "2"


def test_generator_as_function_input():
    langfuse = get_client()
    mock_trace_id = langfuse.create_trace_id()
    mock_output = "Hello, World!"

    def generator_function():
        yield "Hello"
        yield ", "
        yield "World!"

    @observe()
    def nested(gen):
        result = ""
        for item in gen:
            result += item

        return result

    @observe()
    def main(**kwargs):
        gen = generator_function()

        return nested(gen)

    result = main(langfuse_trace_id=mock_trace_id)
    langfuse.flush()

    assert result == mock_output

    trace_data = get_api().trace.get(mock_trace_id)

    nested_obs = next(o for o in trace_data.observations if o.name == "nested")

    assert nested_obs.input["args"][0] == "<generator>"
    assert nested_obs.output == "Hello, World!"

    observation_start_time = nested_obs.start_time
    observation_end_time = nested_obs.end_time

    assert observation_start_time is not None
    assert observation_end_time is not None
    assert observation_start_time <= observation_end_time


def test_nest_list_of_generator_as_function_IO():
    langfuse = get_client()
    mock_trace_id = langfuse.create_trace_id()

    def generator_function():
        yield "Hello"
        yield ", "
        yield "World!"

    @observe()
    def nested(list_of_gens):
        return list_of_gens

    @observe()
    def main(**kwargs):
        gen = generator_function()

        return nested([(gen, gen)])

    main(langfuse_trace_id=mock_trace_id)
    langfuse.flush()

    trace_data = get_api().trace.get(mock_trace_id)

    # Find the observation with name 'nested'
    nested_observation = next(o for o in trace_data.observations if o.name == "nested")

    assert [[["<generator>", "<generator>"]]] == nested_observation.input["args"]

    assert all(
        ["generator" in arg for arg in nested_observation.output[0]],
    )

    observation_start_time = nested_observation.start_time
    observation_end_time = nested_observation.end_time

    assert observation_start_time is not None
    assert observation_end_time is not None
    assert observation_start_time <= observation_end_time


def test_return_dict_for_output():
    langfuse = get_client()
    mock_trace_id = langfuse.create_trace_id()
    mock_output = {"key": "value"}

    @observe()
    def function():
        return mock_output

    result = function(langfuse_trace_id=mock_trace_id)
    langfuse.flush()

    assert result == mock_output

    trace_data = get_api().trace.get(mock_trace_id)
    assert trace_data.observations[0].output == mock_output


def test_media():
    langfuse = get_client()
    mock_trace_id = langfuse.create_trace_id()

    with open("static/bitcoin.pdf", "rb") as pdf_file:
        pdf_bytes = pdf_file.read()

    media = LangfuseMedia(content_bytes=pdf_bytes, content_type="application/pdf")

    @observe()
    def main():
        sleep(1)
        langfuse.update_current_trace(
            input={
                "context": {
                    "nested": media,
                },
            },
            output={
                "context": {
                    "nested": media,
                },
            },
            metadata={
                "context": {
                    "nested": media,
                },
            },
        )

    main(langfuse_trace_id=mock_trace_id)

    langfuse.flush()

    trace_data = get_api().trace.get(mock_trace_id)

    assert (
        "@@@langfuseMedia:type=application/pdf|id="
        in trace_data.input["context"]["nested"]
    )
    assert (
        "@@@langfuseMedia:type=application/pdf|id="
        in trace_data.output["context"]["nested"]
    )
    assert (
        "@@@langfuseMedia:type=application/pdf|id="
        in trace_data.metadata["context"]["nested"]
    )
    parsed_reference_string = LangfuseMedia.parse_reference_string(
        trace_data.metadata["context"]["nested"]
    )
    assert parsed_reference_string["content_type"] == "application/pdf"
    assert parsed_reference_string["media_id"] is not None
    assert parsed_reference_string["source"] == "bytes"


def test_merge_metadata_and_tags():
    langfuse = get_client()
    mock_trace_id = langfuse.create_trace_id()

    @observe
    def nested():
        langfuse.update_current_trace(metadata={"key2": "value2"}, tags=["tag2"])

    @observe
    def main():
        langfuse.update_current_trace(metadata={"key1": "value1"}, tags=["tag1"])

        nested()

    main(langfuse_trace_id=mock_trace_id)

    langfuse.flush()

    trace_data = get_api().trace.get(mock_trace_id)

    assert trace_data.metadata["key1"] == "value1"
    assert trace_data.metadata["key2"] == "value2"

    assert trace_data.tags == ["tag1", "tag2"]


# Multi-project context propagation tests
@pytest.mark.skip(
    reason="Somehow adding another client is polluting the global context in LangfuseResourceManager that makes other test suites fail."
)
def test_multiproject_context_propagation_basic():
    """Test that nested decorated functions inherit langfuse_public_key from parent in multi-project setup"""
    client1 = Langfuse()  # Reads from environment
    Langfuse(public_key="pk-test-project2", secret_key="sk-test-project2")

    # Verify both instances are registered
    assert len(LangfuseResourceManager._instances) == 2

    mock_name = "test_multiproject_context_propagation_basic"
    # Use known public key from environment
    env_public_key = os.environ[LANGFUSE_PUBLIC_KEY]
    # In multi-project setup, must specify which client to use
    langfuse = get_client(public_key=env_public_key)
    mock_trace_id = langfuse.create_trace_id()

    @observe(as_type="generation", capture_output=False)
    def level_3_function():
        # This function should inherit the public key from level_1_function
        # and NOT need langfuse_public_key parameter
        langfuse_client = get_client()
        langfuse_client.update_current_generation(metadata={"level": "3"})
        langfuse_client.update_current_trace(name=mock_name)
        return "level_3"

    @observe()
    def level_2_function():
        # This function should also inherit the public key
        level_3_function()
        langfuse_client = get_client()
        langfuse_client.update_current_span(metadata={"level": "2"})
        return "level_2"

    @observe()
    def level_1_function(*args, **kwargs):
        # Only this top-level function receives langfuse_public_key
        level_2_function()
        langfuse_client = get_client()
        langfuse_client.update_current_span(metadata={"level": "1"})
        return "level_1"

    result = level_1_function(
        *mock_args,
        **mock_kwargs,
        langfuse_trace_id=mock_trace_id,
        langfuse_public_key=env_public_key,  # Only provided to top-level function
    )

    # Use the correct client for flushing
    client1.flush()

    assert result == "level_1"

    # Verify trace was created properly
    trace_data = get_api().trace.get(mock_trace_id)
    assert len(trace_data.observations) == 3
    assert trace_data.name == mock_name

    # Reset instances to not leak to other test suites
    removeMockResourceManagerInstances()


@pytest.mark.skip(
    reason="Somehow adding another client is polluting the global context in LangfuseResourceManager that makes other test suites fail."
)
def test_multiproject_context_propagation_deep_nesting():
    client1 = Langfuse()  # Reads from environment
    Langfuse(public_key="pk-test-project2", secret_key="sk-test-project2")

    # Verify both instances are registered
    assert len(LangfuseResourceManager._instances) == 2

    mock_name = "test_multiproject_context_propagation_deep_nesting"
    env_public_key = os.environ[LANGFUSE_PUBLIC_KEY]
    langfuse = get_client(public_key=env_public_key)
    mock_trace_id = langfuse.create_trace_id()

    @observe(as_type="generation")
    def level_4_function():
        langfuse_client = get_client()
        langfuse_client.update_current_generation(metadata={"level": "4"})
        return "level_4"

    @observe()
    def level_3_function():
        result = level_4_function()
        langfuse_client = get_client()
        langfuse_client.update_current_span(metadata={"level": "3"})
        return result

    @observe()
    def level_2_function():
        result = level_3_function()
        langfuse_client = get_client()
        langfuse_client.update_current_span(metadata={"level": "2"})
        return result

    @observe()
    def level_1_function(*args, **kwargs):
        langfuse_client = get_client()
        langfuse_client.update_current_trace(name=mock_name)
        result = level_2_function()
        langfuse_client.update_current_span(metadata={"level": "1"})
        return result

    result = level_1_function(
        langfuse_trace_id=mock_trace_id, langfuse_public_key=env_public_key
    )
    client1.flush()

    assert result == "level_4"

    trace_data = get_api().trace.get(mock_trace_id)
    assert len(trace_data.observations) == 4
    assert trace_data.name == mock_name

    # Verify all levels were captured
    levels = [
        str(obs.metadata.get("level"))
        for obs in trace_data.observations
        if obs.metadata
    ]
    assert set(levels) == {"1", "2", "3", "4"}

    # Reset instances to not leak to other test suites
    removeMockResourceManagerInstances()


@pytest.mark.skip(
    reason="Somehow adding another client is polluting the global context in LangfuseResourceManager that makes other test suites fail."
)
def test_multiproject_context_propagation_override():
    # Initialize two separate Langfuse instances
    client1 = Langfuse()  # Reads from environment
    client2 = Langfuse(public_key="pk-test-project2", secret_key="sk-test-project2")

    # Verify both instances are registered
    assert len(LangfuseResourceManager._instances) == 2

    mock_name = "test_multiproject_context_propagation_override"
    env_public_key = os.environ[LANGFUSE_PUBLIC_KEY]
    langfuse = get_client(public_key=env_public_key)
    mock_trace_id = langfuse.create_trace_id()

    primary_public_key = env_public_key
    override_public_key = "pk-test-project2"

    @observe(as_type="generation")
    def level_3_function():
        # This function explicitly overrides the inherited public key
        langfuse_client = get_client(public_key=override_public_key)
        langfuse_client.update_current_generation(metadata={"used_override": "true"})
        return "level_3"

    @observe()
    def level_2_function():
        # This function should use the overridden key when calling level_3
        level_3_function(langfuse_public_key=override_public_key)
        langfuse_client = get_client(public_key=primary_public_key)
        langfuse_client.update_current_span(metadata={"level": "2"})
        return "level_2"

    @observe()
    def level_1_function(*args, **kwargs):
        langfuse_client = get_client(public_key=primary_public_key)
        langfuse_client.update_current_trace(name=mock_name)
        level_2_function()
        return "level_1"

    result = level_1_function(
        langfuse_trace_id=mock_trace_id, langfuse_public_key=primary_public_key
    )
    client1.flush()
    client2.flush()

    assert result == "level_1"

    trace_data = get_api().trace.get(mock_trace_id)
    assert len(trace_data.observations) == 2
    assert trace_data.name == mock_name

    # Reset instances to not leak to other test suites
    removeMockResourceManagerInstances()


@pytest.mark.skip(
    reason="Somehow adding another client is polluting the global context in LangfuseResourceManager that makes other test suites fail."
)
def test_multiproject_context_propagation_no_public_key():
    # Initialize two separate Langfuse instances
    client1 = Langfuse()  # Reads from environment
    Langfuse(public_key="pk-test-project2", secret_key="sk-test-project2")

    # Verify both instances are registered
    assert len(LangfuseResourceManager._instances) == 2

    mock_name = "test_multiproject_context_propagation_no_public_key"
    env_public_key = os.environ[LANGFUSE_PUBLIC_KEY]
    langfuse = get_client(public_key=env_public_key)
    mock_trace_id = langfuse.create_trace_id()

    @observe(as_type="generation")
    def level_3_function():
        # Should use default client since no public key provided
        langfuse_client = get_client()
        langfuse_client.update_current_generation(metadata={"level": "3"})
        return "level_3"

    @observe()
    def level_2_function():
        result = level_3_function()
        langfuse_client = get_client()
        langfuse_client.update_current_span(metadata={"level": "2"})
        return result

    @observe()
    def level_1_function(*args, **kwargs):
        langfuse_client = get_client()
        langfuse_client.update_current_trace(name=mock_name)
        result = level_2_function()
        langfuse_client.update_current_span(metadata={"level": "1"})
        return result

    # No langfuse_public_key provided - should use default client
    result = level_1_function(langfuse_trace_id=mock_trace_id)
    client1.flush()

    assert result == "level_3"

    # Should skip tracing entirely in multi-project setup without public key
    # This is expected behavior to prevent cross-project data leakage
    try:
        trace_data = get_api().trace.get(mock_trace_id)
        # If trace is found, it should have no observations (tracing was skipped)
        assert len(trace_data.observations) == 0
    except Exception:
        # Trace not found is also expected - tracing was completely disabled
        pass

    # Reset instances to not leak to other test suites
    removeMockResourceManagerInstances()


@pytest.mark.skip(
    reason="Somehow adding another client is polluting the global context in LangfuseResourceManager that makes other test suites fail."
)
@pytest.mark.asyncio
async def test_multiproject_async_context_propagation_basic():
    """Test that nested async decorated functions inherit langfuse_public_key from parent in multi-project setup"""
    client1 = Langfuse()  # Reads from environment
    Langfuse(public_key="pk-test-project2", secret_key="sk-test-project2")

    # Verify both instances are registered
    assert len(LangfuseResourceManager._instances) == 2

    mock_name = "test_multiproject_async_context_propagation_basic"
    env_public_key = os.environ[LANGFUSE_PUBLIC_KEY]
    langfuse = get_client(public_key=env_public_key)
    mock_trace_id = langfuse.create_trace_id()

    @observe(as_type="generation", capture_output=False)
    async def async_level_3_function():
        # This function should inherit the public key from level_1_function
        # and NOT need langfuse_public_key parameter
        await asyncio.sleep(0.01)  # Simulate async work
        langfuse_client = get_client()
        langfuse_client.update_current_generation(
            metadata={"level": "3", "async": True}
        )
        langfuse_client.update_current_trace(name=mock_name)
        return "async_level_3"

    @observe()
    async def async_level_2_function():
        # This function should also inherit the public key
        result = await async_level_3_function()
        langfuse_client = get_client()
        langfuse_client.update_current_span(metadata={"level": "2", "async": True})
        return result

    @observe()
    async def async_level_1_function(*args, **kwargs):
        # Only this top-level function receives langfuse_public_key
        result = await async_level_2_function()
        langfuse_client = get_client()
        langfuse_client.update_current_span(metadata={"level": "1", "async": True})
        return result

    result = await async_level_1_function(
        *mock_args,
        **mock_kwargs,
        langfuse_trace_id=mock_trace_id,
        langfuse_public_key=env_public_key,  # Only provided to top-level function
    )

    # Use the correct client for flushing
    client1.flush()

    assert result == "async_level_3"

    # Verify trace was created properly
    trace_data = get_api().trace.get(mock_trace_id)
    assert len(trace_data.observations) == 3
    assert trace_data.name == mock_name

    # Verify all observations have async metadata
    async_flags = [
        obs.metadata.get("async") for obs in trace_data.observations if obs.metadata
    ]
    assert all(async_flags)

    # Reset instances to not leak to other test suites
    removeMockResourceManagerInstances()


@pytest.mark.skip(
    reason="Somehow adding another client is polluting the global context in LangfuseResourceManager that makes other test suites fail."
)
@pytest.mark.asyncio
async def test_multiproject_mixed_sync_async_context_propagation():
    """Test context propagation between sync and async decorated functions in multi-project setup"""
    client1 = Langfuse()  # Reads from environment
    Langfuse(public_key="pk-test-project2", secret_key="sk-test-project2")

    # Verify both instances are registered
    assert len(LangfuseResourceManager._instances) == 2

    mock_name = "test_multiproject_mixed_sync_async_context_propagation"
    env_public_key = os.environ[LANGFUSE_PUBLIC_KEY]
    langfuse = get_client(public_key=env_public_key)
    mock_trace_id = langfuse.create_trace_id()

    @observe(as_type="generation")
    def sync_level_4_function():
        # Sync function called from async should inherit context
        langfuse_client = get_client()
        langfuse_client.update_current_generation(
            metadata={"level": "4", "type": "sync"}
        )
        return "sync_level_4"

    @observe()
    async def async_level_3_function():
        # Async function calls sync function
        await asyncio.sleep(0.01)
        result = sync_level_4_function()
        langfuse_client = get_client()
        langfuse_client.update_current_span(metadata={"level": "3", "type": "async"})
        return result

    @observe()
    async def async_level_2_function():
        # Changed to async to avoid event loop issues
        result = await async_level_3_function()
        langfuse_client = get_client()
        langfuse_client.update_current_span(metadata={"level": "2", "type": "async"})
        return result

    @observe()
    async def async_level_1_function(*args, **kwargs):
        # Top-level async function
        langfuse_client = get_client()
        langfuse_client.update_current_trace(name=mock_name)
        result = await async_level_2_function()
        langfuse_client.update_current_span(metadata={"level": "1", "type": "async"})
        return result

    result = await async_level_1_function(
        langfuse_trace_id=mock_trace_id, langfuse_public_key=env_public_key
    )
    client1.flush()

    assert result == "sync_level_4"

    trace_data = get_api().trace.get(mock_trace_id)
    assert len(trace_data.observations) == 4
    assert trace_data.name == mock_name

    # Verify mixed sync/async execution
    types = [
        obs.metadata.get("type") for obs in trace_data.observations if obs.metadata
    ]
    assert "sync" in types
    assert "async" in types

    # Reset instances to not leak to other test suites
    removeMockResourceManagerInstances()


@pytest.mark.skip(
    reason="Somehow adding another client is polluting the global context in LangfuseResourceManager that makes other test suites fail."
)
@pytest.mark.asyncio
async def test_multiproject_concurrent_async_context_isolation():
    """Test that concurrent async executions don't interfere with each other's context in multi-project setup"""
    client1 = Langfuse()  # Reads from environment
    Langfuse(public_key="pk-test-project2", secret_key="sk-test-project2")

    # Verify both instances are registered
    assert len(LangfuseResourceManager._instances) == 2

    mock_name = "test_multiproject_concurrent_async_context_isolation"
    env_public_key = os.environ[LANGFUSE_PUBLIC_KEY]
    langfuse = get_client(public_key=env_public_key)

    trace_id_1 = langfuse.create_trace_id()
    trace_id_2 = langfuse.create_trace_id()

    # Use the same valid public key for both tasks to avoid credential issues
    # The isolation test is about trace contexts, not different projects
    public_key_1 = env_public_key
    public_key_2 = env_public_key

    @observe(as_type="generation")
    async def async_level_3_function(task_id):
        # Simulate work and ensure contexts don't leak
        await asyncio.sleep(0.1)  # Ensure concurrency overlap
        langfuse_client = get_client()
        langfuse_client.update_current_generation(
            metadata={"task_id": task_id, "level": "3"}
        )
        return f"async_level_3_task_{task_id}"

    @observe()
    async def async_level_2_function(task_id):
        result = await async_level_3_function(task_id)
        langfuse_client = get_client()
        langfuse_client.update_current_span(metadata={"task_id": task_id, "level": "2"})
        return result

    @observe()
    async def async_level_1_function(task_id, *args, **kwargs):
        langfuse_client = get_client()
        langfuse_client.update_current_trace(name=f"{mock_name}_task_{task_id}")
        result = await async_level_2_function(task_id)
        langfuse_client.update_current_span(metadata={"task_id": task_id, "level": "1"})
        return result

    # Run two concurrent async tasks with the same public key but different trace contexts
    task1 = async_level_1_function(
        "1", langfuse_trace_id=trace_id_1, langfuse_public_key=public_key_1
    )
    task2 = async_level_1_function(
        "2", langfuse_trace_id=trace_id_2, langfuse_public_key=public_key_2
    )

    result1, result2 = await asyncio.gather(task1, task2)

    client1.flush()

    assert result1 == "async_level_3_task_1"
    assert result2 == "async_level_3_task_2"

    # Verify both traces were created correctly and didn't interfere
    trace_data_1 = get_api().trace.get(trace_id_1)
    trace_data_2 = get_api().trace.get(trace_id_2)

    assert trace_data_1.name == f"{mock_name}_task_1"
    assert trace_data_2.name == f"{mock_name}_task_2"

    # Verify that both traces have the expected number of observations (context propagation worked)
    assert (
        len(trace_data_1.observations) == 3
    )  # All 3 levels should be captured for task 1
    assert (
        len(trace_data_2.observations) == 3
    )  # All 3 levels should be captured for task 2

    # Verify traces are properly isolated (no cross-contamination)
    trace_1_names = [obs.name for obs in trace_data_1.observations]
    trace_2_names = [obs.name for obs in trace_data_2.observations]
    assert "async_level_1_function" in trace_1_names
    assert "async_level_2_function" in trace_1_names
    assert "async_level_3_function" in trace_1_names
    assert "async_level_1_function" in trace_2_names
    assert "async_level_2_function" in trace_2_names
    assert "async_level_3_function" in trace_2_names

    # Reset instances to not leak to other test suites
    removeMockResourceManagerInstances()


@pytest.mark.skip(
    reason="Somehow adding another client is polluting the global context in LangfuseResourceManager that makes other test suites fail."
)
@pytest.mark.asyncio
async def test_multiproject_async_generator_context_propagation():
    """Test context propagation with async generators in multi-project setup"""
    client1 = Langfuse()  # Reads from environment
    Langfuse(public_key="pk-test-project2", secret_key="sk-test-project2")

    # Verify both instances are registered
    assert len(LangfuseResourceManager._instances) == 2

    mock_name = "test_multiproject_async_generator_context_propagation"
    env_public_key = os.environ[LANGFUSE_PUBLIC_KEY]
    langfuse = get_client(public_key=env_public_key)
    mock_trace_id = langfuse.create_trace_id()

    @observe(capture_output=True)
    async def async_generator_function():
        # Async generator should inherit context from parent
        await asyncio.sleep(0.01)
        yield "Hello"
        await asyncio.sleep(0.01)
        yield ", "
        await asyncio.sleep(0.01)
        yield "Async"
        await asyncio.sleep(0.01)
        yield " World!"

    @observe()
    async def async_consumer_function():
        langfuse_client = get_client()
        langfuse_client.update_current_trace(name=mock_name)

        result = ""
        async for item in async_generator_function():
            result += item

        langfuse_client.update_current_span(
            metadata={"type": "consumer", "result": result}
        )
        return result

    result = await async_consumer_function(
        langfuse_trace_id=mock_trace_id, langfuse_public_key=env_public_key
    )
    client1.flush()

    assert result == "Hello, Async World!"

    trace_data = get_api().trace.get(mock_trace_id)
    assert len(trace_data.observations) == 2
    assert trace_data.name == mock_name

    # Verify both generator and consumer were captured by name (most reliable test)
    observation_names = [obs.name for obs in trace_data.observations]
    assert "async_generator_function" in observation_names
    assert "async_consumer_function" in observation_names

    # Verify that context propagation worked - both functions should be in the same trace
    # This confirms that the async generator inherited the public key context
    assert len(trace_data.observations) == 2

    # Reset instances to not leak to other test suites
    removeMockResourceManagerInstances()


@pytest.mark.skip(
    reason="Somehow adding another client is polluting the global context in LangfuseResourceManager that makes other test suites fail."
)
@pytest.mark.asyncio
async def test_multiproject_async_context_exception_handling():
    """Test that async context is properly restored even when exceptions occur in multi-project setup"""
    client1 = Langfuse()  # Reads from environment
    Langfuse(public_key="pk-test-project2", secret_key="sk-test-project2")

    # Verify both instances are registered
    assert len(LangfuseResourceManager._instances) == 2

    mock_name = "test_multiproject_async_context_exception_handling"
    env_public_key = os.environ[LANGFUSE_PUBLIC_KEY]
    langfuse = get_client(public_key=env_public_key)
    mock_trace_id = langfuse.create_trace_id()

    @observe(as_type="generation")
    async def async_failing_function():
        # This function should inherit context but will raise an exception
        await asyncio.sleep(0.01)
        langfuse_client = get_client()
        langfuse_client.update_current_generation(metadata={"will_fail": True})
        langfuse_client.update_current_trace(name=mock_name)
        raise ValueError("Async function failed")

    @observe()
    async def async_caller_function():
        try:
            await async_failing_function()
        except ValueError:
            # Context should still be available here
            langfuse_client = get_client()
            langfuse_client.update_current_span(metadata={"caught_exception": True})
            return "exception_handled"

    @observe()
    async def async_root_function(*args, **kwargs):
        result = await async_caller_function()
        # Context should still be available after exception
        langfuse_client = get_client()
        langfuse_client.update_current_span(metadata={"root": True})
        return result

    result = await async_root_function(
        langfuse_trace_id=mock_trace_id, langfuse_public_key=env_public_key
    )
    client1.flush()

    assert result == "exception_handled"

    trace_data = get_api().trace.get(mock_trace_id)
    assert len(trace_data.observations) == 3
    assert trace_data.name == mock_name

    # Verify exception was properly handled and context maintained
    exception_obs = next(obs for obs in trace_data.observations if obs.level == "ERROR")
    assert exception_obs.status_message == "Async function failed"

    caught_obs = next(
        obs
        for obs in trace_data.observations
        if obs.metadata and obs.metadata.get("caught_exception")
    )
    assert caught_obs is not None

    # Reset instances to not leak to other test suites
    removeMockResourceManagerInstances()
