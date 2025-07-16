import asyncio
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from time import sleep
from typing import Optional

import pytest
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from langfuse import get_client, observe
from langfuse.langchain import CallbackHandler
from langfuse.media import LangfuseMedia
from tests.utils import get_api

mock_metadata = {"key": "metadata"}
mock_deep_metadata = {"key": "mock_deep_metadata"}
mock_session_id = "session-id-1"
mock_args = (1, 2, 3)
mock_kwargs = {"a": 1, "b": 2, "c": 3}


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
