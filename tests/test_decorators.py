import asyncio
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar
from time import sleep
from typing import Optional

import pytest
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from langfuse.decorators import langfuse_context, observe
from langfuse.media import LangfuseMedia
from langfuse.openai import AsyncOpenAI
from tests.utils import create_uuid, get_api, get_llama_index_index

mock_metadata = {"key": "metadata"}
mock_deep_metadata = {"key": "mock_deep_metadata"}
mock_session_id = "session-id-1"
mock_args = (1, 2, 3)
mock_kwargs = {"a": 1, "b": 2, "c": 3}


def test_nested_observations():
    mock_name = "test_nested_observations"
    mock_trace_id = create_uuid()

    @observe(as_type="generation", name="level_3_to_be_overwritten")
    def level_3_function():
        langfuse_context.update_current_observation(metadata=mock_metadata)
        langfuse_context.update_current_observation(
            metadata=mock_deep_metadata,
            usage={"input": 150, "output": 50, "total": 300},
            model="gpt-3.5-turbo",
            output="mock_output",
        )
        langfuse_context.update_current_observation(
            version="version-1", name="overwritten_level_3"
        )

        langfuse_context.update_current_trace(
            session_id=mock_session_id, name=mock_name
        )

        langfuse_context.update_current_trace(
            user_id="user_id",
        )

        return "level_3"

    @observe(name="level_2_manually_set")
    def level_2_function():
        level_3_function()
        langfuse_context.update_current_observation(metadata=mock_metadata)

        return "level_2"

    @observe()
    def level_1_function(*args, **kwargs):
        level_2_function()

        return "level_1"

    result = level_1_function(
        *mock_args, **mock_kwargs, langfuse_observation_id=mock_trace_id
    )
    langfuse_context.flush()

    assert result == "level_1"  # Wrapped function returns correctly

    # ID setting for span or trace

    trace_data = get_api().trace.get(mock_trace_id)
    assert (
        len(trace_data.observations) == 2
    )  # Top-most function is trace, so it's not an observations

    assert trace_data.input == {"args": list(mock_args), "kwargs": mock_kwargs}
    assert trace_data.output == "level_1"

    # trace parameters if set anywhere in the call stack
    assert trace_data.session_id == mock_session_id
    assert trace_data.user_id == "user_id"
    assert trace_data.name == mock_name

    # Check correct nesting
    adjacencies = defaultdict(list)
    for o in trace_data.observations:
        adjacencies[o.parent_observation_id or o.trace_id].append(o)

    assert len(adjacencies[mock_trace_id]) == 1  # Trace has only one child
    assert len(adjacencies) == 2  # Only trace and one observation have children

    level_2_observation = adjacencies[mock_trace_id][0]
    level_3_observation = adjacencies[level_2_observation.id][0]

    assert level_2_observation.name == "level_2_manually_set"
    assert level_2_observation.metadata == mock_metadata

    assert level_3_observation.name == "overwritten_level_3"
    assert level_3_observation.metadata == mock_deep_metadata
    assert level_3_observation.type == "GENERATION"
    assert level_3_observation.calculated_total_cost > 0
    assert level_3_observation.output == "mock_output"
    assert level_3_observation.version == "version-1"


def test_nested_observations_with_non_parentheses_decorator():
    mock_name = "test_nested_observations"
    mock_trace_id = create_uuid()

    @observe(as_type="generation", name="level_3_to_be_overwritten")
    def level_3_function():
        langfuse_context.update_current_observation(metadata=mock_metadata)
        langfuse_context.update_current_observation(
            metadata=mock_deep_metadata,
            usage={"input": 150, "output": 50, "total": 300},
            model="gpt-3.5-turbo",
            output="mock_output",
        )
        langfuse_context.update_current_observation(
            version="version-1", name="overwritten_level_3"
        )

        langfuse_context.update_current_trace(
            session_id=mock_session_id, name=mock_name
        )

        langfuse_context.update_current_trace(
            user_id="user_id",
        )

        return "level_3"

    @observe
    def level_2_function():
        level_3_function()
        langfuse_context.update_current_observation(metadata=mock_metadata)

        return "level_2"

    @observe
    def level_1_function(*args, **kwargs):
        level_2_function()

        return "level_1"

    result = level_1_function(
        *mock_args, **mock_kwargs, langfuse_observation_id=mock_trace_id
    )
    langfuse_context.flush()

    assert result == "level_1"  # Wrapped function returns correctly

    # ID setting for span or trace

    trace_data = get_api().trace.get(mock_trace_id)
    assert (
        len(trace_data.observations) == 2
    )  # Top-most function is trace, so it's not an observations

    assert trace_data.input == {"args": list(mock_args), "kwargs": mock_kwargs}
    assert trace_data.output == "level_1"

    # trace parameters if set anywhere in the call stack
    assert trace_data.session_id == mock_session_id
    assert trace_data.user_id == "user_id"
    assert trace_data.name == mock_name

    # Check correct nesting
    adjacencies = defaultdict(list)
    for o in trace_data.observations:
        adjacencies[o.parent_observation_id or o.trace_id].append(o)

    assert len(adjacencies[mock_trace_id]) == 1  # Trace has only one child
    assert len(adjacencies) == 2  # Only trace and one observation have children

    level_2_observation = adjacencies[mock_trace_id][0]
    level_3_observation = adjacencies[level_2_observation.id][0]

    assert level_2_observation.name == "level_2_function"
    assert level_2_observation.metadata == mock_metadata

    assert level_3_observation.name == "overwritten_level_3"
    assert level_3_observation.metadata == mock_deep_metadata
    assert level_3_observation.type == "GENERATION"
    assert level_3_observation.calculated_total_cost > 0
    assert level_3_observation.output == "mock_output"
    assert level_3_observation.version == "version-1"


# behavior on exceptions
def test_exception_in_wrapped_function():
    mock_name = "test_exception_in_wrapped_function"
    mock_trace_id = create_uuid()

    @observe(as_type="generation")
    def level_3_function():
        langfuse_context.update_current_observation(metadata=mock_metadata)
        langfuse_context.update_current_observation(
            metadata=mock_deep_metadata,
            usage={"input": 150, "output": 50, "total": 300},
            model="gpt-3.5-turbo",
        )
        langfuse_context.update_current_trace(
            session_id=mock_session_id, name=mock_name
        )

        raise ValueError("Mock exception")

    @observe()
    def level_2_function():
        level_3_function()
        langfuse_context.update_current_observation(metadata=mock_metadata)

        return "level_2"

    @observe()
    def level_1_function(*args, **kwargs):
        level_2_function()

        return "level_1"

    # Check that the exception is raised
    with pytest.raises(ValueError):
        level_1_function(
            *mock_args, **mock_kwargs, langfuse_observation_id=mock_trace_id
        )

    langfuse_context.flush()
    sleep(1)

    trace_data = get_api().trace.get(mock_trace_id)

    assert trace_data.input == {"args": list(mock_args), "kwargs": mock_kwargs}
    assert trace_data.output is None  # Output is None if exception is raised

    # trace parameters if set anywhere in the call stack
    assert trace_data.session_id == mock_session_id
    assert trace_data.name == mock_name

    # Check correct nesting
    adjacencies = defaultdict(list)
    for o in trace_data.observations:
        adjacencies[o.parent_observation_id or o.trace_id].append(o)

    assert len(adjacencies[mock_trace_id]) == 1  # Trace has only one child
    assert len(adjacencies) == 2  # Only trace and one observation have children

    level_2_observation = adjacencies[mock_trace_id][0]
    level_3_observation = adjacencies[level_2_observation.id][0]

    assert (
        level_2_observation.metadata == {}
    )  # Exception is raised before metadata is set
    assert level_3_observation.metadata == mock_deep_metadata
    assert level_3_observation.status_message == "Mock exception"
    assert level_3_observation.level == "ERROR"


# behavior on concurrency
def test_concurrent_decorator_executions():
    mock_name = "test_concurrent_decorator_executions"
    mock_trace_id_1 = create_uuid()
    mock_trace_id_2 = create_uuid()

    @observe(as_type="generation")
    def level_3_function():
        langfuse_context.update_current_observation(metadata=mock_metadata)
        langfuse_context.update_current_observation(metadata=mock_deep_metadata)
        langfuse_context.update_current_observation(
            metadata=mock_deep_metadata,
            usage={"input": 150, "output": 50, "total": 300},
            model="gpt-3.5-turbo",
        )
        langfuse_context.update_current_trace(session_id=mock_session_id)

        return "level_3"

    @observe()
    def level_2_function():
        level_3_function()
        langfuse_context.update_current_observation(metadata=mock_metadata)

        return "level_2"

    @observe(name=mock_name)
    def level_1_function(*args, **kwargs):
        level_2_function()

        return "level_1"

    with ThreadPoolExecutor(max_workers=2) as executor:
        future1 = executor.submit(
            level_1_function,
            *mock_args,
            mock_trace_id_1,
            **mock_kwargs,
            langfuse_observation_id=mock_trace_id_1,
        )
        future2 = executor.submit(
            level_1_function,
            *mock_args,
            mock_trace_id_2,
            **mock_kwargs,
            langfuse_observation_id=mock_trace_id_2,
        )

        future1.result()
        future2.result()

    langfuse_context.flush()

    sleep(1)

    for mock_id in [mock_trace_id_1, mock_trace_id_2]:
        trace_data = get_api().trace.get(mock_id)
        assert (
            len(trace_data.observations) == 2
        )  # Top-most function is trace, so it's not an observations

        assert trace_data.input == {
            "args": list(mock_args) + [mock_id],
            "kwargs": mock_kwargs,
        }
        assert trace_data.output == "level_1"

        # trace parameters if set anywhere in the call stack
        assert trace_data.session_id == mock_session_id
        assert trace_data.name == mock_name

        # Check correct nesting
        adjacencies = defaultdict(list)
        for o in trace_data.observations:
            adjacencies[o.parent_observation_id or o.trace_id].append(o)

        assert len(adjacencies[mock_id]) == 1  # Trace has only one child
        assert len(adjacencies) == 2  # Only trace and one observation have children

        level_2_observation = adjacencies[mock_id][0]
        level_3_observation = adjacencies[level_2_observation.id][0]

        assert level_2_observation.metadata == mock_metadata
        assert level_3_observation.metadata == mock_deep_metadata
        assert level_3_observation.type == "GENERATION"
        assert level_3_observation.calculated_total_cost > 0


def test_decorators_llama_index():
    mock_name = "test_decorators_llama_index"
    mock_trace_id = create_uuid()

    @observe()
    def llama_index_operations(*args, **kwargs):
        callback = langfuse_context.get_current_llama_index_handler()
        index = get_llama_index_index(callback, force_rebuild=True)

        return index.as_query_engine().query(kwargs["query"])

    @observe()
    def level_3_function(*args, **kwargs):
        langfuse_context.update_current_observation(metadata=mock_metadata)
        langfuse_context.update_current_observation(metadata=mock_deep_metadata)
        langfuse_context.update_current_trace(
            session_id=mock_session_id, name=mock_name
        )

        return llama_index_operations(*args, **kwargs)

    @observe()
    def level_2_function(*args, **kwargs):
        langfuse_context.update_current_observation(metadata=mock_metadata)

        return level_3_function(*args, **kwargs)

    @observe()
    def level_1_function(*args, **kwargs):
        return level_2_function(*args, **kwargs)

    level_1_function(
        query="What is the authors ambition?", langfuse_observation_id=mock_trace_id
    )

    langfuse_context.flush()

    trace_data = get_api().trace.get(mock_trace_id)
    assert len(trace_data.observations) > 2

    # Check correct nesting
    adjacencies = defaultdict(list)
    for o in trace_data.observations:
        adjacencies[o.parent_observation_id or o.trace_id].append(o)

    assert len(adjacencies[mock_trace_id]) == 1  # Trace has only one child

    # Check that the llama_index_operations is at the correct level
    lvl = 1
    curr_id = mock_trace_id
    llama_index_root_span = None

    while len(adjacencies[curr_id]) > 0:
        o = adjacencies[curr_id][0]
        if o.name == "llama_index_operations":
            llama_index_root_span = o
            break

        curr_id = adjacencies[curr_id][0].id
        lvl += 1

    assert lvl == 3

    assert llama_index_root_span is not None
    assert any([o.name == "OpenAIEmbedding" for o in trace_data.observations])


def test_decorators_langchain():
    mock_name = "test_decorators_langchain"
    mock_trace_id = create_uuid()

    @observe()
    def langchain_operations(*args, **kwargs):
        handler = langfuse_context.get_current_langchain_handler()
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
        langfuse_context.update_current_observation(metadata=mock_metadata)
        langfuse_context.update_current_observation(metadata=mock_deep_metadata)
        langfuse_context.update_current_trace(
            session_id=mock_session_id, name=mock_name
        )

        return langchain_operations(*args, **kwargs)

    @observe()
    def level_2_function(*args, **kwargs):
        langfuse_context.update_current_observation(metadata=mock_metadata)

        return level_3_function(*args, **kwargs)

    @observe()
    def level_1_function(*args, **kwargs):
        return level_2_function(*args, **kwargs)

    level_1_function(topic="socks", langfuse_observation_id=mock_trace_id)

    langfuse_context.flush()

    trace_data = get_api().trace.get(mock_trace_id)
    assert len(trace_data.observations) > 2

    # Check correct nesting
    adjacencies = defaultdict(list)
    for o in trace_data.observations:
        adjacencies[o.parent_observation_id or o.trace_id].append(o)

    assert len(adjacencies[mock_trace_id]) == 1  # Trace has only one child

    # Check that the langchain_operations is at the correct level
    lvl = 1
    curr_id = mock_trace_id
    llama_index_root_span = None

    while len(adjacencies[curr_id]) > 0:
        o = adjacencies[curr_id][0]
        if o.name == "langchain_operations":
            llama_index_root_span = o
            break

        curr_id = adjacencies[curr_id][0].id
        lvl += 1

    assert lvl == 3

    assert llama_index_root_span is not None
    assert any([o.name == "ChatPromptTemplate" for o in trace_data.observations])


@pytest.mark.asyncio
async def test_asyncio_concurrency_inside_nested_span():
    mock_name = "test_asyncio_concurrency_inside_nested_span"
    mock_trace_id = create_uuid()
    mock_observation_id_1 = create_uuid()
    mock_observation_id_2 = create_uuid()

    @observe(as_type="generation")
    async def level_3_function():
        langfuse_context.update_current_observation(metadata=mock_metadata)
        langfuse_context.update_current_observation(
            metadata=mock_deep_metadata,
            usage={"input": 150, "output": 50, "total": 300},
            model="gpt-3.5-turbo",
        )
        langfuse_context.update_current_trace(
            session_id=mock_session_id, name=mock_name
        )

        return "level_3"

    @observe()
    async def level_2_function(*args, **kwargs):
        await level_3_function()
        langfuse_context.update_current_observation(metadata=mock_metadata)

        return "level_2"

    @observe()
    async def level_1_function(*args, **kwargs):
        print("Executing level 1")
        await asyncio.gather(
            level_2_function(
                *mock_args,
                mock_observation_id_1,
                **mock_kwargs,
                langfuse_observation_id=mock_observation_id_1,
            ),
            level_2_function(
                *mock_args,
                mock_observation_id_2,
                **mock_kwargs,
                langfuse_observation_id=mock_observation_id_2,
            ),
        )

        return "level_1"

    await level_1_function(langfuse_observation_id=mock_trace_id)
    langfuse_context.flush()

    trace_data = get_api().trace.get(mock_trace_id)
    assert (
        len(trace_data.observations) == 4
    )  # Top-most function is trace, so it's not an observations

    # trace parameters if set anywhere in the call stack
    assert trace_data.name == mock_name
    assert trace_data.session_id == mock_session_id
    assert trace_data.output == "level_1"

    # Check correct nesting
    adjacencies = defaultdict(list)
    for o in trace_data.observations:
        adjacencies[o.parent_observation_id or o.trace_id].append(o)

    # Trace has two children
    assert len(adjacencies[mock_trace_id]) == 2

    # Each async call has one child
    for mock_id in [mock_observation_id_1, mock_observation_id_2]:
        assert len(adjacencies[mock_id]) == 1

    assert (
        len(adjacencies) == 3
    )  # Only trace and the two lvl-2 observation have children


def test_get_current_ids():
    mock_trace_id = create_uuid()
    mock_deep_observation_id = create_uuid()

    retrieved_trace_id: ContextVar[Optional[str]] = ContextVar(
        "retrieved_trace_id", default=None
    )
    retrieved_observation_id: ContextVar[Optional[str]] = ContextVar(
        "retrieved_observation_id", default=None
    )

    @observe()
    def level_3_function(*args, **kwargs):
        retrieved_trace_id.set(langfuse_context.get_current_trace_id())
        retrieved_observation_id.set(langfuse_context.get_current_observation_id())

        return "level_3"

    @observe()
    def level_2_function():
        return level_3_function(langfuse_observation_id=mock_deep_observation_id)

    @observe()
    def level_1_function(*args, **kwargs):
        level_2_function()

        return "level_1"

    result = level_1_function(
        *mock_args, **mock_kwargs, langfuse_observation_id=mock_trace_id
    )
    langfuse_context.flush()

    assert result == "level_1"  # Wrapped function returns correctly

    # ID setting for span or trace
    trace_data = get_api().trace.get(mock_trace_id)

    assert retrieved_trace_id.get() == mock_trace_id
    assert retrieved_observation_id.get() == mock_deep_observation_id
    assert any(
        [o.id == retrieved_observation_id.get() for o in trace_data.observations]
    )


def test_get_current_trace_url():
    mock_trace_id = create_uuid()

    @observe()
    def level_3_function():
        return langfuse_context.get_current_trace_url()

    @observe()
    def level_2_function():
        return level_3_function()

    @observe()
    def level_1_function(*args, **kwargs):
        return level_2_function()

    result = level_1_function(
        *mock_args, **mock_kwargs, langfuse_observation_id=mock_trace_id
    )
    langfuse_context.flush()

    expected_url = f"http://localhost:3000/project/7a88fb47-b4e2-43b8-a06c-a5ce950dc53a/traces/{mock_trace_id}"
    assert result == expected_url


def test_scoring_observations():
    mock_name = "test_scoring_observations"
    mock_trace_id = create_uuid()

    @observe(as_type="generation")
    def level_3_function():
        langfuse_context.score_current_observation(
            name="test-observation-score", value=1
        )
        langfuse_context.score_current_trace(
            name="another-test-trace-score", value="my_value"
        )
        return "level_3"

    @observe()
    def level_2_function():
        return level_3_function()

    @observe()
    def level_1_function(*args, **kwargs):
        langfuse_context.score_current_observation(name="test-trace-score", value=3)
        langfuse_context.update_current_trace(name=mock_name)
        return level_2_function()

    result = level_1_function(
        *mock_args, **mock_kwargs, langfuse_observation_id=mock_trace_id
    )
    langfuse_context.flush()

    assert result == "level_3"  # Wrapped function returns correctly

    # ID setting for span or trace
    trace_data = get_api().trace.get(mock_trace_id)
    assert (
        len(trace_data.observations) == 2
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
    mock_trace_id = create_uuid()

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
    result = function_with_circular_arg(
        circular_obj, langfuse_observation_id=mock_trace_id
    )

    langfuse_context.flush()

    # Validate that the function executed as expected
    assert result == "function response"

    trace_data = get_api().trace.get(mock_trace_id)

    assert trace_data.input["args"][0]["reference"] == "CircularRefObject"


def test_disabled_io_capture():
    mock_trace_id = create_uuid()

    class Node:
        def __init__(self, value: tuple):
            self.value = value

    @observe(capture_input=False, capture_output=False)
    def nested(*args, **kwargs):
        langfuse_context.update_current_observation(
            input=Node(("manually set tuple", 1)), output="manually set output"
        )
        return "nested response"

    @observe(capture_output=False)
    def main(*args, **kwargs):
        nested(*args, **kwargs)
        return "function response"

    result = main("Hello, World!", name="John", langfuse_observation_id=mock_trace_id)

    langfuse_context.flush()

    assert result == "function response"

    trace_data = get_api().trace.get(mock_trace_id)

    assert trace_data.input == {"args": ["Hello, World!"], "kwargs": {"name": "John"}}
    assert trace_data.output is None

    # Check that disabled capture_io doesn't capture manually set input/output
    assert len(trace_data.observations) == 1
    assert trace_data.observations[0].input["value"] == ["manually set tuple", 1]
    assert trace_data.observations[0].output == "manually set output"


def test_decorated_class_and_instance_methods():
    mock_name = "test_decorated_class_and_instance_methods"
    mock_trace_id = create_uuid()

    class TestClass:
        @classmethod
        @observe()
        def class_method(cls, *args, **kwargs):
            langfuse_context.update_current_observation(name="class_method")
            return "class_method"

        @observe(as_type="generation")
        def level_3_function(self):
            langfuse_context.update_current_observation(metadata=mock_metadata)
            langfuse_context.update_current_observation(
                metadata=mock_deep_metadata,
                usage={"input": 150, "output": 50, "total": 300},
                model="gpt-3.5-turbo",
                output="mock_output",
            )

            langfuse_context.update_current_trace(
                session_id=mock_session_id, name=mock_name
            )

            return "level_3"

        @observe()
        def level_2_function(self):
            TestClass.class_method()

            self.level_3_function()
            langfuse_context.update_current_observation(metadata=mock_metadata)

            return "level_2"

        @observe()
        def level_1_function(self, *args, **kwargs):
            self.level_2_function()

            return "level_1"

    result = TestClass().level_1_function(
        *mock_args, **mock_kwargs, langfuse_observation_id=mock_trace_id
    )

    langfuse_context.flush()

    assert result == "level_1"  # Wrapped function returns correctly

    # ID setting for span or trace

    trace_data = get_api().trace.get(mock_trace_id)
    assert (
        len(trace_data.observations) == 3
    )  # Top-most function is trace, so it's not an observations

    assert trace_data.input == {"args": list(mock_args), "kwargs": mock_kwargs}
    assert trace_data.output == "level_1"

    # trace parameters if set anywhere in the call stack
    assert trace_data.session_id == mock_session_id
    assert trace_data.name == mock_name

    # Check correct nesting
    adjacencies = defaultdict(list)
    for o in trace_data.observations:
        adjacencies[o.parent_observation_id or o.trace_id].append(o)

    assert len(adjacencies[mock_trace_id]) == 1  # Trace has only one child
    assert len(adjacencies) == 2  # Only trace and one observation have children

    level_2_observation = adjacencies[mock_trace_id][0]
    class_method_observation = [
        o for o in adjacencies[level_2_observation.id] if o.name == "class_method"
    ][0]
    level_3_observation = [
        o for o in adjacencies[level_2_observation.id] if o.name != "class_method"
    ][0]

    assert class_method_observation.input == {"args": [], "kwargs": {}}
    assert class_method_observation.output == "class_method"

    assert level_2_observation.metadata == mock_metadata
    assert level_3_observation.metadata == mock_deep_metadata
    assert level_3_observation.type == "GENERATION"
    assert level_3_observation.calculated_total_cost > 0
    assert level_3_observation.output == "mock_output"


def test_generator_as_return_value():
    mock_trace_id = create_uuid()
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

    result = main(langfuse_observation_id=mock_trace_id)
    langfuse_context.flush()

    assert result == mock_output

    trace_data = get_api().trace.get(mock_trace_id)
    assert trace_data.output == mock_output

    assert trace_data.observations[0].output == "Hello--, --World!"


@pytest.mark.asyncio
async def test_async_generator_as_return_value():
    mock_trace_id = create_uuid()
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

    @observe(transform_to_string=custom_transform_to_string)
    async def nested_async():
        gen = async_generator_function()
        print(type(gen))

        async for item in gen:
            yield item

    @observe()
    async def main_async(**kwargs):
        gen = nested_async()

        result = ""
        async for item in gen:
            result += item

        return result

    result = await main_async(langfuse_observation_id=mock_trace_id)
    langfuse_context.flush()

    assert result == mock_output

    trace_data = get_api().trace.get(mock_trace_id)
    assert trace_data.output == result

    assert trace_data.observations[0].output == "Hello--, async --World!"
    assert trace_data.observations[1].output == "Hello--, async --World!"


@pytest.mark.asyncio
async def test_async_nested_openai_chat_stream():
    mock_name = "test_async_nested_openai_chat_stream"
    mock_trace_id = create_uuid()
    mock_tags = ["tag1", "tag2"]
    mock_session_id = "session-id-1"
    mock_user_id = "user-id-1"
    mock_generation_name = "openai generation"

    @observe()
    async def level_2_function():
        gen = await AsyncOpenAI().chat.completions.create(
            name=mock_generation_name,
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "1 + 1 = "}],
            temperature=0,
            metadata={"someKey": "someResponse"},
            session_id=mock_session_id,
            user_id=mock_user_id,
            tags=mock_tags,
            stream=True,
        )

        async for c in gen:
            print(c)

        langfuse_context.update_current_observation(metadata=mock_metadata)
        langfuse_context.update_current_trace(name=mock_name)

        return "level_2"

    @observe()
    async def level_1_function(*args, **kwargs):
        await level_2_function()

        return "level_1"

    result = await level_1_function(
        *mock_args, **mock_kwargs, langfuse_observation_id=mock_trace_id
    )
    langfuse_context.flush()

    assert result == "level_1"  # Wrapped function returns correctly

    # ID setting for span or trace
    trace_data = get_api().trace.get(mock_trace_id)
    assert (
        len(trace_data.observations) == 2
    )  # Top-most function is trace, so it's not an observations

    assert trace_data.input == {"args": list(mock_args), "kwargs": mock_kwargs}
    assert trace_data.output == "level_1"

    # trace parameters if set anywhere in the call stack
    assert trace_data.session_id == mock_session_id
    assert trace_data.name == mock_name

    # Check correct nesting
    adjacencies = defaultdict(list)
    for o in trace_data.observations:
        adjacencies[o.parent_observation_id or o.trace_id].append(o)

    assert len(adjacencies[mock_trace_id]) == 1  # Trace has only one child
    assert len(adjacencies) == 2  # Only trace and one observation have children

    level_2_observation = adjacencies[mock_trace_id][0]
    level_3_observation = adjacencies[level_2_observation.id][0]

    assert level_2_observation.metadata == mock_metadata

    generation = level_3_observation

    assert generation.name == mock_generation_name
    assert generation.metadata == {"someKey": "someResponse"}
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
        "max_tokens": "inf",
        "presence_penalty": 0,
    }
    assert generation.usage.input is not None
    assert generation.usage.output is not None
    assert generation.usage.total is not None
    print(generation)
    assert generation.output == 2


def test_generation_at_highest_level():
    mock_trace_id = create_uuid()
    mock_result = "Hello, World!"

    @observe(as_type="generation")
    def main():
        return mock_result

    result = main(langfuse_observation_id=mock_trace_id)
    langfuse_context.flush()

    assert result == mock_result

    trace_data = get_api().trace.get(mock_trace_id)
    assert (
        trace_data.output is None
    )  # output will be attributed to generation observation

    # Check that the generation is wrapped inside a trace
    assert len(trace_data.observations) == 1

    generation = trace_data.observations[0]
    assert generation.type == "GENERATION"
    assert generation.output == result


def test_generator_as_function_input():
    mock_trace_id = create_uuid()
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

    result = main(langfuse_observation_id=mock_trace_id)
    langfuse_context.flush()

    assert result == mock_output

    trace_data = get_api().trace.get(mock_trace_id)
    assert trace_data.output == mock_output

    assert trace_data.observations[0].input["args"][0] == "<generator>"
    assert trace_data.observations[0].output == "Hello, World!"

    observation_start_time = trace_data.observations[0].start_time
    observation_end_time = trace_data.observations[0].end_time

    assert observation_start_time is not None
    assert observation_end_time is not None
    assert observation_start_time <= observation_end_time


def test_nest_list_of_generator_as_function_IO():
    mock_trace_id = create_uuid()

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

    main(langfuse_observation_id=mock_trace_id)
    langfuse_context.flush()

    trace_data = get_api().trace.get(mock_trace_id)

    assert [[["<generator>", "<generator>"]]] == trace_data.observations[0].input[
        "args"
    ]

    assert all(
        ["generator" in arg for arg in trace_data.observations[0].output[0]],
    )

    observation_start_time = trace_data.observations[0].start_time
    observation_end_time = trace_data.observations[0].end_time

    assert observation_start_time is not None
    assert observation_end_time is not None
    assert observation_start_time <= observation_end_time


def test_return_dict_for_output():
    mock_trace_id = create_uuid()
    mock_output = {"key": "value"}

    @observe()
    def function():
        return mock_output

    result = function(langfuse_observation_id=mock_trace_id)
    langfuse_context.flush()

    assert result == mock_output

    trace_data = get_api().trace.get(mock_trace_id)
    assert trace_data.output == mock_output


def test_manual_context_copy_in_threadpoolexecutor():
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from contextvars import copy_context

    mock_trace_id = create_uuid()

    @observe()
    def execute_task(*args):
        return args

    task_args = [["a", "b"], ["c", "d"]]

    @observe()
    def execute_groups(task_args):
        with ThreadPoolExecutor(3) as executor:
            futures = []

            for task_arg in task_args:
                ctx = copy_context()

                # Using a lambda to capture the current 'task_arg' and context 'ctx' to ensure each task uses its specific arguments and isolated context when executed.
                task = lambda p=task_arg: ctx.run(execute_task, *p)  # noqa

                futures.append(executor.submit(task))

            # Ensure all futures complete
            for future in as_completed(futures):
                future.result()

        return [f.result() for f in futures]

    execute_groups(task_args, langfuse_observation_id=mock_trace_id)

    langfuse_context.flush()

    trace_data = get_api().trace.get(mock_trace_id)

    assert len(trace_data.observations) == 2

    for observation in trace_data.observations:
        assert observation.input["args"] in [["a", "b"], ["c", "d"]]
        assert observation.output in [["a", "b"], ["c", "d"]]

        assert (
            observation.parent_observation_id is None
        )  # Ensure that the observations are not nested


def test_update_trace_io():
    mock_name = "test_update_trace_io"
    mock_trace_id = create_uuid()

    @observe(as_type="generation", name="level_3_to_be_overwritten")
    def level_3_function():
        langfuse_context.update_current_observation(metadata=mock_metadata)
        langfuse_context.update_current_observation(
            metadata=mock_deep_metadata,
            usage={"input": 150, "output": 50, "total": 300},
            model="gpt-3.5-turbo",
            output="mock_output",
        )
        langfuse_context.update_current_observation(
            version="version-1", name="overwritten_level_3"
        )

        langfuse_context.update_current_trace(
            session_id=mock_session_id, name=mock_name, input="nested_input"
        )

        langfuse_context.update_current_trace(
            user_id="user_id",
        )

        return "level_3"

    @observe(name="level_2_manually_set")
    def level_2_function():
        level_3_function()
        langfuse_context.update_current_observation(metadata=mock_metadata)

        return "level_2"

    @observe()
    def level_1_function(*args, **kwargs):
        level_2_function()
        langfuse_context.update_current_trace(output="nested_output")

        return "level_1"

    result = level_1_function(
        *mock_args, **mock_kwargs, langfuse_observation_id=mock_trace_id
    )
    langfuse_context.flush()

    assert result == "level_1"  # Wrapped function returns correctly

    # ID setting for span or trace

    trace_data = get_api().trace.get(mock_trace_id)
    assert (
        len(trace_data.observations) == 2
    )  # Top-most function is trace, so it's not an observations

    assert trace_data.input == "nested_input"
    assert trace_data.output == "nested_output"

    # trace parameters if set anywhere in the call stack
    assert trace_data.session_id == mock_session_id
    assert trace_data.user_id == "user_id"
    assert trace_data.name == mock_name

    # Check correct nesting
    adjacencies = defaultdict(list)
    for o in trace_data.observations:
        adjacencies[o.parent_observation_id or o.trace_id].append(o)

    assert len(adjacencies[mock_trace_id]) == 1  # Trace has only one child
    assert len(adjacencies) == 2  # Only trace and one observation have children

    level_2_observation = adjacencies[mock_trace_id][0]
    level_3_observation = adjacencies[level_2_observation.id][0]

    assert level_2_observation.name == "level_2_manually_set"
    assert level_2_observation.metadata == mock_metadata

    assert level_3_observation.name == "overwritten_level_3"
    assert level_3_observation.metadata == mock_deep_metadata
    assert level_3_observation.type == "GENERATION"
    assert level_3_observation.calculated_total_cost > 0
    assert level_3_observation.output == "mock_output"
    assert level_3_observation.version == "version-1"


def test_parent_trace_id():
    # Create a parent trace
    parent_trace_id = create_uuid()
    observation_id = create_uuid()
    trace_name = "test_parent_trace_id"

    langfuse = langfuse_context.client_instance
    langfuse.trace(id=parent_trace_id, name=trace_name)

    @observe()
    def decorated_function():
        return "decorated_function"

    decorated_function(
        langfuse_parent_trace_id=parent_trace_id, langfuse_observation_id=observation_id
    )

    langfuse_context.flush()

    trace_data = get_api().trace.get(parent_trace_id)

    assert trace_data.id == parent_trace_id
    assert trace_data.name == trace_name

    assert len(trace_data.observations) == 1
    assert trace_data.observations[0].id == observation_id


def test_parent_observation_id():
    parent_trace_id = create_uuid()
    parent_span_id = create_uuid()
    observation_id = create_uuid()
    trace_name = "test_parent_observation_id"
    mock_metadata = {"key": "value"}

    langfuse = langfuse_context.client_instance
    trace = langfuse.trace(id=parent_trace_id, name=trace_name)
    trace.span(id=parent_span_id, name="parent_span")

    @observe()
    def decorated_function():
        langfuse_context.update_current_trace(metadata=mock_metadata)
        langfuse_context.score_current_trace(value=1, name="score_name")

        return "decorated_function"

    decorated_function(
        langfuse_parent_trace_id=parent_trace_id,
        langfuse_parent_observation_id=parent_span_id,
        langfuse_observation_id=observation_id,
    )

    langfuse_context.flush()

    trace_data = get_api().trace.get(parent_trace_id)

    assert trace_data.id == parent_trace_id
    assert trace_data.name == trace_name
    assert trace_data.metadata == mock_metadata
    assert trace_data.scores[0].name == "score_name"
    assert trace_data.scores[0].value == 1

    assert len(trace_data.observations) == 2

    parent_span = next(
        (o for o in trace_data.observations if o.id == parent_span_id), None
    )
    assert parent_span is not None
    assert parent_span.parent_observation_id is None

    execution_span = next(
        (o for o in trace_data.observations if o.id == observation_id), None
    )
    assert execution_span is not None
    assert execution_span.parent_observation_id == parent_span_id


def test_ignore_parent_observation_id_if_parent_trace_id_is_not_set():
    parent_trace_id = create_uuid()
    parent_span_id = create_uuid()
    observation_id = create_uuid()
    trace_name = "test_parent_observation_id"

    langfuse = langfuse_context.client_instance
    trace = langfuse.trace(id=parent_trace_id, name=trace_name)
    trace.span(id=parent_span_id, name="parent_span")

    @observe()
    def decorated_function():
        return "decorated_function"

    decorated_function(
        langfuse_parent_observation_id=parent_span_id,
        langfuse_observation_id=observation_id,
        # No parent trace id set
    )

    langfuse_context.flush()

    trace_data = get_api().trace.get(observation_id)

    assert trace_data.id == observation_id
    assert trace_data.name == "decorated_function"

    assert len(trace_data.observations) == 0


def test_top_level_generation():
    mock_trace_id = create_uuid()
    mock_output = "Hello, World!"

    @observe(as_type="generation")
    def main():
        langfuse_context.update_current_trace(name="updated_name")

        return mock_output

    main(langfuse_observation_id=mock_trace_id)

    langfuse_context.flush()
    sleep(2)

    trace_data = get_api().trace.get(mock_trace_id)
    assert trace_data.name == "updated_name"

    assert len(trace_data.observations) == 1
    assert trace_data.observations[0].name == "main"
    assert trace_data.observations[0].type == "GENERATION"
    assert trace_data.observations[0].output == mock_output


def test_threadpool_executor():
    mock_trace_id = create_uuid()
    mock_parent_observation_id = create_uuid()

    from concurrent.futures import ThreadPoolExecutor, as_completed

    from langfuse.decorators import langfuse_context, observe

    @observe()
    def execute_task(*args):
        return args

    @observe()
    def execute_groups(task_args):
        trace_id = langfuse_context.get_current_trace_id()
        observation_id = langfuse_context.get_current_observation_id()

        with ThreadPoolExecutor(3) as executor:
            futures = [
                executor.submit(
                    execute_task,
                    *task_arg,
                    langfuse_parent_trace_id=trace_id,
                    langfuse_parent_observation_id=observation_id,
                )
                for task_arg in task_args
            ]

            for future in as_completed(futures):
                future.result()

        return [f.result() for f in futures]

    @observe()
    def main():
        task_args = [["a", "b"], ["c", "d"]]

        execute_groups(task_args, langfuse_observation_id=mock_parent_observation_id)

    main(langfuse_observation_id=mock_trace_id)

    langfuse_context.flush()

    trace_data = get_api().trace.get(mock_trace_id)

    assert len(trace_data.observations) == 3

    parent_observation = next(
        (o for o in trace_data.observations if o.id == mock_parent_observation_id), None
    )

    assert parent_observation is not None

    child_observations = [
        o
        for o in trace_data.observations
        if o.parent_observation_id == mock_parent_observation_id
    ]
    assert len(child_observations) == 2


def test_media():
    mock_trace_id = create_uuid()

    with open("static/bitcoin.pdf", "rb") as pdf_file:
        pdf_bytes = pdf_file.read()

    media = LangfuseMedia(content_bytes=pdf_bytes, content_type="application/pdf")

    @observe()
    def main():
        langfuse_context.update_current_trace(
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

    main(langfuse_observation_id=mock_trace_id)

    langfuse_context.flush()

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
