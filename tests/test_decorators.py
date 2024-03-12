import asyncio
from contextvars import ContextVar
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import pytest

from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langfuse.openai import AsyncOpenAI
from langfuse.decorators import langfuse_context, observe
from tests.utils import create_uuid, get_api, get_llama_index_index
from typing import Optional

mock_metadata = "mock_metadata"
mock_deep_metadata = "mock_deep_metadata"
mock_session_id = "session-id-1"
mock_args = (1, 2, 3)
mock_kwargs = {"a": 1, "b": 2, "c": 3}


def test_nested_observations():
    mock_name = "test_nested_observations"
    mock_trace_id = create_uuid()

    @observe(as_type="generation")
    def level_3_function():
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
    assert level_3_observation.metadata == mock_deep_metadata
    assert level_3_observation.type == "GENERATION"
    assert level_3_observation.calculated_total_cost > 0
    assert level_3_observation.output == "mock_output"


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
        level_2_observation.metadata is None
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
        langfuse_context.update_current_trace(
            session_id=mock_session_id, name=mock_name
        )

        return "level_3"

    @observe()
    def level_2_function():
        level_3_function()
        langfuse_context.update_current_observation(metadata=mock_metadata)

        return "level_2"

    @observe()
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

    print("mock_id_1", mock_trace_id_1)
    print("mock_id_2", mock_trace_id_2)

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


def test_scoring_observations():
    mock_name = "test_scoring_observations"
    mock_trace_id = create_uuid()

    @observe(as_type="generation")
    def level_3_function():
        langfuse_context.score_current_observation(
            name="test-observation-score", value=1
        )
        langfuse_context.score_current_trace(name="test-trace-score", value=2)
        return "level_3"

    @observe()
    def level_2_function():
        return level_3_function()

    @observe()
    def level_1_function(*args, **kwargs):
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

    assert len(scores) == 2

    trace_score = [
        s for s in scores if s.trace_id == mock_trace_id and s.observation_id is None
    ][0]
    observation_score = [s for s in scores if s.observation_id is not None][0]

    print(trace_score)
    print(observation_score)

    assert trace_score.name == "test-trace-score"
    assert trace_score.value == 2

    assert observation_score.name == "test-observation-score"
    assert observation_score.value == 1


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


def test_decorated_instance_methods():
    mock_name = "test_decorated_instance_methods"
    mock_trace_id = create_uuid()

    class TestClass:
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
    assert "2" in generation.output
