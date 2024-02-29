from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import pytest

from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langfuse.decorators import langfuse
from tests.utils import create_uuid, get_api, get_llama_index_index

mock_metadata = "mock_metadata"
mock_deep_metadata = "mock_deep_metadata"
mock_session_id = "session-id-1"
mock_args = (1, 2, 3)
mock_kwargs = {"a": 1, "b": 2, "c": 3}


def test_nested_observations():
    mock_name = "test_nested_observations"
    mock_trace_id = create_uuid()

    @langfuse.trace
    def level_3_function():
        langfuse.update_current_observation(metadata=mock_metadata)
        langfuse.update_current_observation(metadata=mock_deep_metadata)

        langfuse.set_current_trace_params(session_id=mock_session_id, name=mock_name)

        return "level_3"

    @langfuse.trace
    def level_2_function():
        level_3_function()
        langfuse.update_current_observation(metadata=mock_metadata)

        return "level_2"

    @langfuse.trace
    def level_1_function(*args, **kwargs):
        level_2_function()

        return "level_1"

    result = level_1_function(
        *mock_args, **mock_kwargs, langfuse_observation_id=mock_trace_id
    )
    langfuse.flush()

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


# behavior on exceptions
def test_exception_in_wrapped_function():
    mock_name = "test_exception_in_wrapped_function"
    mock_trace_id = create_uuid()

    @langfuse.trace
    def level_3_function():
        langfuse.update_current_observation(metadata=mock_metadata)
        langfuse.set_current_trace_params(session_id=mock_session_id, name=mock_name)

        raise ValueError("Mock exception")

    @langfuse.trace
    def level_2_function():
        level_3_function()
        langfuse.update_current_observation(metadata=mock_metadata)

        return "level_2"

    @langfuse.trace
    def level_1_function(*args, **kwargs):
        level_2_function()

        return "level_1"

    # Check that the exception is raised
    with pytest.raises(ValueError):
        level_1_function(
            *mock_args, **mock_kwargs, langfuse_observation_id=mock_trace_id
        )

    langfuse.flush()

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
    assert level_3_observation.metadata == mock_metadata
    assert level_3_observation.status_message == "Mock exception"
    assert level_3_observation.level == "ERROR"


# behavior on concurrency
def test_concurrent_decorator_executions():
    mock_name = "test_concurrent_decorator_executions"
    mock_trace_id_1 = create_uuid()
    mock_trace_id_2 = create_uuid()

    @langfuse.trace
    def level_3_function():
        langfuse.update_current_observation(metadata=mock_metadata)
        langfuse.update_current_observation(metadata=mock_deep_metadata)
        langfuse.set_current_trace_params(session_id=mock_session_id, name=mock_name)

        return "level_3"

    @langfuse.trace
    def level_2_function():
        level_3_function()
        langfuse.update_current_observation(metadata=mock_metadata)

        return "level_2"

    @langfuse.trace
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

    langfuse.flush()

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


def test_decorators_llama_index():
    mock_name = "test_decorators_llama_index"
    mock_trace_id = create_uuid()

    @langfuse.trace
    def llama_index_operations(*args, **kwargs):
        callback = langfuse.get_current_llama_index_handler()
        index = get_llama_index_index(callback, force_rebuild=True)

        return index.as_query_engine().query(kwargs["query"])

    @langfuse.trace
    def level_3_function(*args, **kwargs):
        langfuse.update_current_observation(metadata=mock_metadata)
        langfuse.update_current_observation(metadata=mock_deep_metadata)
        langfuse.set_current_trace_params(session_id=mock_session_id, name=mock_name)

        return llama_index_operations(*args, **kwargs)

    @langfuse.trace
    def level_2_function(*args, **kwargs):
        langfuse.update_current_observation(metadata=mock_metadata)

        return level_3_function(*args, **kwargs)

    @langfuse.trace
    def level_1_function(*args, **kwargs):
        return level_2_function(*args, **kwargs)

    level_1_function(
        query="What is the authors ambition?", langfuse_observation_id=mock_trace_id
    )

    langfuse.flush()

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

    @langfuse.trace
    def langchain_operations(*args, **kwargs):
        handler = langfuse.get_current_langchain_handler()
        prompt = ChatPromptTemplate.from_template("tell me a short joke about {topic}")
        model = ChatOpenAI(temperature=0)

        chain = prompt | model

        return chain.invoke(
            {"topic": kwargs["topic"]},
            config={
                "callbacks": [handler],
            },
        )

    @langfuse.trace
    def level_3_function(*args, **kwargs):
        langfuse.update_current_observation(metadata=mock_metadata)
        langfuse.update_current_observation(metadata=mock_deep_metadata)
        langfuse.set_current_trace_params(session_id=mock_session_id, name=mock_name)

        return langchain_operations(*args, **kwargs)

    @langfuse.trace
    def level_2_function(*args, **kwargs):
        langfuse.update_current_observation(metadata=mock_metadata)

        return level_3_function(*args, **kwargs)

    @langfuse.trace
    def level_1_function(*args, **kwargs):
        return level_2_function(*args, **kwargs)

    level_1_function(topic="socks", langfuse_observation_id=mock_trace_id)

    langfuse.flush()

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
