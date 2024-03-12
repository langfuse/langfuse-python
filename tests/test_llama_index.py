from llama_index.core import (
    Settings,
    PromptTemplate,
)
from llama_index.core.callbacks import CallbackManager
from llama_index.llms.openai import OpenAI
from llama_index.llms.anthropic import Anthropic
from llama_index.core.query_pipeline import QueryPipeline

from langfuse.llama_index import LlamaIndexCallbackHandler
from langfuse.client import Langfuse

from tests.utils import create_uuid, get_api, get_llama_index_index


def validate_embedding_generation(generation):
    return all(
        [
            generation.name == "OpenAIEmbedding",
            generation.usage.input == 0,
            generation.usage.output == 0,
            generation.usage.total > 0,  # For embeddings, only total tokens are logged
            bool(generation.input),
            bool(generation.output),
        ]
    )


def validate_llm_generation(generation, model_name="openai_llm"):
    return all(
        [
            generation.name == model_name,
            generation.usage.input > 0,
            generation.usage.output > 0,
            generation.usage.total > 0,
            bool(generation.input),
            bool(generation.output),
        ]
    )


def test_callback_init():
    callback = LlamaIndexCallbackHandler(
        release="something",
        session_id="session-id",
        user_id="user-id",
    )

    assert callback.trace is None

    assert callback.langfuse.release == "something"
    assert callback.session_id == "session-id"
    assert callback.user_id == "user-id"
    assert callback._task_manager is not None


def test_callback_from_index_construction():
    callback = LlamaIndexCallbackHandler()
    get_llama_index_index(callback, force_rebuild=True)

    assert callback.trace is not None

    trace_id = callback.trace.id
    assert trace_id is not None

    callback.flush()
    trace_data = get_api().trace.get(trace_id)
    assert trace_data is not None

    observations = trace_data.observations

    assert any(o.name == "OpenAIEmbedding" for o in observations)

    # Test embedding generation
    generations = sorted(
        [o for o in trace_data.observations if o.type == "GENERATION"],
        key=lambda o: o.start_time,
    )
    assert len(generations) == 1  # Only one generation event for all embedded chunks

    generation = generations[0]
    assert validate_embedding_generation(generation)


def test_callback_from_query_engine():
    callback = LlamaIndexCallbackHandler()
    index = get_llama_index_index(callback)
    index.as_query_engine().query(
        "What did the speaker achieve in the past twelve months?"
    )

    callback.flush()
    trace_data = get_api().trace.get(callback.trace.id)

    # Test LLM generation
    generations = sorted(
        [o for o in trace_data.observations if o.type == "GENERATION"],
        key=lambda o: o.start_time,
    )
    assert (
        len(generations) == 2
    )  # One generation event for embedding call of query, one for LLM call

    embedding_generation, llm_generation = generations
    assert validate_embedding_generation(embedding_generation)
    assert validate_llm_generation(llm_generation)


def test_callback_from_chat_engine():
    callback = LlamaIndexCallbackHandler()
    index = get_llama_index_index(callback)
    index.as_chat_engine().chat(
        "What did the speaker achieve in the past twelve months?"
    )

    callback.flush()
    trace_data = get_api().trace.get(callback.trace.id)

    # Test LLM generation
    generations = sorted(
        [o for o in trace_data.observations if o.type == "GENERATION"],
        key=lambda o: o.start_time,
    )
    embedding_generations = [g for g in generations if g.name == "OpenAIEmbedding"]
    llm_generations = [g for g in generations if g.name == "openai_llm"]

    assert len(embedding_generations) == 1
    assert len(llm_generations) > 0

    assert all([validate_embedding_generation(g) for g in embedding_generations])
    assert all([validate_llm_generation(g) for g in llm_generations])


def test_callback_from_query_engine_stream():
    callback = LlamaIndexCallbackHandler()
    index = get_llama_index_index(callback)
    stream_response = index.as_query_engine(streaming=True).query(
        "What did the speaker achieve in the past twelve months?"
    )

    for token in stream_response.response_gen:
        print(token, end="")

    callback.flush()
    trace_data = get_api().trace.get(callback.trace.id)

    # Test LLM generation
    generations = sorted(
        [o for o in trace_data.observations if o.type == "GENERATION"],
        key=lambda o: o.start_time,
    )
    embedding_generations = [g for g in generations if g.name == "OpenAIEmbedding"]
    llm_generations = [g for g in generations if g.name == "openai_llm"]

    assert len(embedding_generations) == 1
    assert len(llm_generations) > 0

    assert all([validate_embedding_generation(g) for g in embedding_generations])


def test_callback_from_chat_stream():
    callback = LlamaIndexCallbackHandler()
    index = get_llama_index_index(callback)
    stream_response = index.as_chat_engine().stream_chat(
        "What did the speaker achieve in the past twelve months?"
    )

    for token in stream_response.response_gen:
        print(token, end="")

    callback.flush()
    trace_data = get_api().trace.get(callback.trace.id)

    # Test LLM generation
    generations = sorted(
        [o for o in trace_data.observations if o.type == "GENERATION"],
        key=lambda o: o.start_time,
    )
    embedding_generations = [g for g in generations if g.name == "OpenAIEmbedding"]
    llm_generations = [g for g in generations if g.name == "openai_llm"]

    assert len(embedding_generations) == 1
    assert len(llm_generations) > 0

    assert all([validate_embedding_generation(g) for g in embedding_generations])
    assert all([validate_llm_generation(g) for g in llm_generations])


def test_callback_from_query_pipeline():
    callback = LlamaIndexCallbackHandler()
    Settings.callback_manager = CallbackManager([callback])

    prompt_str = "Please generate related movies to {movie_name}"
    prompt_tmpl = PromptTemplate(prompt_str)
    models = [
        ("openai_llm", OpenAI(model="gpt-3.5-turbo")),
        ("Anthropic_LLM", Anthropic()),
    ]

    for model_name, llm in models:
        pipeline = QueryPipeline(
            chain=[prompt_tmpl, llm],
            verbose=True,
            callback_manager=Settings.callback_manager,
        )
        pipeline.run(movie_name="The Matrix")

        callback.flush()
        trace_data = get_api().trace.get(callback.trace.id)
        observations = trace_data.observations
        llm_generations = list(
            filter(
                lambda o: o.type == "GENERATION" and o.name == model_name,
                observations,
            )
        )

        assert len(llm_generations) == 1
        assert validate_llm_generation(llm_generations[0], model_name=model_name)


def test_callback_with_root_trace():
    callback = LlamaIndexCallbackHandler()
    index = get_llama_index_index(callback)

    langfuse = Langfuse(debug=False)
    trace_id = create_uuid()
    root_trace = langfuse.trace(id=trace_id, name=trace_id)

    callback.set_root(root_trace)
    index.as_query_engine().query(
        "What did the speaker achieve in the past twelve months?"
    )

    assert callback.get_trace_id() == trace_id

    callback.flush()
    trace_data = get_api().trace.get(callback.trace.id)
    assert trace_data is not None

    # Test LLM generation
    generations = sorted(
        [o for o in trace_data.observations if o.type == "GENERATION"],
        key=lambda o: o.start_time,
    )
    assert (
        len(generations) == 2
    )  # One generation event for embedding call of query, one for LLM call

    embedding_generation, llm_generation = generations
    assert validate_embedding_generation(embedding_generation)
    assert validate_llm_generation(llm_generation)

    # Test that further observations are also appended to the root trace
    index.as_query_engine().query("How did the speaker achieve those goals?")

    callback.flush()
    trace_data = get_api().trace.get(callback.trace.id)
    generations = sorted(
        [o for o in trace_data.observations if o.type == "GENERATION"],
        key=lambda o: o.start_time,
    )
    assert len(generations) == 4  # Two more generations are appended

    second_embedding_generation, second_llm_generation = generations[-2:]
    assert validate_embedding_generation(second_embedding_generation)
    assert validate_llm_generation(second_llm_generation)

    # Reset the root trace
    callback.set_root(None)

    index.as_query_engine().query("How did the speaker achieve those goals?")
    new_trace_id = callback.get_trace_id()
    assert callback.get_trace_id() != trace_id

    callback.flush()

    trace_data = get_api().trace.get(new_trace_id)
    assert trace_data is not None

    # Test LLM generation
    generations = sorted(
        [o for o in trace_data.observations if o.type == "GENERATION"],
        key=lambda o: o.start_time,
    )
    assert (
        len(generations) == 2
    )  # One generation event for embedding call of query, one for LLM call

    embedding_generation, llm_generation = generations
    assert validate_embedding_generation(embedding_generation)
    assert validate_llm_generation(llm_generation)


def test_callback_with_root_span():
    callback = LlamaIndexCallbackHandler()
    index = get_llama_index_index(callback)

    langfuse = Langfuse(debug=False)
    trace_id = create_uuid()
    span_id = create_uuid()
    trace = langfuse.trace(id=trace_id, name=trace_id)
    span = trace.span(id=span_id, name=span_id)

    callback.set_root(span)
    index.as_query_engine().query(
        "What did the speaker achieve in the past twelve months?"
    )

    assert callback.get_trace_id() == trace_id
    callback.flush()
    trace_data = get_api().trace.get(trace_id)

    assert trace_data is not None
    assert any([o.id == span_id for o in trace_data.observations])

    # Test LLM generation
    generations = sorted(
        [o for o in trace_data.observations if o.type == "GENERATION"],
        key=lambda o: o.start_time,
    )
    assert (
        len(generations) == 2
    )  # One generation event for embedding call of query, one for LLM call

    embedding_generation, llm_generation = generations
    assert validate_embedding_generation(embedding_generation)
    assert validate_llm_generation(llm_generation)

    # Test that more observations are also appended to the root span
    index.as_query_engine().query("How did the speaker achieve those goals?")

    callback.flush()
    trace_data = get_api().trace.get(trace_id)
    generations = sorted(
        [o for o in trace_data.observations if o.type == "GENERATION"],
        key=lambda o: o.start_time,
    )
    assert len(generations) == 4  # Two more generations are appended

    second_embedding_generation, second_llm_generation = generations[-2:]
    assert validate_embedding_generation(second_embedding_generation)
    assert validate_llm_generation(second_llm_generation)

    # Reset the root span
    callback.set_root(None)
    index.as_query_engine().query("How did the speaker achieve those goals?")

    new_trace_id = callback.get_trace_id()
    assert new_trace_id != trace_id
    callback.flush()

    trace_data = get_api().trace.get(new_trace_id)

    assert trace_data is not None
    assert not any([o.id == span_id for o in trace_data.observations])

    # Test LLM generation
    generations = sorted(
        [o for o in trace_data.observations if o.type == "GENERATION"],
        key=lambda o: o.start_time,
    )
    assert (
        len(generations) == 2
    )  # One generation event for embedding call of query, one for LLM call

    embedding_generation, llm_generation = generations
    assert validate_embedding_generation(embedding_generation)
    assert validate_llm_generation(llm_generation)


def test_callback_with_custom_trace_metadata():
    initial_name = "initial-name"
    initial_user_id = "initial-user-id"
    initial_session_id = "initial-session-id"
    initial_tags = ["initial_value1", "initial_value2"]

    callback = LlamaIndexCallbackHandler(
        trace_name=initial_name,
        user_id=initial_user_id,
        session_id=initial_session_id,
        tags=initial_tags,
    )

    index = get_llama_index_index(callback)
    index.as_query_engine().query(
        "What did the speaker achieve in the past twelve months?"
    )

    callback.flush()
    trace_data = get_api().trace.get(callback.trace.id)

    assert trace_data.name == initial_name
    assert trace_data.user_id == initial_user_id
    assert trace_data.session_id == initial_session_id
    assert trace_data.tags == initial_tags

    # Update trace metadata on existing handler
    updated_name = "updated-name"
    updated_user_id = "updated-user-id"
    updated_session_id = "updated-session-id"
    updated_tags = ["updated_value1", "updated_value2"]

    callback.set_trace_params(
        name=updated_name,
        user_id=updated_user_id,
        session_id=updated_session_id,
        tags=updated_tags,
    )

    index.as_query_engine().query(
        "What did the speaker achieve in the past twelve months?"
    )
    callback.flush()
    trace_data = get_api().trace.get(callback.trace.id)

    assert trace_data.name == updated_name
    assert trace_data.user_id == updated_user_id
    assert trace_data.session_id == updated_session_id
    assert trace_data.tags == updated_tags
