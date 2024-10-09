from typing import Optional
from langfuse.client import Langfuse
from langfuse.llama_index import LlamaIndexInstrumentor
from llama_index.llms import openai, anthropic
from llama_index.core.prompts import PromptTemplate
from llama_index.core.query_pipeline import QueryPipeline

from tests.utils import get_api, get_llama_index_index, create_uuid


def is_embedding_generation_name(name: Optional[str]) -> bool:
    return name is not None and any(
        embedding_class in name
        for embedding_class in ("OpenAIEmbedding.", "BaseEmbedding")
    )


def is_llm_generation_name(name: Optional[str], model_name: str = "OpenAI") -> bool:
    return name is not None and f"{model_name}." in name


def validate_embedding_generation(generation):
    return all(
        [
            is_embedding_generation_name(generation.name),
            # generation.usage.input == 0,
            # generation.usage.output == 0,
            # generation.usage.total > 0,  # For embeddings, only total tokens are logged
            bool(generation.input),
            bool(generation.output),
        ]
    )


def validate_llm_generation(generation, model_name="OpenAI"):
    return all(
        [
            is_llm_generation_name(generation.name, model_name),
            generation.usage.input > 0,
            # generation.usage.output > 0, # streamed generations currently broken with no output
            generation.usage.total > 0,
            bool(generation.input),
            # bool(generation.output), # streamed generations currently broken with no output
        ]
    )


def test_instrumentor_from_index_construction():
    trace_id = create_uuid()
    instrumentor = LlamaIndexInstrumentor()
    instrumentor.start()

    with instrumentor.observe(trace_id=trace_id):
        get_llama_index_index(None, force_rebuild=True)

    instrumentor.flush()

    trace_data = get_api().trace.get(trace_id)
    assert trace_data is not None

    observations = trace_data.observations
    assert any(
        is_embedding_generation_name(o.name) for o in observations if o.name is not None
    )

    generations = sorted(
        [o for o in trace_data.observations if o.type == "GENERATION"],
        key=lambda o: o.start_time,
    )
    assert len(generations) == 1  # Only one generation event for all embedded chunks

    generation = generations[0]
    assert validate_embedding_generation(generation)


def test_instrumentor_from_query_engine():
    trace_id = create_uuid()
    instrumentor = LlamaIndexInstrumentor()
    instrumentor.start()

    with instrumentor.observe(
        trace_id=trace_id,
        user_id="test_user_id",
        session_id="test_session_id",
        version="test_version",
        release="test_release",
        metadata={"test_metadata": "test_metadata"},
        tags=["test_tag"],
        public=True,
    ):
        index = get_llama_index_index(None, force_rebuild=True)
        index.as_query_engine().query(
            "What did the speaker achieve in the past twelve months?"
        )

    instrumentor.flush()

    trace_data = get_api().trace.get(trace_id)

    # Test LLM generation
    generations = sorted(
        [o for o in trace_data.observations if o.type == "GENERATION"],
        key=lambda o: o.start_time,
    )
    assert (
        len(generations) == 3
    )  # One generation event for embedding call of query, one for LLM call

    embedding_generations = [
        g for g in generations if is_embedding_generation_name(g.name)
    ]
    llm_generations = [g for g in generations if is_llm_generation_name(g.name)]

    assert all([validate_embedding_generation(g) for g in embedding_generations])
    assert all([validate_llm_generation(g) for g in llm_generations])


def test_instrumentor_from_chat_engine():
    trace_id = create_uuid()
    instrumentor = LlamaIndexInstrumentor()
    instrumentor.start()

    with instrumentor.observe(trace_id=trace_id):
        index = get_llama_index_index(None)
        index.as_chat_engine().chat(
            "What did the speaker achieve in the past twelve months?"
        )

    instrumentor.flush()
    trace_data = get_api().trace.get(trace_id)

    # Test LLM generation
    generations = sorted(
        [o for o in trace_data.observations if o.type == "GENERATION"],
        key=lambda o: o.start_time,
    )

    embedding_generations = [
        g for g in generations if is_embedding_generation_name(g.name)
    ]
    llm_generations = [g for g in generations if is_llm_generation_name(g.name)]

    assert len(embedding_generations) == 1
    assert len(llm_generations) > 0

    assert all([validate_embedding_generation(g) for g in embedding_generations])
    assert all([validate_llm_generation(g) for g in llm_generations])


def test_instrumentor_from_query_engine_stream():
    trace_id = create_uuid()

    instrumentor = LlamaIndexInstrumentor()
    instrumentor.start()

    with instrumentor.observe(trace_id=trace_id):
        index = get_llama_index_index(None)
        stream_response = index.as_query_engine(streaming=True).query(
            "What did the speaker achieve in the past twelve months?"
        )

        for token in stream_response.response_gen:
            print(token, end="")

    instrumentor.flush()
    trace_data = get_api().trace.get(trace_id)

    # Test LLM generation
    generations = sorted(
        [o for o in trace_data.observations if o.type == "GENERATION"],
        key=lambda o: o.start_time,
    )
    embedding_generations = [
        g for g in generations if is_embedding_generation_name(g.name)
    ]
    llm_generations = [g for g in generations if is_llm_generation_name(g.name)]

    assert len(embedding_generations) == 1
    assert len(llm_generations) > 0

    assert all([validate_embedding_generation(g) for g in embedding_generations])


def test_instrumentor_from_chat_stream():
    trace_id = create_uuid()
    instrumentor = LlamaIndexInstrumentor()

    with instrumentor.observe(trace_id=trace_id):
        index = get_llama_index_index(None)
        stream_response = index.as_chat_engine().stream_chat(
            "What did the speaker achieve in the past twelve months?"
        )

        for token in stream_response.response_gen:
            print(token, end="")

    instrumentor.flush()
    trace_data = get_api().trace.get(trace_id)

    # Test LLM generation
    generations = sorted(
        [o for o in trace_data.observations if o.type == "GENERATION"],
        key=lambda o: o.start_time,
    )
    embedding_generations = [
        g for g in generations if is_embedding_generation_name(g.name)
    ]
    llm_generations = [g for g in generations if is_llm_generation_name(g.name)]

    assert len(embedding_generations) == 1
    assert len(llm_generations) > 0

    assert all([validate_embedding_generation(g) for g in embedding_generations])
    assert all([validate_llm_generation(g) for g in llm_generations])


def test_instrumentor_from_query_pipeline():
    instrumentor = LlamaIndexInstrumentor()

    # index = get_llama_index_index(None)

    prompt_str = "Please generate related movies to {movie_name}"
    prompt_tmpl = PromptTemplate(prompt_str)
    models = [
        ("OpenAI", openai.OpenAI(model="gpt-3.5-turbo")),
        ("Anthropic", anthropic.Anthropic()),
    ]

    for model_name, llm in models:
        trace_id = create_uuid()
        pipeline = QueryPipeline(
            chain=[prompt_tmpl, llm],
            verbose=True,
        )

        with instrumentor.observe(trace_id=trace_id):
            pipeline.run(movie_name="The Matrix")

        instrumentor.flush()

        trace_data = get_api().trace.get(trace_id)
        observations = trace_data.observations
        llm_generations = [
            o
            for o in observations
            if is_llm_generation_name(o.name, model_name) and o.type == "GENERATION"
        ]

        assert len(llm_generations) == 1
        assert validate_llm_generation(llm_generations[0], model_name=model_name)


def test_instrumentor_with_root_trace():
    instrumentor = LlamaIndexInstrumentor()

    index = get_llama_index_index(None)

    langfuse = Langfuse()

    trace_id = create_uuid()
    langfuse.trace(id=trace_id, name=trace_id)

    with instrumentor.observe(trace_id=trace_id):
        index.as_query_engine().query(
            "What did the speaker achieve in the past twelve months?"
        )

    instrumentor.flush()
    trace_data = get_api().trace.get(trace_id)

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


def test_instrumentor_with_root_span():
    instrumentor = LlamaIndexInstrumentor()
    index = get_llama_index_index(None)

    langfuse = Langfuse(debug=False)
    trace_id = create_uuid()
    span_id = create_uuid()
    trace = langfuse.trace(id=trace_id, name=trace_id)
    trace.span(id=span_id, name=span_id)

    with instrumentor.observe(trace_id=trace_id, parent_observation_id=span_id):
        index.as_query_engine().query(
            "What did the speaker achieve in the past twelve months?"
        )

    instrumentor.flush()
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


def test_instrumentor_with_custom_trace_metadata():
    initial_name = "initial-name"
    initial_user_id = "initial-user-id"
    initial_session_id = "initial-session-id"
    initial_tags = ["initial_value1", "initial_value2"]

    instrumentor = LlamaIndexInstrumentor()

    trace = Langfuse().trace(
        name=initial_name,
        user_id=initial_user_id,
        session_id=initial_session_id,
        tags=initial_tags,
    )

    with instrumentor.observe(trace_id=trace.id, update_parent=False):
        index = get_llama_index_index(None)
        index.as_query_engine().query(
            "What did the speaker achieve in the past twelve months?"
        )

    instrumentor.flush()
    trace_data = get_api().trace.get(trace.id)

    assert trace_data.name == initial_name
    assert trace_data.user_id == initial_user_id
    assert trace_data.session_id == initial_session_id
    assert trace_data.tags == initial_tags
