import types

import pytest
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.schema import StrOutputParser
from langchain_openai import ChatOpenAI, OpenAI

from langfuse import Langfuse
from langfuse.langchain import CallbackHandler
from tests.utils import get_api

from .utils import create_uuid


def _is_streaming_response(response):
    return isinstance(response, types.GeneratorType) or isinstance(
        response, types.AsyncGeneratorType
    )


# Streaming in chat models
@pytest.mark.skip(
    reason="This test suite is not properly isolated and fails flakily. TODO: Investigate why"
)
@pytest.mark.parametrize("model_name", ["gpt-3.5-turbo", "gpt-4"])
def test_stream_chat_models(model_name):
    name = f"test_stream_chat_models-{create_uuid()}"
    tags = ["Hello", "world"]
    model = ChatOpenAI(
        streaming=True, max_completion_tokens=300, tags=tags, model=model_name
    )
    handler = CallbackHandler()

    langfuse_client = handler.client
    trace_id = Langfuse.create_trace_id()
    with langfuse_client.start_as_current_span(
        name=name, trace_context={"trace_id": trace_id}
    ):
        res = model.stream(
            [{"role": "user", "content": "return the exact phrase - This is a test!"}],
            config={"callbacks": [handler]},
        )
        response_str = []
        assert _is_streaming_response(res)
        for chunk in res:
            response_str.append(chunk.content)

    langfuse_client.flush()
    assert handler.runs == {}
    api = get_api()
    trace = api.trace.get(trace_id)
    generationList = list(filter(lambda o: o.type == "GENERATION", trace.observations))
    assert len(generationList) != 0

    generation = generationList[0]

    assert len(response_str) > 1  # To check there are more than one chunk.
    assert len(trace.observations) == 2
    assert trace.name == name
    assert model_name in generation.model
    assert generation.input is not None
    assert generation.output is not None
    assert generation.model_parameters.get("max_completion_tokens") is not None
    assert generation.model_parameters.get("temperature") is not None
    assert generation.metadata["tags"] == tags
    assert generation.usage.output is not None
    assert generation.usage.total is not None
    assert generation.output["content"] is not None
    assert generation.output["role"] is not None
    assert generation.input_price is not None
    assert generation.output_price is not None
    assert generation.calculated_input_cost is not None
    assert generation.calculated_output_cost is not None
    assert generation.calculated_total_cost is not None
    assert generation.latency is not None


# Streaming in completions models
@pytest.mark.skip(
    reason="This test suite is not properly isolated and fails flakily. TODO: Investigate why"
)
@pytest.mark.parametrize("model_name", ["gpt-3.5-turbo-instruct"])
def test_stream_completions_models(model_name):
    name = f"test_stream_completions_models-{create_uuid()}"
    tags = ["Hello", "world"]
    model = OpenAI(streaming=True, max_tokens=300, tags=tags, model=model_name)
    handler = CallbackHandler()

    langfuse_client = handler.client
    trace_id = Langfuse.create_trace_id()
    with langfuse_client.start_as_current_span(
        name=name, trace_context={"trace_id": trace_id}
    ):
        res = model.stream(
            "return the exact phrase - This is a test!",
            config={"callbacks": [handler]},
        )
        response_str = []
        assert _is_streaming_response(res)
        for chunk in res:
            response_str.append(chunk)

    langfuse_client.flush()
    assert handler.runs == {}
    api = get_api()
    trace = api.trace.get(trace_id)
    generationList = list(filter(lambda o: o.type == "GENERATION", trace.observations))
    assert len(generationList) != 0

    generation = generationList[0]

    assert len(response_str) > 1  # To check there are more than one chunk.
    assert len(trace.observations) == 2
    assert trace.name == name
    assert model_name in generation.model
    assert generation.input is not None
    assert generation.output is not None
    assert generation.model_parameters.get("max_tokens") is not None
    assert generation.model_parameters.get("temperature") is not None
    assert generation.metadata["tags"] == tags
    assert generation.usage.output is not None
    assert generation.usage.total is not None
    assert generation.output is not None
    assert generation.input_price is not None
    assert generation.output_price is not None
    assert generation.calculated_input_cost is not None
    assert generation.calculated_output_cost is not None
    assert generation.calculated_total_cost is not None
    assert generation.latency is not None


# Invoke in chat models
@pytest.mark.skip(
    reason="This test suite is not properly isolated and fails flakily. TODO: Investigate why"
)
@pytest.mark.parametrize("model_name", ["gpt-3.5-turbo", "gpt-4"])
def test_invoke_chat_models(model_name):
    name = f"test_invoke_chat_models-{create_uuid()}"
    tags = ["Hello", "world"]
    model = ChatOpenAI(max_completion_tokens=300, tags=tags, model=model_name)
    handler = CallbackHandler()

    langfuse_client = handler.client
    trace_id = Langfuse.create_trace_id()
    with langfuse_client.start_as_current_span(
        name=name, trace_context={"trace_id": trace_id}
    ):
        _ = model.invoke(
            [{"role": "user", "content": "return the exact phrase - This is a test!"}],
            config={"callbacks": [handler]},
        )

    langfuse_client.flush()
    assert handler.runs == {}
    api = get_api()
    trace = api.trace.get(trace_id)
    generationList = list(filter(lambda o: o.type == "GENERATION", trace.observations))
    assert len(generationList) != 0

    generation = generationList[0]

    assert len(trace.observations) == 2
    assert trace.name == name
    assert model_name in generation.model
    assert generation.input is not None
    assert generation.output is not None
    assert generation.model_parameters.get("max_completion_tokens") is not None
    assert generation.model_parameters.get("temperature") is not None
    assert generation.metadata["tags"] == tags
    assert generation.usage.output is not None
    assert generation.usage.total is not None
    assert generation.output["content"] is not None
    assert generation.output["role"] is not None
    assert generation.input_price is not None
    assert generation.output_price is not None
    assert generation.calculated_input_cost is not None
    assert generation.calculated_output_cost is not None
    assert generation.calculated_total_cost is not None
    assert generation.latency is not None


# Invoke in completions models
@pytest.mark.skip(
    reason="This test suite is not properly isolated and fails flakily. TODO: Investigate why"
)
@pytest.mark.parametrize("model_name", ["gpt-3.5-turbo-instruct"])
def test_invoke_in_completions_models(model_name):
    name = f"test_invoke_in_completions_models-{create_uuid()}"
    tags = ["Hello", "world"]
    model = OpenAI(max_tokens=300, tags=tags, model=model_name)
    handler = CallbackHandler()

    langfuse_client = handler.client
    trace_id = Langfuse.create_trace_id()
    with langfuse_client.start_as_current_span(
        name=name, trace_context={"trace_id": trace_id}
    ):
        test_phrase = "This is a test!"
        _ = model.invoke(
            f"return the exact phrase - {test_phrase}",
            config={"callbacks": [handler]},
        )

    langfuse_client.flush()
    assert handler.runs == {}
    api = get_api()
    trace = api.trace.get(trace_id)
    generationList = list(filter(lambda o: o.type == "GENERATION", trace.observations))
    assert len(generationList) != 0

    generation = generationList[0]

    assert len(trace.observations) == 2
    assert trace.name == name
    assert model_name in generation.model
    assert generation.input is not None
    assert generation.output is not None
    assert generation.model_parameters.get("max_tokens") is not None
    assert generation.model_parameters.get("temperature") is not None
    assert generation.metadata["tags"] == tags
    assert generation.usage.output is not None
    assert generation.usage.total is not None
    assert test_phrase in generation.output
    assert generation.input_price is not None
    assert generation.output_price is not None
    assert generation.calculated_input_cost is not None
    assert generation.calculated_output_cost is not None
    assert generation.calculated_total_cost is not None
    assert generation.latency is not None


@pytest.mark.skip(
    reason="This test suite is not properly isolated and fails flakily. TODO: Investigate why"
)
@pytest.mark.parametrize("model_name", ["gpt-3.5-turbo-instruct"])
def test_batch_in_completions_models(model_name):
    name = f"test_batch_in_completions_models-{create_uuid()}"
    tags = ["Hello", "world"]
    model = OpenAI(max_tokens=300, tags=tags, model=model_name)
    handler = CallbackHandler()

    langfuse_client = handler.client
    trace_id = Langfuse.create_trace_id()
    with langfuse_client.start_as_current_span(
        name=name, trace_context={"trace_id": trace_id}
    ):
        input1 = "Who is the first president of America ?"
        input2 = "Who is the first president of Ireland ?"
        _ = model.batch(
            [input1, input2],
            config={"callbacks": [handler]},
        )

    langfuse_client.flush()
    assert handler.runs == {}
    api = get_api()
    trace = api.trace.get(trace_id)
    generationList = list(filter(lambda o: o.type == "GENERATION", trace.observations))
    assert len(generationList) != 0

    generation = generationList[0]

    assert len(trace.observations) == 3
    assert trace.name == name
    assert model_name in generation.model
    assert generation.input is not None
    assert generation.output is not None
    assert generation.model_parameters.get("max_tokens") is not None
    assert generation.model_parameters.get("temperature") is not None
    assert generation.metadata["tags"] == tags
    assert generation.usage.output is not None
    assert generation.usage.total is not None
    assert generation.input_price is not None
    assert generation.output_price is not None
    assert generation.calculated_input_cost is not None
    assert generation.calculated_output_cost is not None
    assert generation.calculated_total_cost is not None
    assert generation.latency is not None


@pytest.mark.skip(
    reason="This test suite is not properly isolated and fails flakily. TODO: Investigate why"
)
@pytest.mark.parametrize("model_name", ["gpt-3.5-turbo", "gpt-4"])
def test_batch_in_chat_models(model_name):
    name = f"test_batch_in_chat_models-{create_uuid()}"
    tags = ["Hello", "world"]
    model = ChatOpenAI(max_completion_tokens=300, tags=tags, model=model_name)
    handler = CallbackHandler()

    langfuse_client = handler.client
    trace_id = Langfuse.create_trace_id()
    with langfuse_client.start_as_current_span(
        name=name, trace_context={"trace_id": trace_id}
    ):
        input1 = "Who is the first president of America ?"
        input2 = "Who is the first president of Ireland ?"
        _ = model.batch(
            [input1, input2],
            config={"callbacks": [handler]},
        )

    langfuse_client.flush()
    assert handler.runs == {}
    api = get_api()
    trace = api.trace.get(trace_id)
    generationList = list(filter(lambda o: o.type == "GENERATION", trace.observations))
    assert len(generationList) != 0

    assert len(trace.observations) == 3
    assert trace.name == name
    for generation in generationList:
        assert model_name in generation.model
        assert generation.input is not None
        assert generation.output is not None
        assert generation.model_parameters.get("max_completion_tokens") is not None
        assert generation.model_parameters.get("temperature") is not None
        assert generation.metadata["tags"] == tags
        assert generation.usage.output is not None
        assert generation.usage.total is not None
        assert generation.input_price is not None
        assert generation.output_price is not None
        assert generation.calculated_input_cost is not None
        assert generation.calculated_output_cost is not None
        assert generation.calculated_total_cost is not None
        assert generation.latency is not None


# Async stream in chat models
@pytest.mark.skip(
    reason="This test suite is not properly isolated and fails flakily. TODO: Investigate why"
)
@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", ["gpt-3.5-turbo", "gpt-4"])
async def test_astream_chat_models(model_name):
    name = f"test_astream_chat_models-{create_uuid()}"
    tags = ["Hello", "world"]
    model = ChatOpenAI(
        streaming=True, max_completion_tokens=300, tags=tags, model=model_name
    )
    handler = CallbackHandler()

    langfuse_client = handler.client
    trace_id = Langfuse.create_trace_id()
    with langfuse_client.start_as_current_span(
        name=name, trace_context={"trace_id": trace_id}
    ):
        res = model.astream(
            [{"role": "user", "content": "Who was the first American president "}],
            config={"callbacks": [handler]},
        )
        response_str = []
        assert _is_streaming_response(res)
        async for chunk in res:
            response_str.append(chunk.content)

    langfuse_client.flush()
    assert handler.runs == {}
    api = get_api()
    trace = api.trace.get(trace_id)
    generationList = list(filter(lambda o: o.type == "GENERATION", trace.observations))
    assert len(generationList) != 0

    generation = generationList[0]

    assert len(response_str) > 1  # To check there are more than one chunk.
    assert len(trace.observations) == 2
    assert model_name in generation.model
    assert generation.input is not None
    assert generation.output is not None
    assert generation.model_parameters.get("max_completion_tokens") is not None
    assert generation.model_parameters.get("temperature") is not None
    assert generation.metadata["tags"] == tags
    assert generation.usage.output is not None
    assert generation.usage.total is not None
    assert generation.output["content"] is not None
    assert generation.output["role"] is not None
    assert generation.input_price is not None
    assert generation.output_price is not None
    assert generation.calculated_input_cost is not None
    assert generation.calculated_output_cost is not None
    assert generation.calculated_total_cost is not None
    assert generation.latency is not None


# Async stream in completions model
@pytest.mark.skip(
    reason="This test suite is not properly isolated and fails flakily. TODO: Investigate why"
)
@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", ["gpt-3.5-turbo-instruct"])
async def test_astream_completions_models(model_name):
    name = f"test_astream_completions_models-{create_uuid()}"
    tags = ["Hello", "world"]
    model = OpenAI(streaming=True, max_tokens=300, tags=tags, model=model_name)
    handler = CallbackHandler()

    langfuse_client = handler.client

    trace_id = Langfuse.create_trace_id()
    with langfuse_client.start_as_current_span(
        name=name, trace_context={"trace_id": trace_id}
    ):
        test_phrase = "This is a test!"
        res = model.astream(
            f"return the exact phrase - {test_phrase}",
            config={"callbacks": [handler]},
        )
        response_str = []
        assert _is_streaming_response(res)
        async for chunk in res:
            response_str.append(chunk)

    langfuse_client.flush()
    assert handler.runs == {}
    api = get_api()
    trace = api.trace.get(trace_id)
    generationList = list(filter(lambda o: o.type == "GENERATION", trace.observations))
    assert len(generationList) != 0

    generation = generationList[0]

    assert len(response_str) > 1  # To check there are more than one chunk.
    assert len(trace.observations) == 2
    assert test_phrase in "".join(response_str)
    assert model_name in generation.model
    assert generation.input is not None
    assert generation.output is not None
    assert generation.model_parameters.get("max_tokens") is not None
    assert generation.model_parameters.get("temperature") is not None
    assert generation.metadata["tags"] == tags
    assert generation.usage.output is not None
    assert generation.usage.total is not None
    assert test_phrase in generation.output
    assert generation.input_price is not None
    assert generation.output_price is not None
    assert generation.calculated_input_cost is not None
    assert generation.calculated_output_cost is not None
    assert generation.calculated_total_cost is not None
    assert generation.latency is not None


# Async invoke in chat models
@pytest.mark.skip(
    reason="This test suite is not properly isolated and fails flakily. TODO: Investigate why"
)
@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", ["gpt-3.5-turbo", "gpt-4"])
async def test_ainvoke_chat_models(model_name):
    name = f"test_ainvoke_chat_models-{create_uuid()}"
    tags = ["Hello", "world"]
    model = ChatOpenAI(max_completion_tokens=300, tags=tags, model=model_name)
    handler = CallbackHandler()

    langfuse_client = handler.client
    trace_id = Langfuse.create_trace_id()
    with langfuse_client.start_as_current_span(
        name=name, trace_context={"trace_id": trace_id}
    ):
        test_phrase = "This is a test!"
        _ = await model.ainvoke(
            [{"role": "user", "content": f"return the exact phrase - {test_phrase} "}],
            config={"callbacks": [handler]},
        )

    langfuse_client.flush()
    assert handler.runs == {}
    api = get_api()
    trace = api.trace.get(trace_id)
    generationList = list(filter(lambda o: o.type == "GENERATION", trace.observations))
    assert len(generationList) != 0

    generation = generationList[0]

    assert len(trace.observations) == 2
    assert trace.name == name
    assert model_name in generation.model
    assert generation.input is not None
    assert generation.output is not None
    assert generation.model_parameters.get("max_completion_tokens") is not None
    assert generation.model_parameters.get("temperature") is not None
    assert generation.metadata["tags"] == tags
    assert generation.usage.output is not None
    assert generation.usage.total is not None
    assert generation.output["content"] is not None
    assert generation.output["role"] is not None
    assert generation.input_price is not None
    assert generation.output_price is not None
    assert generation.calculated_input_cost is not None
    assert generation.calculated_output_cost is not None
    assert generation.calculated_total_cost is not None
    assert generation.latency is not None


@pytest.mark.skip(
    reason="This test suite is not properly isolated and fails flakily. TODO: Investigate why"
)
@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", ["gpt-3.5-turbo-instruct"])
async def test_ainvoke_in_completions_models(model_name):
    name = f"test_ainvoke_in_completions_models-{create_uuid()}"
    tags = ["Hello", "world"]
    model = OpenAI(max_tokens=300, tags=tags, model=model_name)
    handler = CallbackHandler()

    langfuse_client = handler.client
    trace_id = Langfuse.create_trace_id()
    with langfuse_client.start_as_current_span(
        name=name, trace_context={"trace_id": trace_id}
    ):
        test_phrase = "This is a test!"
        _ = await model.ainvoke(
            f"return the exact phrase - {test_phrase}",
            config={"callbacks": [handler]},
        )

    langfuse_client.flush()
    assert handler.runs == {}
    api = get_api()
    trace = api.trace.get(trace_id)
    generationList = list(filter(lambda o: o.type == "GENERATION", trace.observations))
    assert len(generationList) != 0

    generation = generationList[0]

    assert len(trace.observations) == 2
    assert trace.name == name
    assert model_name in generation.model
    assert generation.input is not None
    assert generation.output is not None
    assert generation.model_parameters.get("max_tokens") is not None
    assert generation.model_parameters.get("temperature") is not None
    assert generation.metadata["tags"] == tags
    assert generation.usage.output is not None
    assert generation.usage.total is not None
    assert test_phrase in generation.output
    assert generation.input_price is not None
    assert generation.output_price is not None
    assert generation.calculated_input_cost is not None
    assert generation.calculated_output_cost is not None
    assert generation.calculated_total_cost is not None
    assert generation.latency is not None


# Chains


# Sync batch in chains and chat models
@pytest.mark.skip(
    reason="This test suite is not properly isolated and fails flakily. TODO: Investigate why"
)
@pytest.mark.parametrize("model_name", ["gpt-3.5-turbo", "gpt-4"])
def test_chains_batch_in_chat_models(model_name):
    name = f"test_chains_batch_in_chat_models-{create_uuid()}"
    tags = ["Hello", "world"]
    model = ChatOpenAI(max_completion_tokens=300, tags=tags, model=model_name)
    handler = CallbackHandler()

    langfuse_client = handler.client
    trace_id = Langfuse.create_trace_id()
    with langfuse_client.start_as_current_span(
        name=name, trace_context={"trace_id": trace_id}
    ):
        prompt = ChatPromptTemplate.from_template(
            "tell me a joke about {foo} in 300 words"
        )
        inputs = [{"foo": "bears"}, {"foo": "cats"}]
        chain = prompt | model | StrOutputParser()
        _ = chain.batch(
            inputs,
            config={"callbacks": [handler]},
        )

    langfuse_client.flush()
    assert handler.runs == {}
    api = get_api()
    trace = api.trace.get(trace_id)
    generationList = list(filter(lambda o: o.type == "GENERATION", trace.observations))
    assert len(generationList) != 0

    assert len(trace.observations) == 9
    for generation in generationList:
        assert trace.name == name
        assert model_name in generation.model
        assert generation.input is not None
        assert generation.output is not None
        assert generation.model_parameters.get("max_completion_tokens") is not None
        assert generation.model_parameters.get("temperature") is not None
        assert all(x in generation.metadata["tags"] for x in tags)
        assert generation.usage.output is not None
        assert generation.usage.total is not None
        assert generation.input_price is not None
        assert generation.output_price is not None
        assert generation.calculated_input_cost is not None
        assert generation.calculated_output_cost is not None
        assert generation.calculated_total_cost is not None
        assert generation.latency is not None


@pytest.mark.skip(
    reason="This test suite is not properly isolated and fails flakily. TODO: Investigate why"
)
@pytest.mark.parametrize("model_name", ["gpt-3.5-turbo-instruct"])
def test_chains_batch_in_completions_models(model_name):
    name = f"test_chains_batch_in_completions_models-{create_uuid()}"
    tags = ["Hello", "world"]
    model = OpenAI(max_tokens=300, tags=tags, model=model_name)
    handler = CallbackHandler()

    langfuse_client = handler.client
    trace_id = Langfuse.create_trace_id()
    with langfuse_client.start_as_current_span(
        name=name, trace_context={"trace_id": trace_id}
    ):
        prompt = ChatPromptTemplate.from_template(
            "tell me a joke about {foo} in 300 words"
        )
        inputs = [{"foo": "bears"}, {"foo": "cats"}]
        chain = prompt | model | StrOutputParser()
        _ = chain.batch(
            inputs,
            config={"callbacks": [handler]},
        )

    langfuse_client.flush()
    assert handler.runs == {}
    api = get_api()
    trace = api.trace.get(trace_id)
    generationList = list(filter(lambda o: o.type == "GENERATION", trace.observations))
    assert len(generationList) != 0

    assert len(trace.observations) == 9
    for generation in generationList:
        assert trace.name == name
        assert model_name in generation.model
        assert generation.input is not None
        assert generation.output is not None
        assert generation.model_parameters.get("max_tokens") is not None
        assert generation.model_parameters.get("temperature") is not None
        assert all(x in generation.metadata["tags"] for x in tags)
        assert generation.usage.output is not None
        assert generation.usage.total is not None
        assert generation.input_price is not None
        assert generation.output_price is not None
        assert generation.calculated_input_cost is not None
        assert generation.calculated_output_cost is not None
        assert generation.calculated_total_cost is not None
        assert generation.latency is not None


# Async batch call with chains and chat models
@pytest.mark.skip(
    reason="This test suite is not properly isolated and fails flakily. TODO: Investigate why"
)
@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", ["gpt-3.5-turbo", "gpt-4"])
async def test_chains_abatch_in_chat_models(model_name):
    name = f"test_chains_abatch_in_chat_models-{create_uuid()}"
    tags = ["Hello", "world"]
    model = ChatOpenAI(max_completion_tokens=300, tags=tags, model=model_name)
    handler = CallbackHandler()

    langfuse_client = handler.client
    trace_id = Langfuse.create_trace_id()
    with langfuse_client.start_as_current_span(
        name=name, trace_context={"trace_id": trace_id}
    ):
        prompt = ChatPromptTemplate.from_template(
            "tell me a joke about {foo} in 300 words"
        )
        inputs = [{"foo": "bears"}, {"foo": "cats"}]
        chain = prompt | model | StrOutputParser()
        _ = await chain.abatch(
            inputs,
            config={"callbacks": [handler]},
        )

    langfuse_client.flush()
    assert handler.runs == {}
    api = get_api()
    trace = api.trace.get(trace_id)
    generationList = list(filter(lambda o: o.type == "GENERATION", trace.observations))
    assert len(generationList) != 0

    assert len(trace.observations) == 9
    for generation in generationList:
        assert trace.name == name
        assert model_name in generation.model
        assert generation.input is not None
        assert generation.output is not None
        assert generation.model_parameters.get("max_completion_tokens") is not None
        assert generation.model_parameters.get("temperature") is not None
        assert all(x in generation.metadata["tags"] for x in tags)
        assert generation.usage.output is not None
        assert generation.usage.total is not None
        assert generation.input_price is not None
        assert generation.output_price is not None
        assert generation.calculated_input_cost is not None
        assert generation.calculated_output_cost is not None
        assert generation.calculated_total_cost is not None
        assert generation.latency is not None


# Async batch call with chains and completions models
@pytest.mark.skip(
    reason="This test suite is not properly isolated and fails flakily. TODO: Investigate why"
)
@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", ["gpt-3.5-turbo-instruct"])
async def test_chains_abatch_in_completions_models(model_name):
    name = f"test_chains_abatch_in_completions_models-{create_uuid()}"
    tags = ["Hello", "world"]
    model = OpenAI(max_tokens=300, tags=tags, model=model_name)
    handler = CallbackHandler()

    langfuse_client = handler.client
    trace_id = Langfuse.create_trace_id()
    with langfuse_client.start_as_current_span(
        name=name, trace_context={"trace_id": trace_id}
    ):
        prompt = ChatPromptTemplate.from_template(
            "tell me a joke about {foo} in 300 words"
        )
        inputs = [{"foo": "bears"}, {"foo": "cats"}]
        chain = prompt | model | StrOutputParser()
        _ = await chain.abatch(inputs, config={"callbacks": [handler]})

    langfuse_client.flush()
    assert handler.runs == {}
    api = get_api()
    trace = api.trace.get(trace_id)
    generationList = list(filter(lambda o: o.type == "GENERATION", trace.observations))
    assert len(generationList) != 0
    assert len(trace.observations) == 9
    for generation in generationList:
        assert trace.name == name
        assert model_name in generation.model
        assert generation.input is not None
        assert generation.output is not None
        assert generation.model_parameters.get("max_tokens") is not None
        assert generation.model_parameters.get("temperature") is not None
        assert all(x in generation.metadata["tags"] for x in tags)
        assert generation.usage.output is not None
        assert generation.usage.total is not None
        assert generation.input_price is not None
        assert generation.output_price is not None
        assert generation.calculated_input_cost is not None
        assert generation.calculated_output_cost is not None
        assert generation.calculated_total_cost is not None
        assert generation.latency is not None


# Async invoke in chains and chat models
@pytest.mark.skip(
    reason="This test suite is not properly isolated and fails flakily. TODO: Investigate why"
)
@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", ["gpt-3.5-turbo"])
async def test_chains_ainvoke_chat_models(model_name):
    name = f"test_chains_ainvoke_chat_models-{create_uuid()}"
    tags = ["Hello", "world"]
    model = ChatOpenAI(max_completion_tokens=300, tags=tags, model=model_name)
    handler = CallbackHandler()

    langfuse_client = handler.client
    trace_id = Langfuse.create_trace_id()
    with langfuse_client.start_as_current_span(
        name=name, trace_context={"trace_id": trace_id}
    ):
        prompt1 = ChatPromptTemplate.from_template(
            """You are a skilled writer tasked with crafting an engaging introduction for a blog post on the following topic:
            Topic: {topic}
            Introduction: This is an engaging introduction for the blog post on the topic above:"""
        )
        chain = prompt1 | model | StrOutputParser()
        await chain.ainvoke(
            {"topic": "The Impact of Climate Change"},
            config={"callbacks": [handler]},
        )

    langfuse_client.flush()
    assert handler.runs == {}
    api = get_api()
    trace = api.trace.get(trace_id)
    generationList = list(filter(lambda o: o.type == "GENERATION", trace.observations))
    assert len(generationList) != 0

    assert len(trace.observations) == 5
    assert trace.name == name
    for generation in generationList:
        assert model_name in generation.model
        assert generation.input is not None
        assert generation.output is not None
        assert generation.model_parameters.get("max_completion_tokens") is not None
        assert generation.model_parameters.get("temperature") is not None
        assert all(x in generation.metadata["tags"] for x in tags)
        assert generation.usage.output is not None
        assert generation.usage.total is not None
        assert generation.output["content"] is not None
        assert generation.output["role"] is not None
        assert generation.input_price is not None
        assert generation.output_price is not None
        assert generation.calculated_input_cost is not None
        assert generation.calculated_output_cost is not None
        assert generation.calculated_total_cost is not None
        assert generation.latency is not None


# Async invoke in chains and completions models
@pytest.mark.skip(
    reason="This test suite is not properly isolated and fails flakily. TODO: Investigate why"
)
@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", ["gpt-3.5-turbo-instruct"])
async def test_chains_ainvoke_completions_models(model_name):
    name = f"test_chains_ainvoke_completions_models-{create_uuid()}"
    tags = ["Hello", "world"]
    model = OpenAI(max_tokens=300, tags=tags, model=model_name)
    handler = CallbackHandler()

    langfuse_client = handler.client
    trace_id = Langfuse.create_trace_id()
    with langfuse_client.start_as_current_span(
        name=name, trace_context={"trace_id": trace_id}
    ):
        prompt1 = PromptTemplate.from_template(
            """You are a skilled writer tasked with crafting an engaging introduction for a blog post on the following topic:
            Topic: {topic}
            Introduction: This is an engaging introduction for the blog post on the topic above:"""
        )
        chain = prompt1 | model | StrOutputParser()
        await chain.ainvoke(
            {"topic": "The Impact of Climate Change"},
            config={"callbacks": [handler]},
        )

    langfuse_client.flush()
    assert handler.runs == {}
    api = get_api()
    trace = api.trace.get(trace_id)
    generationList = list(filter(lambda o: o.type == "GENERATION", trace.observations))
    assert len(generationList) != 0

    generation = generationList[0]
    assert len(trace.observations) == 5
    assert trace.name == name
    assert model_name in generation.model
    assert generation.input is not None
    assert generation.output is not None
    assert generation.model_parameters.get("max_tokens") is not None
    assert generation.model_parameters.get("temperature") is not None
    assert all(x in generation.metadata["tags"] for x in tags)
    assert generation.usage.output is not None
    assert generation.usage.total is not None
    assert generation.input_price is not None
    assert generation.output_price is not None
    assert generation.calculated_input_cost is not None
    assert generation.calculated_output_cost is not None
    assert generation.calculated_total_cost is not None
    assert generation.latency is not None


# Async streaming in chat models
@pytest.mark.skip(
    reason="This test suite is not properly isolated and fails flakily. TODO: Investigate why"
)
@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", ["gpt-3.5-turbo", "gpt-4"])
async def test_chains_astream_chat_models(model_name):
    name = f"test_chains_astream_chat_models-{create_uuid()}"
    tags = ["Hello", "world"]
    model = ChatOpenAI(
        streaming=True, max_completion_tokens=300, tags=tags, model=model_name
    )
    handler = CallbackHandler()

    langfuse_client = handler.client
    trace_id = Langfuse.create_trace_id()
    with langfuse_client.start_as_current_span(
        name=name, trace_context={"trace_id": trace_id}
    ):
        prompt1 = PromptTemplate.from_template(
            """You are a skilled writer tasked with crafting an engaging introduction for a blog post on the following topic:
            Topic: {topic}
            Introduction: This is an engaging introduction for the blog post on the topic above:"""
        )
        chain = prompt1 | model | StrOutputParser()
        res = chain.astream(
            {"topic": "The Impact of Climate Change"},
            config={"callbacks": [handler]},
        )
        response_str = []
        assert _is_streaming_response(res)
        async for chunk in res:
            response_str.append(chunk)

    langfuse_client.flush()
    assert handler.runs == {}
    api = get_api()
    trace = api.trace.get(trace_id)
    generationList = list(filter(lambda o: o.type == "GENERATION", trace.observations))
    assert len(generationList) != 0

    generation = generationList[0]

    assert len(response_str) > 1  # To check there are more than one chunk.
    assert len(trace.observations) == 5
    assert trace.name == name
    assert model_name in generation.model
    assert generation.input is not None
    assert generation.output is not None
    assert generation.model_parameters.get("max_completion_tokens") is not None
    assert generation.model_parameters.get("temperature") is not None
    assert all(x in generation.metadata["tags"] for x in tags)
    assert generation.usage.output is not None
    assert generation.usage.total is not None
    assert generation.output["content"] is not None
    assert generation.output["role"] is not None
    assert generation.input_price is not None
    assert generation.output_price is not None
    assert generation.calculated_input_cost is not None
    assert generation.calculated_output_cost is not None
    assert generation.calculated_total_cost is not None
    assert generation.latency is not None


# Async Streaming in completions models
@pytest.mark.skip(
    reason="This test suite is not properly isolated and fails flakily. TODO: Investigate why"
)
@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", ["gpt-3.5-turbo-instruct"])
async def test_chains_astream_completions_models(model_name):
    name = f"test_chains_astream_completions_models-{create_uuid()}"
    tags = ["Hello", "world"]
    model = OpenAI(streaming=True, max_tokens=300, tags=tags, model=model_name)
    handler = CallbackHandler()

    langfuse_client = handler.client
    trace_id = Langfuse.create_trace_id()
    with langfuse_client.start_as_current_span(
        name=name, trace_context={"trace_id": trace_id}
    ):
        prompt1 = PromptTemplate.from_template(
            """You are a skilled writer tasked with crafting an engaging introduction for a blog post on the following topic:
            Topic: {topic}
            Introduction: This is an engaging introduction for the blog post on the topic above:"""
        )
        chain = prompt1 | model | StrOutputParser()
        res = chain.astream(
            {"topic": "The Impact of Climate Change"},
            config={"callbacks": [handler]},
        )
        response_str = []
        assert _is_streaming_response(res)
        async for chunk in res:
            response_str.append(chunk)

    langfuse_client.flush()
    assert handler.runs == {}
    api = get_api()
    trace = api.trace.get(trace_id)
    generationList = list(filter(lambda o: o.type == "GENERATION", trace.observations))
    assert len(generationList) != 0

    generation = generationList[0]

    assert len(response_str) > 1  # To check there are more than one chunk.
    assert len(trace.observations) == 5
    assert trace.name == name
    assert model_name in generation.model
    assert generation.input is not None
    assert generation.output is not None
    assert generation.model_parameters.get("max_tokens") is not None
    assert generation.model_parameters.get("temperature") is not None
    assert all(x in generation.metadata["tags"] for x in tags)
    assert generation.usage.output is not None
    assert generation.usage.total is not None
    assert generation.input_price is not None
    assert generation.output_price is not None
    assert generation.calculated_input_cost is not None
    assert generation.calculated_output_cost is not None
    assert generation.calculated_total_cost is not None
    assert generation.latency is not None
