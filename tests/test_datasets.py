import json
import os
from typing import List
from concurrent.futures import ThreadPoolExecutor

from langchain import LLMChain, OpenAI, PromptTemplate

from langfuse import Langfuse
from langfuse.decorators import observe, langfuse_context
from langfuse.api.resources.commons.types.observation import Observation
from tests.utils import create_uuid, get_api, get_llama_index_index


def test_create_and_get_dataset():
    langfuse = Langfuse(debug=False)

    name = create_uuid()
    langfuse.create_dataset(name=name)
    dataset = langfuse.get_dataset(name)
    assert dataset.name == name

    name = create_uuid()
    langfuse.create_dataset(
        name=name, description="This is a test dataset", metadata={"key": "value"}
    )
    dataset = langfuse.get_dataset(name)
    assert dataset.name == name
    assert dataset.description == "This is a test dataset"
    assert dataset.metadata == {"key": "value"}


def test_create_dataset_item():
    langfuse = Langfuse(debug=False)
    name = create_uuid()
    langfuse.create_dataset(name=name)

    generation = langfuse.generation(name="test")
    langfuse.flush()

    input = {"input": "Hello World"}
    # 2
    langfuse.create_dataset_item(dataset_name=name, input=input)
    # 1
    langfuse.create_dataset_item(
        dataset_name=name,
        input=input,
        expected_output="Output",
        metadata={"key": "value"},
        source_observation_id=generation.id,
        source_trace_id=generation.trace_id,
    )
    # 0 - no data
    langfuse.create_dataset_item(
        dataset_name=name,
    )

    dataset = langfuse.get_dataset(name)
    assert len(dataset.items) == 3
    assert dataset.items[2].input == input
    assert dataset.items[2].expected_output is None
    assert dataset.items[2].dataset_name == name

    assert dataset.items[1].input == input
    assert dataset.items[1].expected_output == "Output"
    assert dataset.items[1].metadata == {"key": "value"}
    assert dataset.items[1].source_observation_id == generation.id
    assert dataset.items[1].source_trace_id == generation.trace_id
    assert dataset.items[1].dataset_name == name

    assert dataset.items[0].input is None
    assert dataset.items[0].expected_output is None
    assert dataset.items[0].metadata is None
    assert dataset.items[0].source_observation_id is None
    assert dataset.items[0].source_trace_id is None
    assert dataset.items[0].dataset_name == name


def test_get_all_items():
    langfuse = Langfuse(debug=False)
    name = create_uuid()
    langfuse.create_dataset(name=name)

    input = {"input": "Hello World"}
    for _ in range(99):
        langfuse.create_dataset_item(dataset_name=name, input=input)

    dataset = langfuse.get_dataset(name)
    assert len(dataset.items) == 99

    dataset_2 = langfuse.get_dataset(name, fetch_items_page_size=9)
    assert len(dataset_2.items) == 99

    dataset_3 = langfuse.get_dataset(name, fetch_items_page_size=2)
    assert len(dataset_3.items) == 99


def test_upsert_and_get_dataset_item():
    langfuse = Langfuse(debug=False)
    name = create_uuid()
    langfuse.create_dataset(name=name)
    input = {"input": "Hello World"}
    item = langfuse.create_dataset_item(
        dataset_name=name, input=input, expected_output=input
    )

    get_item = langfuse.get_dataset_item(item.id)
    assert get_item.input == input
    assert get_item.id == item.id
    assert get_item.expected_output == input

    new_input = {"input": "Hello World 2"}
    langfuse.create_dataset_item(
        dataset_name=name,
        input=new_input,
        id=item.id,
        expected_output=new_input,
        status="ARCHIVED",
    )
    get_new_item = langfuse.get_dataset_item(item.id)
    assert get_new_item.input == new_input
    assert get_new_item.id == item.id
    assert get_new_item.expected_output == new_input
    assert get_new_item.status == "ARCHIVED"


def test_linking_observation():
    langfuse = Langfuse(debug=False)

    dataset_name = create_uuid()
    langfuse.create_dataset(name=dataset_name)

    input = json.dumps({"input": "Hello World"})
    langfuse.create_dataset_item(dataset_name=dataset_name, input=input)

    dataset = langfuse.get_dataset(dataset_name)
    assert len(dataset.items) == 1
    assert dataset.items[0].input == input

    run_name = create_uuid()
    generation_id = create_uuid()
    trace_id = None

    for item in dataset.items:
        generation = langfuse.generation(id=generation_id)
        trace_id = generation.trace_id

        item.link(generation, run_name)

    run = langfuse.get_dataset_run(dataset_name, run_name)

    assert run.name == run_name
    assert len(run.dataset_run_items) == 1
    assert run.dataset_run_items[0].observation_id == generation_id
    assert run.dataset_run_items[0].trace_id == trace_id


def test_linking_trace_and_run_metadata_and_description():
    langfuse = Langfuse(debug=False)

    dataset_name = create_uuid()
    langfuse.create_dataset(name=dataset_name)

    input = json.dumps({"input": "Hello World"})
    langfuse.create_dataset_item(dataset_name=dataset_name, input=input)

    dataset = langfuse.get_dataset(dataset_name)
    assert len(dataset.items) == 1
    assert dataset.items[0].input == input

    run_name = create_uuid()
    trace_id = create_uuid()

    for item in dataset.items:
        trace = langfuse.trace(id=trace_id)

        item.link(
            trace,
            run_name,
            run_metadata={"key": "value"},
            run_description="This is a test run",
        )

    run = langfuse.get_dataset_run(dataset_name, run_name)

    assert run.name == run_name
    assert run.metadata == {"key": "value"}
    assert run.description == "This is a test run"
    assert len(run.dataset_run_items) == 1
    assert run.dataset_run_items[0].trace_id == trace_id
    assert run.dataset_run_items[0].observation_id is None


def test_get_runs():
    langfuse = Langfuse(debug=False)

    dataset_name = create_uuid()
    langfuse.create_dataset(name=dataset_name)

    input = json.dumps({"input": "Hello World"})
    langfuse.create_dataset_item(dataset_name=dataset_name, input=input)

    dataset = langfuse.get_dataset(dataset_name)
    assert len(dataset.items) == 1
    assert dataset.items[0].input == input

    run_name_1 = create_uuid()
    trace_id_1 = create_uuid()

    for item in dataset.items:
        trace = langfuse.trace(id=trace_id_1)

        item.link(
            trace,
            run_name_1,
            run_metadata={"key": "value"},
            run_description="This is a test run",
        )

    run_name_2 = create_uuid()
    trace_id_2 = create_uuid()

    for item in dataset.items:
        trace = langfuse.trace(id=trace_id_2)

        item.link(
            trace,
            run_name_2,
            run_metadata={"key": "value"},
            run_description="This is a test run",
        )

    runs = langfuse.get_dataset_runs(dataset_name)

    assert len(runs.data) == 2
    assert runs.data[0].name == run_name_2
    assert runs.data[0].metadata == {"key": "value"}
    assert runs.data[0].description == "This is a test run"
    assert runs.data[1].name == run_name_1
    assert runs.meta.total_items == 2
    assert runs.meta.total_pages == 1
    assert runs.meta.page == 1
    assert runs.meta.limit == 50


def test_linking_via_id_observation_arg_legacy():
    langfuse = Langfuse(debug=False)

    dataset_name = create_uuid()
    langfuse.create_dataset(name=dataset_name)

    input = json.dumps({"input": "Hello World"})
    langfuse.create_dataset_item(dataset_name=dataset_name, input=input)

    dataset = langfuse.get_dataset(dataset_name)
    assert len(dataset.items) == 1
    assert dataset.items[0].input == input

    run_name = create_uuid()
    generation_id = create_uuid()
    trace_id = None

    for item in dataset.items:
        generation = langfuse.generation(id=generation_id)
        trace_id = generation.trace_id
        langfuse.flush()

        item.link(generation_id, run_name)

    run = langfuse.get_dataset_run(dataset_name, run_name)

    assert run.name == run_name
    assert len(run.dataset_run_items) == 1
    assert run.dataset_run_items[0].observation_id == generation_id
    assert run.dataset_run_items[0].trace_id == trace_id


def test_linking_via_id_trace_kwarg():
    langfuse = Langfuse(debug=False)

    dataset_name = create_uuid()
    langfuse.create_dataset(name=dataset_name)

    input = json.dumps({"input": "Hello World"})
    langfuse.create_dataset_item(dataset_name=dataset_name, input=input)

    dataset = langfuse.get_dataset(dataset_name)
    assert len(dataset.items) == 1
    assert dataset.items[0].input == input

    run_name = create_uuid()
    trace_id = create_uuid()

    for item in dataset.items:
        langfuse.trace(id=trace_id)
        langfuse.flush()

        item.link(None, run_name, trace_id=trace_id)

    run = langfuse.get_dataset_run(dataset_name, run_name)

    assert run.name == run_name
    assert len(run.dataset_run_items) == 1
    assert run.dataset_run_items[0].observation_id is None
    assert run.dataset_run_items[0].trace_id == trace_id


def test_linking_via_id_generation_kwarg():
    langfuse = Langfuse(debug=False)

    dataset_name = create_uuid()
    langfuse.create_dataset(name=dataset_name)

    input = json.dumps({"input": "Hello World"})
    langfuse.create_dataset_item(dataset_name=dataset_name, input=input)

    dataset = langfuse.get_dataset(dataset_name)
    assert len(dataset.items) == 1
    assert dataset.items[0].input == input

    run_name = create_uuid()
    generation_id = create_uuid()
    trace_id = None

    for item in dataset.items:
        generation = langfuse.generation(id=generation_id)
        trace_id = generation.trace_id
        langfuse.flush()

        item.link(None, run_name, trace_id=trace_id, observation_id=generation_id)

    run = langfuse.get_dataset_run(dataset_name, run_name)

    assert run.name == run_name
    assert len(run.dataset_run_items) == 1
    assert run.dataset_run_items[0].observation_id == generation_id
    assert run.dataset_run_items[0].trace_id == trace_id


def test_langchain_dataset():
    langfuse = Langfuse(debug=False)
    dataset_name = create_uuid()
    langfuse.create_dataset(name=dataset_name)

    input = json.dumps({"input": "Hello World"})
    langfuse.create_dataset_item(dataset_name=dataset_name, input=input)

    dataset = langfuse.get_dataset(dataset_name)

    run_name = create_uuid()

    dataset_item_id = None

    for item in dataset.items:
        handler = item.get_langchain_handler(run_name=run_name)
        dataset_item_id = item.id
        llm = OpenAI(openai_api_key=os.environ.get("OPENAI_API_KEY"))
        template = """You are a playwright. Given the title of play, it is your job to write a synopsis for that title.
            Title: {title}
            Playwright: This is a synopsis for the above play:"""

        prompt_template = PromptTemplate(input_variables=["title"], template=template)
        synopsis_chain = LLMChain(llm=llm, prompt=prompt_template)

        synopsis_chain.run("Tragedy at sunset on the beach", callbacks=[handler])

    langfuse.flush()
    run = langfuse.get_dataset_run(dataset_name, run_name)

    assert run.name == run_name
    assert len(run.dataset_run_items) == 1
    assert run.dataset_run_items[0].dataset_run_id == run.id

    api = get_api()

    trace = api.trace.get(handler.get_trace_id())

    assert len(trace.observations) == 2

    sorted_observations = sorted_dependencies(trace.observations)

    assert sorted_observations[0].id == sorted_observations[1].parent_observation_id
    assert sorted_observations[0].parent_observation_id is None

    assert trace.name == "LLMChain"  # Overwritten by the Langchain run
    assert trace.metadata == {
        "dataset_item_id": dataset_item_id,
        "run_name": run_name,
        "dataset_id": dataset.id,
    }

    assert sorted_observations[0].name == "LLMChain"

    assert sorted_observations[1].name == "OpenAI"
    assert sorted_observations[1].type == "GENERATION"
    assert sorted_observations[1].input is not None
    assert sorted_observations[1].output is not None
    assert sorted_observations[1].input != ""
    assert sorted_observations[1].output != ""
    assert sorted_observations[1].usage.total is not None
    assert sorted_observations[1].usage.input is not None
    assert sorted_observations[1].usage.output is not None


def test_llama_index_dataset():
    langfuse = Langfuse(debug=False)
    dataset_name = create_uuid()
    langfuse.create_dataset(name=dataset_name)

    langfuse.create_dataset_item(
        dataset_name=dataset_name, input={"input": "Hello World"}
    )

    dataset = langfuse.get_dataset(dataset_name)

    run_name = create_uuid()

    dataset_item_id = None

    for item in dataset.items:
        with item.observe_llama_index(run_name=run_name) as handler:
            dataset_item_id = item.id

            index = get_llama_index_index(handler)
            index.as_query_engine().query(
                "What did the speaker achieve in the past twelve months?"
            )

    langfuse.flush()
    run = langfuse.get_dataset_run(dataset_name, run_name)

    assert run.name == run_name
    assert len(run.dataset_run_items) == 1
    assert run.dataset_run_items[0].dataset_run_id == run.id

    api = get_api()

    trace = api.trace.get(handler.get_trace_id())

    sorted_observations = sorted_dependencies(trace.observations)

    assert sorted_observations[0].id == sorted_observations[1].parent_observation_id
    assert sorted_observations[0].parent_observation_id is None

    assert trace.name == "LlamaIndex_query"  # Overwritten by the Langchain run
    assert trace.metadata == {
        "dataset_item_id": dataset_item_id,
        "run_name": run_name,
        "dataset_id": dataset.id,
    }

    assert sorted_observations[0].name == "query"
    assert sorted_observations[1].name == "synthesize"


def sorted_dependencies(
    observations: List[Observation],
):
    # observations have an id and a parent_observation_id. Return a sorted list starting with the root observation where the parent_observation_id is None
    parent_to_observation = {obs.parent_observation_id: obs for obs in observations}

    # Start with the root observation (parent_observation_id is None)
    current_observation = parent_to_observation[None]
    dependencies = [current_observation]

    while current_observation.id in parent_to_observation:
        current_observation = parent_to_observation[current_observation.id]
        dependencies.append(current_observation)

    return dependencies


def test_observe_dataset_run():
    # Create dataset
    langfuse = Langfuse(debug=True)
    dataset_name = create_uuid()
    langfuse.create_dataset(name=dataset_name)

    items_data = []
    num_items = 3

    for i in range(num_items):
        trace_id = create_uuid()
        dataset_item_input = "Hello World " + str(i)
        langfuse.create_dataset_item(
            dataset_name=dataset_name, input=dataset_item_input
        )

        items_data.append((dataset_item_input, trace_id))

    dataset = langfuse.get_dataset(dataset_name)
    assert len(dataset.items) == num_items

    run_name = create_uuid()

    @observe()
    def run_llm_app_on_dataset_item(input):
        return input

    def wrapperFunc(input):
        return run_llm_app_on_dataset_item(input)

    def execute_dataset_item(item, run_name, trace_id):
        with item.observe(run_name=run_name, trace_id=trace_id):
            wrapperFunc(item.input)

    items = zip(dataset.items[::-1], items_data)  # Reverse order to reflect input order

    with ThreadPoolExecutor() as executor:
        for item, (_, trace_id) in items:
            result = executor.submit(
                execute_dataset_item,
                item,
                run_name=run_name,
                trace_id=trace_id,
            )

            result.result()

    langfuse_context.flush()

    # Check dataset run
    run = langfuse.get_dataset_run(dataset_name, run_name)

    assert run.name == run_name
    assert len(run.dataset_run_items) == num_items
    assert run.dataset_run_items[0].dataset_run_id == run.id

    for _, trace_id in items_data:
        assert any(
            item.trace_id == trace_id for item in run.dataset_run_items
        ), f"Trace {trace_id} not found in run"

    # Check trace
    api = get_api()

    for dataset_item_input, trace_id in items_data:
        trace = api.trace.get(trace_id)

        assert trace.name == "run_llm_app_on_dataset_item"
        assert len(trace.observations) == 0
        assert trace.input["args"][0] == dataset_item_input
        assert trace.output == dataset_item_input

    # Check that the decorator context is not polluted
    new_trace_id = create_uuid()
    run_llm_app_on_dataset_item(
        "non-dataset-run-afterwards", langfuse_observation_id=new_trace_id
    )

    langfuse_context.flush()

    next_trace = api.trace.get(new_trace_id)
    assert next_trace.name == "run_llm_app_on_dataset_item"
    assert next_trace.input["args"][0] == "non-dataset-run-afterwards"
    assert next_trace.output == "non-dataset-run-afterwards"
    assert len(next_trace.observations) == 0
    assert next_trace.id != trace_id
