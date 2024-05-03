import json
import os
from typing import List

from langchain import LLMChain, OpenAI, PromptTemplate

from langfuse import Langfuse
from langfuse.api.resources.commons.types.observation import Observation
from tests.utils import create_uuid, get_api


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

    assert dataset.items[1].input == input
    assert dataset.items[1].expected_output == "Output"
    assert dataset.items[1].metadata == {"key": "value"}
    assert dataset.items[1].source_observation_id == generation.id
    assert dataset.items[1].source_trace_id == generation.trace_id

    assert dataset.items[0].input is None
    assert dataset.items[0].expected_output is None
    assert dataset.items[0].metadata is None
    assert dataset.items[0].source_observation_id is None
    assert dataset.items[0].source_trace_id is None


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
        dataset_name=name, input=new_input, id=item.id, expected_output=new_input
    )
    get_new_item = langfuse.get_dataset_item(item.id)
    assert get_new_item.input == new_input
    assert get_new_item.id == item.id
    assert get_new_item.expected_output == new_input


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
