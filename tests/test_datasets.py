import json
import os
from typing import List

from langchain import LLMChain, OpenAI, PromptTemplate
import pytest

from langfuse import Langfuse
from langfuse.api.resources.commons.types.observation import Observation

from langfuse.model import CreateDatasetItemRequest, InitialGeneration
from langfuse.model import CreateDatasetRequest


from tests.utils import create_uuid, get_api


def test_create_and_get_dataset():
    langfuse = Langfuse(debug=False)

    name = create_uuid()
    langfuse.create_dataset(CreateDatasetRequest(name=name))
    dataset = langfuse.get_dataset(name)
    assert dataset.name == name


def test_create_dataset_item():
    langfuse = Langfuse(debug=False)
    name = create_uuid()
    langfuse.create_dataset(CreateDatasetRequest(name=name))

    input = json.dumps({"input": "Hello World"})
    langfuse.create_dataset_item(CreateDatasetItemRequest(dataset_name=name, input=input))

    dataset = langfuse.get_dataset(name)
    assert len(dataset.items) == 1
    assert dataset.items[0].input == input


def test_linking_observation():
    langfuse = Langfuse(debug=False)

    dataset_name = create_uuid()
    langfuse.create_dataset(CreateDatasetRequest(name=dataset_name))

    input = json.dumps({"input": "Hello World"})
    langfuse.create_dataset_item(CreateDatasetItemRequest(dataset_name=dataset_name, input=input))

    dataset = langfuse.get_dataset(dataset_name)
    assert len(dataset.items) == 1
    assert dataset.items[0].input == input

    run_name = create_uuid()
    generation_id = create_uuid()

    for item in dataset.items:
        generation = langfuse.generation(InitialGeneration(id=generation_id))

        item.link(generation, run_name)

    run = langfuse.get_dataset_run(dataset_name, run_name)

    assert run.name == run_name
    assert len(run.dataset_run_items) == 1
    assert run.dataset_run_items[0].observation_id == generation_id


def test_linking_via_id_observation():
    langfuse = Langfuse(debug=False)

    dataset_name = create_uuid()
    langfuse.create_dataset(CreateDatasetRequest(name=dataset_name))

    input = json.dumps({"input": "Hello World"})
    langfuse.create_dataset_item(CreateDatasetItemRequest(dataset_name=dataset_name, input=input))

    dataset = langfuse.get_dataset(dataset_name)
    assert len(dataset.items) == 1
    assert dataset.items[0].input == input

    run_name = create_uuid()
    generation_id = create_uuid()

    for item in dataset.items:
        langfuse.generation(InitialGeneration(id=generation_id))
        langfuse.flush()

        item.link(generation_id, run_name)

    run = langfuse.get_dataset_run(dataset_name, run_name)

    assert run.name == run_name
    assert len(run.dataset_run_items) == 1
    assert run.dataset_run_items[0].observation_id == generation_id


@pytest.mark.skip(reason="inference cost")
def test_langchain_dataset():
    langfuse = Langfuse(debug=False)
    dataset_name = create_uuid()
    langfuse.create_dataset(CreateDatasetRequest(name=dataset_name))

    input = json.dumps({"input": "Hello World"})
    langfuse.create_dataset_item(CreateDatasetItemRequest(dataset_name=dataset_name, input=input))

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

    run = langfuse.get_dataset_run(dataset_name, run_name)

    assert run.name == run_name
    assert len(run.dataset_run_items) == 1
    assert run.dataset_run_items[0].dataset_run_id == run.id

    api = get_api()

    trace = api.trace.get(handler.get_trace_id())

    assert len(trace.observations) == 3

    sorted_observations = sorted_dependencies(trace.observations)

    assert sorted_observations[1].id == sorted_observations[2].parent_observation_id
    assert sorted_observations[0].id == sorted_observations[1].parent_observation_id
    assert sorted_observations[0].parent_observation_id is None

    assert trace.name == "dataset-run"
    assert sorted_observations[0].name == "dataset-run"
    assert trace.metadata == {"dataset_item_id": dataset_item_id, "run_name": run_name, "dataset_id": dataset.id}

    assert sorted_observations[0].metadata == {
        "dataset_item_id": dataset_item_id,
        "run_name": run_name,
        "dataset_id": dataset.id,
    }

    generations = list(filter(lambda obs: obs.type == "GENERATION", sorted_observations))

    assert len(generations) > 0
    for generation in generations:
        assert generation.input is not None
        assert generation.output is not None
        assert generation.input != ""
        assert generation.output != ""
        assert generation.total_tokens is not None
        assert generation.prompt_tokens is not None
        assert generation.completion_tokens is not None


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
