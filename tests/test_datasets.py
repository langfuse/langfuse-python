import json
import os
from langfuse import Langfuse
from langfuse.model import CreateDatasetItemRequest, InitialGeneration
from langfuse.model import CreateDatasetRequest


from tests.utils import create_uuid


def test_create_and_get_dataset():
    langfuse = Langfuse(os.environ.get("LF_PK"), os.environ.get("LF_SK"), os.environ.get("HOST"), debug=True)

    name = create_uuid()
    langfuse.create_dataset(CreateDatasetRequest(name=name))
    dataset = langfuse.get_dataset(name)
    assert dataset.name == name


def test_create_dataset_item():
    langfuse = Langfuse(os.environ.get("LF_PK"), os.environ.get("LF_SK"), os.environ.get("HOST"), debug=True)
    name = create_uuid()
    langfuse.create_dataset(CreateDatasetRequest(name=name))

    input = json.dumps({"input": "Hello World"})
    langfuse.create_dataset_item(CreateDatasetItemRequest(dataset_name=name, input=input))

    dataset = langfuse.get_dataset(name)
    assert len(dataset.items) == 1
    assert dataset.items[0].input == input


def test_linking_observation():
    langfuse = Langfuse(os.environ.get("LF_PK"), os.environ.get("LF_SK"), os.environ.get("HOST"), debug=True)

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
