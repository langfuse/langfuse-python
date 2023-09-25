import json
import os

from langfuse import Langfuse
from langfuse.model import CreateDatasetItemRequest
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
