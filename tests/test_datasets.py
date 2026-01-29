import time
from datetime import timedelta

from langfuse import Langfuse
from langfuse.api import DatasetStatus
from tests.utils import create_uuid


def test_create_and_get_dataset():
    langfuse = Langfuse(debug=False)

    name = "Text with spaces " + create_uuid()[:5]
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

    generation = langfuse.start_generation(name="test").end()
    langfuse.flush()

    input = {"input": "Hello World"}
    langfuse.create_dataset_item(dataset_name=name, input=input)
    langfuse.create_dataset_item(
        dataset_name=name,
        input=input,
        expected_output="Output",
        metadata={"key": "value"},
        source_observation_id=generation.id,
        source_trace_id=generation.trace_id,
    )
    langfuse.create_dataset_item(
        input="Hello",
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

    assert dataset.items[0].input == "Hello"
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

    # Instead, get all dataset items and find the one with matching ID
    dataset = langfuse.get_dataset(name)
    get_item = None
    for i in dataset.items:
        if i.id == item.id:
            get_item = i
            break

    assert get_item is not None
    assert get_item.input == input
    assert get_item.id == item.id
    assert get_item.expected_output == input

    new_input = {"input": "Hello World 2"}
    langfuse.create_dataset_item(
        dataset_name=name,
        input=new_input,
        id=item.id,
        expected_output=new_input,
        status=DatasetStatus.ARCHIVED,
    )

    # Refresh dataset and find updated item
    dataset = langfuse.get_dataset(name)
    get_new_item = None
    for i in dataset.items:
        if i.id == item.id:
            get_new_item = i
            break

    assert get_new_item is not None
    assert get_new_item.input == new_input
    assert get_new_item.id == item.id
    assert get_new_item.expected_output == new_input
    assert get_new_item.status == DatasetStatus.ARCHIVED


def test_run_experiment():
    """Test running an experiment on a dataset using run_experiment()."""
    langfuse = Langfuse(debug=False)

    dataset_name = create_uuid()
    langfuse.create_dataset(name=dataset_name)

    input_data = {"input": "Hello World"}
    langfuse.create_dataset_item(dataset_name=dataset_name, input=input_data)

    dataset = langfuse.get_dataset(dataset_name)
    assert len(dataset.items) == 1
    assert dataset.items[0].input == input_data

    run_name = create_uuid()

    def simple_task(*, item, **kwargs):
        return f"Processed: {item.input}"

    result = dataset.run_experiment(
        name=run_name,
        task=simple_task,
        metadata={"key": "value"},
    )

    langfuse.flush()
    time.sleep(1)  # Give API time to process

    assert result is not None
    assert len(result.item_results) == 1
    assert result.item_results[0].output == f"Processed: {input_data}"


def test_get_dataset_with_version():
    """Test that get_dataset correctly filters items by version timestamp."""

    langfuse = Langfuse(debug=False)

    # Create dataset
    name = create_uuid()
    langfuse.create_dataset(name=name)

    # Create first item
    item1 = langfuse.create_dataset_item(dataset_name=name, input={"version": "v1"})
    langfuse.flush()
    time.sleep(3)  # Ensure persistence

    # Fetch dataset to get the actual server-assigned timestamp of item1
    dataset_after_item1 = langfuse.get_dataset(name)
    assert len(dataset_after_item1.items) == 1
    item1_created_at = dataset_after_item1.items[0].created_at

    # Use a timestamp 1 second after item1's actual creation time
    query_timestamp = item1_created_at + timedelta(seconds=1)
    time.sleep(3)  # Ensure temporal separation

    # Create second item
    langfuse.create_dataset_item(dataset_name=name, input={"version": "v2"})
    langfuse.flush()
    time.sleep(3)  # Ensure persistence

    # Fetch at the query_timestamp (should only return first item)
    dataset = langfuse.get_dataset(name, version=query_timestamp)

    # Verify only first item is retrieved
    assert len(dataset.items) == 1
    assert dataset.items[0].input == {"version": "v1"}
    assert dataset.items[0].id == item1.id

    # Verify fetching without version returns both items (latest)
    dataset_latest = langfuse.get_dataset(name)
    assert len(dataset_latest.items) == 2


def test_run_experiment_with_versioned_dataset():
    """Test that running an experiment on a versioned dataset works correctly."""
    from datetime import timedelta
    import time

    langfuse = Langfuse(debug=False)

    # Create dataset
    name = create_uuid()
    langfuse.create_dataset(name=name)

    # Create first item
    langfuse.create_dataset_item(
        dataset_name=name, input={"question": "What is 2+2?"}, expected_output=4
    )
    langfuse.flush()
    time.sleep(3)

    # Fetch dataset to get the actual server-assigned timestamp of item1
    dataset_after_item1 = langfuse.get_dataset(name)
    assert len(dataset_after_item1.items) == 1
    item1_id = dataset_after_item1.items[0].id
    item1_created_at = dataset_after_item1.items[0].created_at

    # Use a timestamp 1 second after item1's creation
    version_timestamp = item1_created_at + timedelta(seconds=1)
    time.sleep(3)

    # Update item1 after the version timestamp (this should not affect versioned query)
    langfuse.create_dataset_item(
        id=item1_id,
        dataset_name=name,
        input={"question": "What is 4+4?"},
        expected_output=8,
    )
    langfuse.flush()
    time.sleep(3)

    # Create second item (after version timestamp)
    langfuse.create_dataset_item(
        dataset_name=name, input={"question": "What is 3+3?"}, expected_output=6
    )
    langfuse.flush()
    time.sleep(3)

    # Get versioned dataset (should only have first item with ORIGINAL state)
    versioned_dataset = langfuse.get_dataset(name, version=version_timestamp)
    assert len(versioned_dataset.items) == 1
    assert versioned_dataset.version == version_timestamp
    # Verify it returns the ORIGINAL version of item1 (before the update)
    assert versioned_dataset.items[0].input == {"question": "What is 2+2?"}
    assert versioned_dataset.items[0].expected_output == 4
    assert versioned_dataset.items[0].id == item1_id

    # Run a simple experiment on the versioned dataset
    def simple_task(*, item, **kwargs):
        # Just return a static answer
        return item.expected_output

    result = versioned_dataset.run_experiment(
        name="Versioned Dataset Test",
        description="Testing experiment with versioned dataset",
        task=simple_task,
    )

    # Verify experiment ran successfully
    assert result.name == "Versioned Dataset Test"
    assert len(result.item_results) == 1  # Only one item in versioned dataset
    assert result.item_results[0].output == 4
