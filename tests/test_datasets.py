import json
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Sequence

from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAI

from langfuse import Langfuse, observe
from langfuse.api.resources.commons.types.dataset_status import DatasetStatus
from langfuse.api.resources.commons.types.observation import Observation
from langfuse.langchain import CallbackHandler
from tests.utils import create_uuid, get_api


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


def test_dataset_run_with_metadata_and_description():
    langfuse = Langfuse(debug=False)

    dataset_name = create_uuid()
    langfuse.create_dataset(name=dataset_name)

    input = {"input": "Hello World"}
    langfuse.create_dataset_item(dataset_name=dataset_name, input=input)

    dataset = langfuse.get_dataset(dataset_name)
    assert len(dataset.items) == 1
    assert dataset.items[0].input == input

    run_name = create_uuid()

    for item in dataset.items:
        # Use run() with metadata and description
        with item.run(
            run_name=run_name,
            run_metadata={"key": "value"},
            run_description="This is a test run",
        ) as span:
            span.update_trace(name=run_name, metadata={"key": "value"})

    langfuse.flush()
    time.sleep(1)  # Give API time to process

    # Get trace using the API directly
    api = get_api()
    response = api.trace.list(name=run_name)

    assert response.data, "No traces found for the dataset run"
    trace = api.trace.get(response.data[0].id)

    assert trace.name == run_name
    assert trace.metadata is not None
    assert "key" in trace.metadata
    assert trace.metadata["key"] == "value"
    assert trace.id is not None


def test_get_dataset_runs():
    langfuse = Langfuse(debug=False)

    dataset_name = create_uuid()
    langfuse.create_dataset(name=dataset_name)

    input = {"input": "Hello World"}
    langfuse.create_dataset_item(dataset_name=dataset_name, input=input)

    dataset = langfuse.get_dataset(dataset_name)
    assert len(dataset.items) == 1
    assert dataset.items[0].input == input

    run_name_1 = create_uuid()

    for item in dataset.items:
        with item.run(
            run_name=run_name_1,
            run_metadata={"key": "value"},
            run_description="This is a test run",
        ):
            pass

    langfuse.flush()
    time.sleep(1)  # Give API time to process

    run_name_2 = create_uuid()

    for item in dataset.items:
        with item.run(
            run_name=run_name_2,
            run_metadata={"key": "value"},
            run_description="This is a test run",
        ):
            pass

    langfuse.flush()
    time.sleep(1)  # Give API time to process
    runs = langfuse.api.datasets.get_runs(dataset_name)

    assert len(runs.data) == 2
    assert runs.data[0].name == run_name_2
    assert runs.data[0].metadata == {"key": "value"}
    assert runs.data[0].description == "This is a test run"
    assert runs.data[1].name == run_name_1
    assert runs.meta.total_items == 2
    assert runs.meta.total_pages == 1
    assert runs.meta.page == 1
    assert runs.meta.limit == 50


def test_langchain_dataset():
    langfuse = Langfuse(debug=False)
    dataset_name = create_uuid()
    langfuse.create_dataset(name=dataset_name)

    input = json.dumps({"input": "Hello World"})
    langfuse.create_dataset_item(dataset_name=dataset_name, input=input)

    dataset = langfuse.get_dataset(dataset_name)

    run_name = create_uuid()

    dataset_item_id = None
    final_trace_id = None

    for item in dataset.items:
        # Run item with the Langchain model inside the context manager
        with item.run(run_name=run_name) as span:
            dataset_item_id = item.id
            final_trace_id = span.trace_id

            llm = OpenAI()
            template = """You are a playwright. Given the title of play, it is your job to write a synopsis for that title.
                Title: {title}
                Playwright: This is a synopsis for the above play:"""

            prompt_template = PromptTemplate(
                input_variables=["title"], template=template
            )
            chain = prompt_template | llm

            # Create an OpenAI generation as a nested
            handler = CallbackHandler()
            chain.invoke(
                "Tragedy at sunset on the beach", config={"callbacks": [handler]}
            )

    langfuse.flush()
    time.sleep(1)  # Give API time to process

    # Get the trace directly
    api = get_api()
    assert final_trace_id is not None, "No trace ID was created"
    trace = api.trace.get(final_trace_id)

    assert trace is not None
    assert len(trace.observations) >= 1

    # Update the sorted_dependencies function to handle ObservationsView
    def sorted_dependencies_from_trace(trace):
        parent_to_observation = {}
        for obs in trace.observations:
            # Filter out the generation that might leak in due to the monkey patching OpenAI integration
            # that might have run in the previous test suite. TODO: fix this hack
            if obs.name == "OpenAI-generation":
                continue

            parent_to_observation[obs.parent_observation_id] = obs

        # Start with the root observation (parent_observation_id is None)
        if None not in parent_to_observation:
            return []

        current_observation = parent_to_observation[None]
        dependencies = [current_observation]

        next_parent_id = current_observation.id
        while next_parent_id in parent_to_observation:
            current_observation = parent_to_observation[next_parent_id]
            dependencies.append(current_observation)
            next_parent_id = current_observation.id

        return dependencies

    sorted_observations = sorted_dependencies_from_trace(trace)

    if len(sorted_observations) >= 2:
        assert sorted_observations[0].id == sorted_observations[1].parent_observation_id
        assert sorted_observations[0].parent_observation_id is None

    assert trace.name == f"Dataset run: {run_name}"
    assert trace.metadata["dataset_item_id"] == dataset_item_id
    assert trace.metadata["run_name"] == run_name
    assert trace.metadata["dataset_id"] == dataset.id

    if len(sorted_observations) >= 2:
        assert sorted_observations[1].name == "RunnableSequence"
        assert sorted_observations[1].type == "CHAIN"
        assert sorted_observations[1].input is not None
        assert sorted_observations[1].output is not None
        assert sorted_observations[1].input != ""
        assert sorted_observations[1].output != ""


def sorted_dependencies(
    observations: Sequence[Observation],
):
    # observations have an id and a parent_observation_id. Return a sorted list starting with the root observation where the parent_observation_id is None
    parent_to_observation = {obs.parent_observation_id: obs for obs in observations}

    if None not in parent_to_observation:
        return []

    # Start with the root observation (parent_observation_id is None)
    current_observation = parent_to_observation[None]
    dependencies = [current_observation]

    next_parent_id = current_observation.id
    while next_parent_id in parent_to_observation:
        current_observation = parent_to_observation[next_parent_id]
        dependencies.append(current_observation)
        next_parent_id = current_observation.id

    return dependencies


def test_observe_dataset_run():
    # Create dataset
    langfuse = Langfuse()
    dataset_name = create_uuid()
    langfuse.create_dataset(name=dataset_name)

    items_data = []
    num_items = 3

    for i in range(num_items):
        trace_id = langfuse.create_trace_id()
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

    def execute_dataset_item(item, run_name):
        with item.run(run_name=run_name) as span:
            trace_id = span.trace_id
            span.update_trace(
                name="run_llm_app_on_dataset_item",
                input={"args": [item.input]},
                output=item.input,
            )
            wrapperFunc(item.input)
            return trace_id

    # Execute dataset items in parallel
    items = dataset.items[::-1]  # Reverse order to reflect input order
    trace_ids = []

    with ThreadPoolExecutor() as executor:
        for item in items:
            result = executor.submit(
                execute_dataset_item,
                item,
                run_name=run_name,
            )
            trace_ids.append(result.result())

    langfuse.flush()
    time.sleep(1)  # Give API time to process

    # Verify each trace individually
    api = get_api()
    for i, trace_id in enumerate(trace_ids):
        trace = api.trace.get(trace_id)
        assert trace is not None
        assert trace.name == "run_llm_app_on_dataset_item"
        assert trace.output is not None
        # Verify the input was properly captured
        expected_input = dataset.items[len(dataset.items) - 1 - i].input
        assert trace.input is not None
        assert "args" in trace.input
        assert trace.input["args"][0] == expected_input
        assert trace.output == expected_input


def test_dataset_runs_with_special_characters():
    """Test that dataset runs work correctly with special characters in names."""
    langfuse = Langfuse(debug=False)

    # Test with various special characters that need URL encoding
    dataset_name = f"test/dataset with spaces & special chars {create_uuid()[:5]}"
    run_name = f"run/name with % and # {create_uuid()[:5]}"

    langfuse.create_dataset(name=dataset_name)
    input_data = json.dumps({"input": "Test data"})
    langfuse.create_dataset_item(dataset_name=dataset_name, input=input_data)

    dataset = langfuse.get_dataset(dataset_name)
    assert len(dataset.items) == 1

    # Create a dataset run with special characters in the run name
    for item in dataset.items:
        with item.run(
            run_name=run_name,
            run_metadata={"test": "value"},
            run_description="Test run with special chars",
        ):
            pass

    langfuse.flush()
    time.sleep(1)

    # Test get_dataset_runs with special characters in dataset name
    runs = langfuse.get_dataset_runs(dataset_name=dataset_name)
    assert len(runs.data) == 1
    assert runs.data[0].name == run_name
    assert runs.data[0].metadata == {"test": "value"}
    assert runs.data[0].description == "Test run with special chars"

    # Test get_dataset_run with special characters in both dataset and run name
    run = langfuse.get_dataset_run(dataset_name=dataset_name, run_name=run_name)
    assert run.run_name == run_name
    assert run.dataset_name == dataset_name
    assert len(run.dataset_run_items) == 1

    # Test delete_dataset_run with special characters
    delete_response = langfuse.delete_dataset_run(
        dataset_name=dataset_name, run_name=run_name
    )
    assert delete_response.deleted_run_items_count == 1

    # Verify the run was deleted
    runs_after_delete = langfuse.get_dataset_runs(dataset_name=dataset_name)
    assert len(runs_after_delete.data) == 0
