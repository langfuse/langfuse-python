"""Comprehensive tests for Langfuse experiment functionality matching JS SDK."""

import time

import pytest

from langfuse import get_client
from tests.utils import create_uuid, get_api


@pytest.fixture
def sample_dataset():
    """Sample dataset for experiments."""
    return [
        {"input": "Germany", "expected_output": "Berlin"},
        {"input": "France", "expected_output": "Paris"},
        {"input": "Spain", "expected_output": "Madrid"},
    ]


def mock_task(item):
    """Mock task function that simulates processing."""
    input_val = (
        item.get("input")
        if isinstance(item, dict)
        else getattr(item, "input", "unknown")
    )
    return f"Capital of {input_val}"


def simple_evaluator(*, input, output, expected_output=None, **kwargs):
    """Simple evaluator that returns output length."""
    return {"name": "length_check", "value": len(output)}


def factuality_evaluator(*, input, output, expected_output=None, **kwargs):
    """Mock factuality evaluator."""
    # Simple mock: check if expected output is in the output
    if expected_output and expected_output.lower() in output.lower():
        return {"name": "factuality", "value": 1.0, "comment": "Correct answer found"}
    return {"name": "factuality", "value": 0.0, "comment": "Incorrect answer"}


def run_evaluator_average_length(*, item_results, **kwargs):
    """Run evaluator that calculates average output length."""
    if not item_results:
        return {"name": "average_length", "value": 0}

    avg_length = sum(len(r["output"]) for r in item_results) / len(item_results)
    return {"name": "average_length", "value": avg_length}


# Basic Functionality Tests
def test_run_experiment_on_local_dataset(sample_dataset):
    """Test running experiment on local dataset."""
    langfuse_client = get_client()
    result = langfuse_client.run_experiment(
        name="Euro capitals",
        description="Country capital experiment",
        data=sample_dataset,
        task=mock_task,
        evaluators=[simple_evaluator, factuality_evaluator],
        run_evaluators=[run_evaluator_average_length],
    )

    # Validate basic result structure
    assert len(result["item_results"]) == 3
    assert len(result["run_evaluations"]) == 1
    assert result["run_evaluations"][0]["name"] == "average_length"
    assert result["dataset_run_id"] is None  # No dataset_run_id for local datasets

    # Validate item results structure
    for item_result in result["item_results"]:
        assert "output" in item_result
        assert "evaluations" in item_result
        assert "trace_id" in item_result
        assert (
            item_result["dataset_run_id"] is None
        )  # No dataset_run_id for local datasets
        assert len(item_result["evaluations"]) == 2  # Both evaluators should run

    # Flush and wait for server processing
    langfuse_client.flush()
    time.sleep(2)


def test_run_experiment_on_langfuse_dataset():
    """Test running experiment on Langfuse dataset."""
    langfuse_client = get_client()
    # Create dataset
    dataset_name = "test-dataset-" + create_uuid()
    langfuse_client.create_dataset(name=dataset_name)

    # Add items to dataset
    test_items = [
        {"input": "Germany", "expected_output": "Berlin"},
        {"input": "France", "expected_output": "Paris"},
    ]

    for item in test_items:
        langfuse_client.create_dataset_item(
            dataset_name=dataset_name,
            input=item["input"],
            expected_output=item["expected_output"],
        )

    # Get dataset and run experiment
    dataset = langfuse_client.get_dataset(dataset_name)

    result = dataset.run_experiment(
        name="Dataset Test",
        description="Test on Langfuse dataset",
        task=mock_task,
        evaluators=[factuality_evaluator],
        run_evaluators=[run_evaluator_average_length],
    )

    # Should have dataset run ID for Langfuse datasets
    assert result["dataset_run_id"] is not None
    assert len(result["item_results"]) == 2
    assert all(item["dataset_run_id"] is not None for item in result["item_results"])

    # Flush and wait for server processing
    langfuse_client.flush()
    time.sleep(3)

    # Verify dataset run exists via API
    api = get_api()
    runs = api.datasets.get_runs(dataset_name)
    assert len(runs.data) >= 1


# Error Handling Tests
def test_evaluator_failures_handled_gracefully():
    """Test that evaluator failures don't break the experiment."""
    langfuse_client = get_client()

    def failing_evaluator(**kwargs):
        raise Exception("Evaluator failed")

    def working_evaluator(**kwargs):
        return {"name": "working_eval", "value": 1.0}

    result = langfuse_client.run_experiment(
        name="Error test",
        data=[{"input": "test"}],
        task=lambda x: "result",
        evaluators=[working_evaluator, failing_evaluator],
    )

    # Should complete with only working evaluator
    assert len(result["item_results"]) == 1
    # Only the working evaluator should have produced results
    assert (
        len(
            [
                eval
                for eval in result["item_results"][0]["evaluations"]
                if eval["name"] == "working_eval"
            ]
        )
        == 1
    )

    langfuse_client.flush()
    time.sleep(1)


def test_task_failures_handled_gracefully():
    """Test that task failures are handled gracefully and don't stop the experiment."""
    langfuse_client = get_client()

    def failing_task(item):
        raise Exception("Task failed")

    def working_task(item):
        return f"Processed: {item['input']}"

    # Test with mixed data - some will fail, some will succeed
    result = langfuse_client.run_experiment(
        name="Task error test",
        data=[{"input": "test1"}, {"input": "test2"}],
        task=failing_task,
    )

    # Should complete but with no valid results since all tasks failed
    assert len(result["item_results"]) == 0

    langfuse_client.flush()
    time.sleep(1)


def test_run_evaluator_failures_handled():
    """Test that run evaluator failures don't break the experiment."""
    langfuse_client = get_client()

    def failing_run_evaluator(**kwargs):
        raise Exception("Run evaluator failed")

    result = langfuse_client.run_experiment(
        name="Run evaluator error test",
        data=[{"input": "test"}],
        task=lambda x: "result",
        run_evaluators=[failing_run_evaluator],
    )

    # Should complete but run evaluations should be empty
    assert len(result["item_results"]) == 1
    assert len(result["run_evaluations"]) == 0

    langfuse_client.flush()
    time.sleep(1)


# Edge Cases Tests
def test_empty_dataset_handling():
    """Test experiment with empty dataset."""
    langfuse_client = get_client()

    result = langfuse_client.run_experiment(
        name="Empty dataset test",
        data=[],
        task=lambda x: "result",
        run_evaluators=[run_evaluator_average_length],
    )

    assert len(result["item_results"]) == 0
    assert len(result["run_evaluations"]) == 1  # Run evaluators still execute

    langfuse_client.flush()
    time.sleep(1)


def test_dataset_with_missing_fields():
    """Test handling dataset with missing fields."""
    langfuse_client = get_client()

    incomplete_dataset = [
        {"input": "Germany"},  # Missing expected_output
        {"expected_output": "Paris"},  # Missing input
        {"input": "Spain", "expected_output": "Madrid"},  # Complete
    ]

    result = langfuse_client.run_experiment(
        name="Incomplete data test",
        data=incomplete_dataset,
        task=lambda x: "result",
    )

    # Should handle missing fields gracefully
    assert len(result["item_results"]) == 3
    for item_result in result["item_results"]:
        assert "trace_id" in item_result
        assert "output" in item_result

    langfuse_client.flush()
    time.sleep(1)


def test_large_dataset_with_concurrency():
    """Test handling large dataset with concurrency control."""
    langfuse_client = get_client()

    large_dataset = [
        {"input": f"Item {i}", "expected_output": f"Output {i}"} for i in range(20)
    ]

    result = langfuse_client.run_experiment(
        name="Large dataset test",
        data=large_dataset,
        task=lambda x: f"Processed {x['input']}",
        evaluators=[lambda **kwargs: {"name": "simple_eval", "value": 1.0}],
        max_concurrency=5,
    )

    assert len(result["item_results"]) == 20
    for item_result in result["item_results"]:
        assert len(item_result["evaluations"]) == 1
        assert "trace_id" in item_result

    langfuse_client.flush()
    time.sleep(3)


# Evaluator Configuration Tests
def test_single_evaluation_return():
    """Test evaluators returning single evaluation instead of array."""
    langfuse_client = get_client()

    def single_evaluator(**kwargs):
        return {"name": "single_eval", "value": 1, "comment": "Single evaluation"}

    result = langfuse_client.run_experiment(
        name="Single evaluation test",
        data=[{"input": "test"}],
        task=lambda x: "result",
        evaluators=[single_evaluator],
    )

    assert len(result["item_results"]) == 1
    assert len(result["item_results"][0]["evaluations"]) == 1
    assert result["item_results"][0]["evaluations"][0]["name"] == "single_eval"

    langfuse_client.flush()
    time.sleep(1)


def test_no_evaluators():
    """Test experiment with no evaluators."""
    langfuse_client = get_client()

    result = langfuse_client.run_experiment(
        name="No evaluators test",
        data=[{"input": "test"}],
        task=lambda x: "result",
        evaluators=[],
    )

    assert len(result["item_results"]) == 1
    assert len(result["item_results"][0]["evaluations"]) == 0
    assert len(result["run_evaluations"]) == 0

    langfuse_client.flush()
    time.sleep(1)


def test_only_run_evaluators():
    """Test experiment with only run evaluators."""
    langfuse_client = get_client()

    def run_only_evaluator(**kwargs):
        return {
            "name": "run_only_eval",
            "value": 10,
            "comment": "Run-level evaluation",
        }

    result = langfuse_client.run_experiment(
        name="Only run evaluators test",
        data=[{"input": "test"}],
        task=lambda x: "result",
        evaluators=[],
        run_evaluators=[run_only_evaluator],
    )

    assert len(result["item_results"]) == 1
    assert len(result["item_results"][0]["evaluations"]) == 0  # No item evaluations
    assert len(result["run_evaluations"]) == 1
    assert result["run_evaluations"][0]["name"] == "run_only_eval"

    langfuse_client.flush()
    time.sleep(1)


def test_different_data_types():
    """Test evaluators returning different data types."""
    langfuse_client = get_client()

    def number_evaluator(**kwargs):
        return {"name": "number_eval", "value": 42}

    def string_evaluator(**kwargs):
        return {"name": "string_eval", "value": "excellent"}

    def boolean_evaluator(**kwargs):
        return {"name": "boolean_eval", "value": True}

    result = langfuse_client.run_experiment(
        name="Different data types test",
        data=[{"input": "test"}],
        task=lambda x: "result",
        evaluators=[number_evaluator, string_evaluator, boolean_evaluator],
    )

    evaluations = result["item_results"][0]["evaluations"]
    assert len(evaluations) == 3

    eval_by_name = {e["name"]: e["value"] for e in evaluations}
    assert eval_by_name["number_eval"] == 42
    assert eval_by_name["string_eval"] == "excellent"
    assert eval_by_name["boolean_eval"] is True

    langfuse_client.flush()
    time.sleep(1)


# Data Persistence Tests
def test_scores_are_persisted():
    """Test that scores are properly persisted to the database."""
    langfuse_client = get_client()

    # Create dataset
    dataset_name = "score-persistence-" + create_uuid()
    langfuse_client.create_dataset(name=dataset_name)

    langfuse_client.create_dataset_item(
        dataset_name=dataset_name,
        input="Test input",
        expected_output="Test output",
    )

    dataset = langfuse_client.get_dataset(dataset_name)

    def test_evaluator(**kwargs):
        return {
            "name": "persistence_test",
            "value": 0.85,
            "comment": "Test evaluation for persistence",
        }

    def test_run_evaluator(**kwargs):
        return {
            "name": "persistence_run_test",
            "value": 0.9,
            "comment": "Test run evaluation for persistence",
        }

    result = dataset.run_experiment(
        name="Score persistence test",
        description="Test score persistence",
        task=mock_task,
        evaluators=[test_evaluator],
        run_evaluators=[test_run_evaluator],
    )

    assert result["dataset_run_id"] is not None
    assert len(result["item_results"]) == 1
    assert len(result["run_evaluations"]) == 1

    langfuse_client.flush()
    time.sleep(3)

    # Verify scores are persisted via API
    api = get_api()
    runs = api.datasets.get_runs(dataset_name)
    assert len(runs.data) >= 1

    # Verify the run exists with correct name
    run_names = [run.name for run in runs.data]
    assert "Score persistence test" in run_names


def test_multiple_experiments_on_same_dataset():
    """Test running multiple experiments on the same dataset."""
    langfuse_client = get_client()

    # Create dataset
    dataset_name = "multi-experiment-" + create_uuid()
    langfuse_client.create_dataset(name=dataset_name)

    for item in [
        {"input": "Germany", "expected_output": "Berlin"},
        {"input": "France", "expected_output": "Paris"},
    ]:
        langfuse_client.create_dataset_item(
            dataset_name=dataset_name,
            input=item["input"],
            expected_output=item["expected_output"],
        )

    dataset = langfuse_client.get_dataset(dataset_name)

    # Run first experiment
    result1 = dataset.run_experiment(
        name="Experiment 1",
        description="First experiment",
        task=mock_task,
        evaluators=[factuality_evaluator],
    )

    langfuse_client.flush()
    time.sleep(2)

    # Run second experiment
    result2 = dataset.run_experiment(
        name="Experiment 2",
        description="Second experiment",
        task=mock_task,
        evaluators=[simple_evaluator],
    )

    langfuse_client.flush()
    time.sleep(2)

    # Both experiments should have different run IDs
    assert result1["dataset_run_id"] is not None
    assert result2["dataset_run_id"] is not None
    assert result1["dataset_run_id"] != result2["dataset_run_id"]

    # Verify both runs exist in database
    api = get_api()
    runs = api.datasets.get_runs(dataset_name)
    assert len(runs.data) >= 2

    run_names = [run.name for run in runs.data]
    assert "Experiment 1" in run_names
    assert "Experiment 2" in run_names


# Result Formatting Tests
def test_format_experiment_results_basic():
    """Test basic result formatting functionality."""
    langfuse_client = get_client()

    result = langfuse_client.run_experiment(
        name="Formatting test",
        description="Test result formatting",
        data=[{"input": "Hello", "expected_output": "Hi"}],
        task=lambda x: f"Processed: {x['input']}",
        evaluators=[simple_evaluator],
        run_evaluators=[run_evaluator_average_length],
    )

    # Basic validation that result structure is correct for formatting
    assert len(result["item_results"]) == 1
    assert len(result["run_evaluations"]) == 1
    assert "trace_id" in result["item_results"][0]
    assert "evaluations" in result["item_results"][0]

    langfuse_client.flush()
    time.sleep(1)
