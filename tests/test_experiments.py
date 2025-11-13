"""Comprehensive tests for Langfuse experiment functionality matching JS SDK."""

import time
from typing import Any, Dict, List

import pytest

from langfuse import get_client
from langfuse.experiment import (
    Evaluation,
    ExperimentData,
    ExperimentItem,
    ExperimentItemResult,
)
from tests.utils import create_uuid, get_api


@pytest.fixture
def sample_dataset():
    """Sample dataset for experiments."""
    return [
        {"input": "Germany", "expected_output": "Berlin"},
        {"input": "France", "expected_output": "Paris"},
        {"input": "Spain", "expected_output": "Madrid"},
    ]


def mock_task(*, item: ExperimentItem, **kwargs: Dict[str, Any]):
    """Mock task function that simulates processing."""
    input_val = (
        item.get("input")
        if isinstance(item, dict)
        else getattr(item, "input", "unknown")
    )
    return f"Capital of {input_val}"


def simple_evaluator(*, input, output, expected_output=None, **kwargs):
    """Return output length."""
    return Evaluation(name="length_check", value=len(output))


def factuality_evaluator(*, input, output, expected_output=None, **kwargs):
    """Mock factuality evaluator."""
    # Simple mock: check if expected output is in the output
    if expected_output and expected_output.lower() in output.lower():
        return Evaluation(name="factuality", value=1.0, comment="Correct answer found")
    return Evaluation(name="factuality", value=0.0, comment="Incorrect answer")


def run_evaluator_average_length(*, item_results: List[ExperimentItemResult], **kwargs):
    """Run evaluator that calculates average output length."""
    if not item_results:
        return Evaluation(name="average_length", value=0)

    avg_length = sum(len(r.output) for r in item_results) / len(item_results)

    return Evaluation(name="average_length", value=avg_length)


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
    assert len(result.item_results) == 3
    assert len(result.run_evaluations) == 1
    assert result.run_evaluations[0].name == "average_length"
    assert result.dataset_run_id is None  # No dataset_run_id for local datasets

    # Validate item results structure
    for item_result in result.item_results:
        assert hasattr(item_result, "output")
        assert hasattr(item_result, "evaluations")
        assert hasattr(item_result, "trace_id")
        assert (
            item_result.dataset_run_id is None
        )  # No dataset_run_id for local datasets
        assert len(item_result.evaluations) == 2  # Both evaluators should run

    # Flush and wait for server processing
    langfuse_client.flush()
    time.sleep(2)

    # Validate traces are correctly persisted with input/output/metadata
    api = get_api()
    expected_inputs = ["Germany", "France", "Spain"]
    expected_outputs = ["Capital of Germany", "Capital of France", "Capital of Spain"]

    for i, item_result in enumerate(result.item_results):
        trace_id = item_result.trace_id
        assert trace_id is not None, f"Item {i} should have a trace_id"

        # Fetch trace from API
        trace = api.trace.get(trace_id)
        assert trace is not None, f"Trace {trace_id} should exist"

        # Validate trace name
        assert trace.name == "experiment-item-run", (
            f"Trace {trace_id} should have correct name"
        )

        # Validate trace input - should contain the experiment item
        assert trace.input is not None, f"Trace {trace_id} should have input"
        expected_input = expected_inputs[i]
        # The input should contain the item data in some form
        assert expected_input in str(trace.input), (
            f"Trace {trace_id} input should contain '{expected_input}'"
        )

        # Validate trace output - should be the task result
        assert trace.output is not None, f"Trace {trace_id} should have output"
        expected_output = expected_outputs[i]
        assert trace.output == expected_output, (
            f"Trace {trace_id} output should be '{expected_output}', got '{trace.output}'"
        )

        # Validate trace metadata contains experiment name
        assert trace.metadata is not None, f"Trace {trace_id} should have metadata"
        assert "experiment_name" in trace.metadata, (
            f"Trace {trace_id} metadata should contain experiment_name"
        )
        assert trace.metadata["experiment_name"] == "Euro capitals", (
            f"Trace {trace_id} metadata should have correct experiment_name"
        )


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

    # Use unique experiment name for proper identification
    experiment_name = "Dataset Test " + create_uuid()[:8]
    result = dataset.run_experiment(
        name=experiment_name,
        description="Test on Langfuse dataset",
        task=mock_task,
        evaluators=[factuality_evaluator],
        run_evaluators=[run_evaluator_average_length],
    )

    # Should have dataset run ID for Langfuse datasets
    assert result.dataset_run_id is not None
    assert len(result.item_results) == 2
    assert all(item.dataset_run_id is not None for item in result.item_results)

    # Flush and wait for server processing
    langfuse_client.flush()
    time.sleep(3)

    # Verify dataset run exists via API
    api = get_api()
    dataset_run = api.datasets.get_run(
        dataset_name=dataset_name, run_name=result.run_name
    )

    # Validate traces are correctly persisted with input/output/metadata
    expected_data = {"Germany": "Capital of Germany", "France": "Capital of France"}
    dataset_run_id = result.dataset_run_id

    # Create a mapping from dataset item ID to dataset item for validation
    dataset_item_map = {item.id: item for item in dataset.items}

    for i, item_result in enumerate(result.item_results):
        trace_id = item_result.trace_id
        assert trace_id is not None, f"Item {i} should have a trace_id"

        # Fetch trace from API
        trace = api.trace.get(trace_id)
        assert trace is not None, f"Trace {trace_id} should exist"

        # Validate trace name
        assert trace.name == "experiment-item-run", (
            f"Trace {trace_id} should have correct name"
        )

        # Validate trace input and output match expected pairs
        assert trace.input is not None, f"Trace {trace_id} should have input"
        trace_input_str = str(trace.input)

        # Find which expected input this trace corresponds to
        matching_input = None
        for expected_input in expected_data.keys():
            if expected_input in trace_input_str:
                matching_input = expected_input
                break

        assert matching_input is not None, (
            f"Trace {trace_id} input '{trace_input_str}' should contain one of {list(expected_data.keys())}"
        )

        # Validate trace output matches the expected output for this input
        assert trace.output is not None, f"Trace {trace_id} should have output"
        expected_output = expected_data[matching_input]
        assert trace.output == expected_output, (
            f"Trace {trace_id} output should be '{expected_output}', got '{trace.output}'"
        )

        # Validate trace metadata contains experiment and dataset info
        assert trace.metadata is not None, f"Trace {trace_id} should have metadata"
        assert "experiment_name" in trace.metadata, (
            f"Trace {trace_id} metadata should contain experiment_name"
        )
        assert trace.metadata["experiment_name"] == experiment_name, (
            f"Trace {trace_id} metadata should have correct experiment_name"
        )

        # Validate dataset-specific metadata fields
        assert "dataset_id" in trace.metadata, (
            f"Trace {trace_id} metadata should contain dataset_id"
        )
        assert trace.metadata["dataset_id"] == dataset.id, (
            f"Trace {trace_id} metadata should have correct dataset_id"
        )

        assert "dataset_item_id" in trace.metadata, (
            f"Trace {trace_id} metadata should contain dataset_item_id"
        )
        # Get the dataset item ID from metadata and validate it exists
        dataset_item_id = trace.metadata["dataset_item_id"]
        assert dataset_item_id in dataset_item_map, (
            f"Trace {trace_id} metadata dataset_item_id should correspond to a valid dataset item"
        )

        # Validate the dataset item input matches the trace input
        dataset_item = dataset_item_map[dataset_item_id]
        assert dataset_item.input == matching_input, (
            f"Trace {trace_id} should correspond to dataset item with input '{matching_input}'"
        )

    assert dataset_run is not None, f"Dataset run {dataset_run_id} should exist"
    assert dataset_run.name == result.run_name, "Dataset run should have correct name"
    assert dataset_run.description == "Test on Langfuse dataset", (
        "Dataset run should have correct description"
    )

    # Get dataset run items to verify trace linkage
    dataset_run_items = api.dataset_run_items.list(
        dataset_id=dataset.id, run_name=result.run_name
    )
    assert len(dataset_run_items.data) == 2, "Dataset run should have 2 items"

    # Verify each dataset run item links to the correct trace
    run_item_trace_ids = {
        item.trace_id for item in dataset_run_items.data if item.trace_id
    }
    result_trace_ids = {item.trace_id for item in result.item_results}

    assert run_item_trace_ids == result_trace_ids, (
        f"Dataset run items should link to the same traces as experiment results. "
        f"Run items: {run_item_trace_ids}, Results: {result_trace_ids}"
    )


# Error Handling Tests
def test_evaluator_failures_handled_gracefully():
    """Test that evaluator failures don't break the experiment."""
    langfuse_client = get_client()

    def failing_evaluator(**kwargs):
        raise Exception("Evaluator failed")

    def working_evaluator(**kwargs):
        return Evaluation(name="working_eval", value=1.0)

    result = langfuse_client.run_experiment(
        name="Error test",
        data=[{"input": "test"}],
        task=lambda **kwargs: "result",
        evaluators=[working_evaluator, failing_evaluator],
    )

    # Should complete with only working evaluator
    assert len(result.item_results) == 1
    # Only the working evaluator should have produced results
    assert (
        len(
            [
                eval
                for eval in result.item_results[0].evaluations
                if eval.name == "working_eval"
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
    assert len(result.item_results) == 0

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
        task=lambda **kwargs: "result",
        run_evaluators=[failing_run_evaluator],
    )

    # Should complete but run evaluations should be empty
    assert len(result.item_results) == 1
    assert len(result.run_evaluations) == 0

    langfuse_client.flush()
    time.sleep(1)


# Edge Cases Tests
def test_empty_dataset_handling():
    """Test experiment with empty dataset."""
    langfuse_client = get_client()

    result = langfuse_client.run_experiment(
        name="Empty dataset test",
        data=[],
        task=lambda **kwargs: "result",
        run_evaluators=[run_evaluator_average_length],
    )

    assert len(result.item_results) == 0
    assert len(result.run_evaluations) == 1  # Run evaluators still execute

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
        task=lambda **kwargs: "result",
    )

    # Should handle missing fields gracefully
    assert len(result.item_results) == 2
    for item_result in result.item_results:
        assert hasattr(item_result, "trace_id")
        assert hasattr(item_result, "output")

    langfuse_client.flush()
    time.sleep(1)


def test_large_dataset_with_concurrency():
    """Test handling large dataset with concurrency control."""
    langfuse_client = get_client()

    large_dataset: ExperimentData = [
        {"input": f"Item {i}", "expected_output": f"Output {i}"} for i in range(20)
    ]

    result = langfuse_client.run_experiment(
        name="Large dataset test",
        data=large_dataset,
        task=lambda **kwargs: f"Processed {kwargs['item']}",
        evaluators=[lambda **kwargs: Evaluation(name="simple_eval", value=1.0)],
        max_concurrency=5,
    )

    assert len(result.item_results) == 20
    for item_result in result.item_results:
        assert len(item_result.evaluations) == 1
        assert hasattr(item_result, "trace_id")

    langfuse_client.flush()
    time.sleep(3)


# Evaluator Configuration Tests
def test_single_evaluation_return():
    """Test evaluators returning single evaluation instead of array."""
    langfuse_client = get_client()

    def single_evaluator(**kwargs):
        return Evaluation(name="single_eval", value=1, comment="Single evaluation")

    result = langfuse_client.run_experiment(
        name="Single evaluation test",
        data=[{"input": "test"}],
        task=lambda **kwargs: "result",
        evaluators=[single_evaluator],
    )

    assert len(result.item_results) == 1
    assert len(result.item_results[0].evaluations) == 1
    assert result.item_results[0].evaluations[0].name == "single_eval"

    langfuse_client.flush()
    time.sleep(1)


def test_no_evaluators():
    """Test experiment with no evaluators."""
    langfuse_client = get_client()

    result = langfuse_client.run_experiment(
        name="No evaluators test",
        data=[{"input": "test"}],
        task=lambda **kwargs: "result",
    )

    assert len(result.item_results) == 1
    assert len(result.item_results[0].evaluations) == 0
    assert len(result.run_evaluations) == 0

    langfuse_client.flush()
    time.sleep(1)


def test_only_run_evaluators():
    """Test experiment with only run evaluators."""
    langfuse_client = get_client()

    def run_only_evaluator(**kwargs):
        return Evaluation(
            name="run_only_eval", value=10, comment="Run-level evaluation"
        )

    result = langfuse_client.run_experiment(
        name="Only run evaluators test",
        data=[{"input": "test"}],
        task=lambda **kwargs: "result",
        run_evaluators=[run_only_evaluator],
    )

    assert len(result.item_results) == 1
    assert len(result.item_results[0].evaluations) == 0  # No item evaluations
    assert len(result.run_evaluations) == 1
    assert result.run_evaluations[0].name == "run_only_eval"

    langfuse_client.flush()
    time.sleep(1)


def test_different_data_types():
    """Test evaluators returning different data types."""
    langfuse_client = get_client()

    def number_evaluator(**kwargs):
        return Evaluation(name="number_eval", value=42)

    def string_evaluator(**kwargs):
        return Evaluation(name="string_eval", value="excellent")

    def boolean_evaluator(**kwargs):
        return Evaluation(name="boolean_eval", value=True)

    result = langfuse_client.run_experiment(
        name="Different data types test",
        data=[{"input": "test"}],
        task=lambda **kwargs: "result",
        evaluators=[number_evaluator, string_evaluator, boolean_evaluator],
    )

    evaluations = result.item_results[0].evaluations
    assert len(evaluations) == 3

    eval_by_name = {e.name: e.value for e in evaluations}
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
        return Evaluation(
            name="persistence_test",
            value=0.85,
            comment="Test evaluation for persistence",
        )

    def test_run_evaluator(**kwargs):
        return Evaluation(
            name="persistence_run_test",
            value=0.9,
            comment="Test run evaluation for persistence",
        )

    result = dataset.run_experiment(
        name="Score persistence test",
        run_name="Score persistence test",
        description="Test score persistence",
        task=mock_task,
        evaluators=[test_evaluator],
        run_evaluators=[test_run_evaluator],
    )

    assert result.dataset_run_id is not None
    assert len(result.item_results) == 1
    assert len(result.run_evaluations) == 1

    langfuse_client.flush()
    time.sleep(3)

    # Verify scores are persisted via API
    api = get_api()
    dataset_run = api.datasets.get_run(
        dataset_name=dataset_name, run_name=result.run_name
    )

    assert dataset_run.name == "Score persistence test"


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
        run_name="Experiment 1",
        description="First experiment",
        task=mock_task,
        evaluators=[factuality_evaluator],
    )

    langfuse_client.flush()
    time.sleep(2)

    # Run second experiment
    result2 = dataset.run_experiment(
        name="Experiment 2",
        run_name="Experiment 2",
        description="Second experiment",
        task=mock_task,
        evaluators=[simple_evaluator],
    )

    langfuse_client.flush()
    time.sleep(2)

    # Both experiments should have different run IDs
    assert result1.dataset_run_id is not None
    assert result2.dataset_run_id is not None
    assert result1.dataset_run_id != result2.dataset_run_id

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
        task=lambda **kwargs: f"Processed: {kwargs['item']}",
        evaluators=[simple_evaluator],
        run_evaluators=[run_evaluator_average_length],
    )

    # Basic validation that result structure is correct for formatting
    assert len(result.item_results) == 1
    assert len(result.run_evaluations) == 1
    assert hasattr(result.item_results[0], "trace_id")
    assert hasattr(result.item_results[0], "evaluations")

    langfuse_client.flush()
    time.sleep(1)


def test_boolean_score_types():
    """Test that BOOLEAN score types are properly ingested and persisted."""
    from langfuse.api import ScoreDataType

    langfuse_client = get_client()

    def boolean_evaluator(*, input, output, expected_output=None, **kwargs):
        """Boolean evaluator that checks if output contains the expected answer."""
        if not expected_output:
            return Evaluation(
                name="has_expected_content",
                value=False,
                data_type=ScoreDataType.BOOLEAN,
                comment="No expected output to check",
            )

        contains_expected = expected_output.lower() in str(output).lower()
        return Evaluation(
            name="has_expected_content",
            value=contains_expected,
            data_type=ScoreDataType.BOOLEAN,
            comment=f"Output {'contains' if contains_expected else 'does not contain'} expected content",
        )

    def boolean_run_evaluator(*, item_results: List[ExperimentItemResult], **kwargs):
        """Run evaluator that returns boolean based on all items passing."""
        if not item_results:
            return Evaluation(
                name="all_items_pass",
                value=False,
                data_type=ScoreDataType.BOOLEAN,
                comment="No items to evaluate",
            )

        # Check if all boolean evaluations are True
        all_pass = True
        for item_result in item_results:
            for evaluation in item_result.evaluations:
                if (
                    evaluation.name == "has_expected_content"
                    and evaluation.value is False
                ):
                    all_pass = False
                    break
            if not all_pass:
                break

        return Evaluation(
            name="all_items_pass",
            value=all_pass,
            data_type=ScoreDataType.BOOLEAN,
            comment=f"{'All' if all_pass else 'Not all'} items passed the boolean evaluation",
        )

    # Test data where some items should pass and some should fail
    test_data = [
        {"input": "What is the capital of Germany?", "expected_output": "Berlin"},
        {"input": "What is the capital of France?", "expected_output": "Paris"},
        {"input": "What is the capital of Spain?", "expected_output": "Madrid"},
    ]

    # Task that returns correct answers for Germany and France, but wrong for Spain
    def mock_task_with_boolean_results(*, item: ExperimentItem, **kwargs):
        input_val = (
            item.get("input")
            if isinstance(item, dict)
            else getattr(item, "input", "unknown")
        )
        input_str = str(input_val) if input_val is not None else ""

        if "Germany" in input_str:
            return "The capital is Berlin"
        elif "France" in input_str:
            return "The capital is Paris"
        else:
            return "I don't know the capital"

    result = langfuse_client.run_experiment(
        name="Boolean score type test",
        description="Test BOOLEAN data type in scores",
        data=test_data,
        task=mock_task_with_boolean_results,
        evaluators=[boolean_evaluator],
        run_evaluators=[boolean_run_evaluator],
    )

    # Validate basic result structure
    assert len(result.item_results) == 3
    assert len(result.run_evaluations) == 1

    # Validate individual item evaluations have boolean values
    expected_results = [
        True,
        True,
        False,
    ]  # Germany and France should pass, Spain should fail
    for i, item_result in enumerate(result.item_results):
        assert len(item_result.evaluations) == 1
        eval_result = item_result.evaluations[0]
        assert eval_result.name == "has_expected_content"
        assert isinstance(eval_result.value, bool)
        assert eval_result.value == expected_results[i]
        assert eval_result.data_type == ScoreDataType.BOOLEAN

    # Validate run evaluation is boolean and should be False (not all items passed)
    run_eval = result.run_evaluations[0]
    assert run_eval.name == "all_items_pass"
    assert isinstance(run_eval.value, bool)
    assert run_eval.value is False  # Spain should fail, so not all pass
    assert run_eval.data_type == ScoreDataType.BOOLEAN

    # Flush and wait for server processing
    langfuse_client.flush()
    time.sleep(3)

    # Verify scores are persisted via API with correct data types
    api = get_api()
    for i, item_result in enumerate(result.item_results):
        trace_id = item_result.trace_id
        assert trace_id is not None, f"Item {i} should have a trace_id"

        # Fetch trace from API to verify score persistence
        trace = api.trace.get(trace_id)
        assert trace is not None, f"Trace {trace_id} should exist"

        for score in trace.scores:
            assert score.data_type == "BOOLEAN"


def test_experiment_composite_evaluator_weighted_average():
    """Test composite evaluator in experiments that computes weighted average."""
    langfuse_client = get_client()

    def accuracy_evaluator(*, input, output, **kwargs):
        return Evaluation(name="accuracy", value=0.8)

    def relevance_evaluator(*, input, output, **kwargs):
        return Evaluation(name="relevance", value=0.9)

    def composite_evaluator(*, input, output, expected_output, metadata, evaluations):
        weights = {"accuracy": 0.6, "relevance": 0.4}
        total = sum(
            e.value * weights.get(e.name, 0)
            for e in evaluations
            if isinstance(e.value, (int, float))
        )

        return Evaluation(
            name="composite_score",
            value=total,
            comment=f"Weighted average of {len(evaluations)} metrics",
        )

    data = [
        {"input": "Test 1", "expected_output": "Output 1"},
        {"input": "Test 2", "expected_output": "Output 2"},
    ]

    result = langfuse_client.run_experiment(
        name=f"Composite Test {create_uuid()}",
        data=data,
        task=mock_task,
        evaluators=[accuracy_evaluator, relevance_evaluator],
        composite_evaluator=composite_evaluator,
    )

    # Verify results
    assert len(result.item_results) == 2

    for item_result in result.item_results:
        # Should have 3 evaluations: accuracy, relevance, and composite_score
        assert len(item_result.evaluations) == 3
        eval_names = [e.name for e in item_result.evaluations]
        assert "accuracy" in eval_names
        assert "relevance" in eval_names
        assert "composite_score" in eval_names

        # Check composite score value
        composite_eval = next(
            e for e in item_result.evaluations if e.name == "composite_score"
        )
        expected_value = 0.8 * 0.6 + 0.9 * 0.4  # 0.84
        assert abs(composite_eval.value - expected_value) < 0.001


def test_experiment_composite_evaluator_pass_fail():
    """Test composite evaluator in experiments that implements pass/fail logic."""

    langfuse_client = get_client()

    def metric1_evaluator(*, input, output, **kwargs):
        return Evaluation(name="metric1", value=0.9)

    def metric2_evaluator(*, input, output, **kwargs):
        return Evaluation(name="metric2", value=0.7)

    def pass_fail_composite(*, input, output, expected_output, metadata, evaluations):
        thresholds = {"metric1": 0.8, "metric2": 0.6}

        passes = all(
            e.value >= thresholds.get(e.name, 0)
            for e in evaluations
            if isinstance(e.value, (int, float))
        )

        return Evaluation(
            name="passes_all_checks",
            value=1.0 if passes else 0.0,
            comment="All checks passed" if passes else "Some checks failed",
        )

    data = [{"input": "Test 1"}]

    result = langfuse_client.run_experiment(
        name=f"Pass/Fail Composite Test {create_uuid()}",
        data=data,
        task=mock_task,
        evaluators=[metric1_evaluator, metric2_evaluator],
        composite_evaluator=pass_fail_composite,
    )

    # Verify composite evaluation
    assert len(result.item_results) == 1
    item_result = result.item_results[0]
    assert len(item_result.evaluations) == 3

    composite_eval = next(
        e for e in item_result.evaluations if e.name == "passes_all_checks"
    )
    assert composite_eval.value == 1.0
    assert composite_eval.comment == "All checks passed"


@pytest.mark.asyncio
async def test_experiment_async_composite_evaluator():
    """Test async composite evaluator in experiments."""
    import asyncio

    langfuse_client = get_client()

    def evaluator1(*, input, output, **kwargs):
        return Evaluation(name="eval1", value=0.8)

    async def async_composite(*, input, output, expected_output, metadata, evaluations):
        await asyncio.sleep(0.01)  # Simulate async processing
        avg = sum(
            e.value for e in evaluations if isinstance(e.value, (int, float))
        ) / len(evaluations)
        return Evaluation(name="async_composite", value=avg)

    data = [{"input": "Test 1"}]

    result = langfuse_client.run_experiment(
        name=f"Async Composite Test {create_uuid()}",
        data=data,
        task=mock_task,
        evaluators=[evaluator1],
        composite_evaluator=async_composite,
    )

    # Verify async composite evaluation
    assert len(result.item_results) == 1
    item_result = result.item_results[0]
    assert len(item_result.evaluations) == 2

    composite_eval = next(
        e for e in item_result.evaluations if e.name == "async_composite"
    )
    assert composite_eval.value == 0.8


def test_experiment_composite_evaluator_with_no_evaluations():
    """Test composite evaluator in experiments when no evaluations are present."""

    langfuse_client = get_client()

    def always_failing_evaluator(*, input, output, **kwargs):
        raise Exception("Always fails")

    def composite_evaluator(*, input, output, expected_output, metadata, evaluations):
        # Should not be called if no evaluations succeed
        return Evaluation(name="composite", value=0.0)

    data = [{"input": "Test 1"}]

    result = langfuse_client.run_experiment(
        name=f"No Evals Composite Test {create_uuid()}",
        data=data,
        task=mock_task,
        evaluators=[always_failing_evaluator],
        composite_evaluator=composite_evaluator,
    )

    # Composite evaluator should not run if no evaluations
    assert len(result.item_results) == 1
    item_result = result.item_results[0]
    # Should have no evaluations since the evaluator failed
    assert len(item_result.evaluations) == 0


def test_experiment_composite_evaluator_failure_handling():
    """Test that composite evaluator failures are handled gracefully in experiments."""

    langfuse_client = get_client()

    def evaluator1(*, input, output, **kwargs):
        return Evaluation(name="eval1", value=0.8)

    def failing_composite(*, input, output, expected_output, metadata, evaluations):
        raise ValueError("Composite evaluator failed")

    data = [{"input": "Test 1"}]

    result = langfuse_client.run_experiment(
        name=f"Failing Composite Test {create_uuid()}",
        data=data,
        task=mock_task,
        evaluators=[evaluator1],
        composite_evaluator=failing_composite,
    )

    # Regular evaluations should still be present
    assert len(result.item_results) == 1
    item_result = result.item_results[0]
    # Should only have the regular evaluation, not the composite one
    assert len(item_result.evaluations) == 1
    assert item_result.evaluations[0].name == "eval1"
