"""Comprehensive tests for Langfuse experiment functionality matching JS SDK."""

import os
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, TypeVar
from uuid import uuid4

import pytest
from opentelemetry import trace as otel_trace_api

from langfuse import get_client
from langfuse._client.attributes import LangfuseOtelSpanAttributes
from langfuse.api import LangfuseAPI
from langfuse.experiment import (
    Evaluation,
    ExperimentData,
    ExperimentItem,
    ExperimentItemResult,
)

T = TypeVar("T")

READ_TIMEOUT_SECONDS = float(os.environ.get("LANGFUSE_E2E_READ_TIMEOUT_SECONDS", "30"))
READ_INTERVAL_SECONDS = float(
    os.environ.get("LANGFUSE_E2E_READ_INTERVAL_SECONDS", "0.5")
)
ITEM_FIELDS = "core,dataset,io,metadata,experimentMetadata,scores"


def create_uuid() -> str:
    return str(uuid4())


def get_api() -> LangfuseAPI:
    return LangfuseAPI(
        username=os.environ.get("LANGFUSE_PUBLIC_KEY"),
        password=os.environ.get("LANGFUSE_SECRET_KEY"),
        base_url=os.environ.get("LANGFUSE_BASE_URL"),
    )


def _read_window_start() -> datetime:
    return datetime.now(timezone.utc) - timedelta(hours=1)


def poll(operation: Callable[[], T], is_ready: Callable[[T], bool]) -> T:
    """Re-read until ready or timeout; returns the last result for the caller to assert."""
    deadline = time.monotonic() + READ_TIMEOUT_SECONDS
    while True:
        result = operation()
        if is_ready(result) or time.monotonic() >= deadline:
            return result
        time.sleep(READ_INTERVAL_SECONDS)


def get_experiment_items(
    experiment_id: str, *, expected_count: int, with_scores: bool = False
) -> list:
    api = get_api()
    return poll(
        lambda: (
            api.experiments.list_items(
                from_start_time=_read_window_start(),
                experiment_id=experiment_id,
                fields=ITEM_FIELDS,
                limit=100,
            ).data
        ),
        lambda items: (
            len(items) >= expected_count
            and (not with_scores or all(item.scores for item in items))
        ),
    )


def get_experiment(
    experiment_id: str, *, is_ready: Callable[[Any], bool] = lambda _: True
):
    api = get_api()
    experiments = poll(
        lambda: (
            api.experiments.list(
                from_start_time=_read_window_start(),
                id=experiment_id,
                fields="core,metadata,scores",
            ).data
        ),
        lambda data: len(data) == 1 and is_ready(data[0]),
    )
    assert len(experiments) == 1, f"Experiment {experiment_id} should exist"
    return experiments[0]


def get_observations(trace_id: str, *, is_ready: Callable[[list], bool]) -> list:
    api = get_api()
    return poll(
        lambda: (
            api.observations.get_many(
                trace_id=trace_id,
                fields="core,basic,io,metadata",
                from_start_time=_read_window_start(),
            ).data
        ),
        is_ready,
    )


def get_trace_scores(trace_id: str, *, expected_count: int) -> list:
    api = get_api()
    return poll(
        lambda: api.scores_v3.get_many_v3(trace_id=trace_id, fields="subject").data,
        lambda scores: len(scores) >= expected_count,
    )


def has_score(name: str) -> Callable[[Any], bool]:
    return lambda experiment: any(s.name == name for s in experiment.scores or [])


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
    assert len(result.experiment_id) == 16
    assert result.experiment_url is not None
    assert result.experiment_url.endswith(
        f"/experiments/results?baseline={result.experiment_id}"
    )
    with pytest.warns(DeprecationWarning):
        assert result.dataset_run_id == result.experiment_id

    # Validate item results structure
    for item_result in result.item_results:
        assert hasattr(item_result, "output")
        assert hasattr(item_result, "evaluations")
        assert hasattr(item_result, "trace_id")
        assert item_result.experiment_id == result.experiment_id
        assert len(item_result.evaluations) == 2  # Both evaluators should run

    langfuse_client.flush()

    expected = {
        "Germany": ("Capital of Germany", "Berlin"),
        "France": ("Capital of France", "Paris"),
        "Spain": ("Capital of Spain", "Madrid"),
    }
    items = get_experiment_items(result.experiment_id, expected_count=3)
    assert len(items) == 3
    assert {item.trace_id for item in items} == {
        r.trace_id for r in result.item_results
    }

    for item in items:
        assert item.experiment_name == result.run_name
        assert item.experiment_dataset_id is None
        assert item.input in expected
        expected_output, expected_answer = expected[item.input]
        assert item.output == expected_output
        assert item.expected_output == expected_answer
        assert item.metadata is not None
        assert item.metadata["experiment_name"] == "Euro capitals"

    # Run-level evaluations are persisted for local data, too
    experiment = get_experiment(
        result.experiment_id, is_ready=has_score("average_length")
    )
    assert experiment.name == result.run_name
    assert experiment.description == "Country capital experiment"
    assert experiment.item_count == 3
    assert [s.name for s in experiment.scores or []] == ["average_length"]


def test_run_experiment_flattens_large_metadata_for_server_ingestion():
    """Server ingestion handles flattened experiment metadata on non-SDK child spans."""
    langfuse_client = get_client()
    external_tracer = otel_trace_api.get_tracer("ai.langfuse-python.e2e")
    external_span_name = "external-experiment-metadata-child-" + create_uuid()[:8]

    experiment_metadata = {
        "mode": "offline",
        "job_name": "agent-eval/PR-4",
        "build_url": "https://example.com/job/agent-eval-example/job/PR-4",
        "agent_name": "agent-eval-example",
    }

    def task_with_external_child(*, item: ExperimentItem, **kwargs: Dict[str, Any]):
        with external_tracer.start_as_current_span(external_span_name) as span:
            span.set_attribute("gen_ai.operation.name", "experiment-metadata-e2e")

        return "processed"

    result = langfuse_client.run_experiment(
        name="Flattened Experiment Metadata " + create_uuid()[:8],
        data=[{"input": "test input", "expected_output": "processed"}],
        task=task_with_external_child,
        metadata=experiment_metadata,
    )

    langfuse_client.flush()

    trace_id = result.item_results[0].trace_id
    assert trace_id is not None

    observations = get_observations(
        trace_id,
        is_ready=lambda observations: any(
            observation.name == external_span_name for observation in observations
        ),
    )

    root_observation = next(
        observation
        for observation in observations
        if observation.name == "experiment-item-run"
    )
    assert root_observation.metadata is not None
    for metadata_key, metadata_value in experiment_metadata.items():
        assert root_observation.metadata[metadata_key] == metadata_value

    external_observation = next(
        observation
        for observation in observations
        if observation.name == external_span_name
    )
    external_metadata = external_observation.metadata or {}

    assert not any(
        key == LangfuseOtelSpanAttributes.EXPERIMENT_METADATA
        or key.startswith(f"{LangfuseOtelSpanAttributes.EXPERIMENT_METADATA}.")
        for key in external_metadata
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

    project_id = langfuse_client._get_project_id()
    assert project_id is not None
    assert result.experiment_id == langfuse_client._create_experiment_id(
        project_id=project_id, dataset_id=dataset.id, run_name=result.run_name
    )
    assert result.experiment_url == (
        f"{os.environ['LANGFUSE_BASE_URL']}/project/{project_id}"
        f"/experiments/results?baseline={result.experiment_id}"
    )
    assert len(result.item_results) == 2
    assert all(
        item.experiment_id == result.experiment_id for item in result.item_results
    )

    langfuse_client.flush()

    expected_data = {"Germany": "Capital of Germany", "France": "Capital of France"}
    dataset_item_map = {item.id: item for item in dataset.items}

    items = get_experiment_items(
        result.experiment_id, expected_count=2, with_scores=True
    )
    assert len(items) == 2, "Experiment should have 2 items"
    assert {item.trace_id for item in items} == {
        r.trace_id for r in result.item_results
    }
    assert {item.experiment_item_id for item in items} == set(dataset_item_map)

    for item in items:
        assert item.experiment_name == result.run_name
        assert item.experiment_dataset_id == dataset.id
        assert item.experiment_description == "Test on Langfuse dataset"
        assert item.experiment_item_version is None

        dataset_item = dataset_item_map[item.experiment_item_id]
        assert item.input == dataset_item.input
        assert item.output == expected_data[dataset_item.input]
        assert item.expected_output == dataset_item.expected_output

        assert item.metadata is not None
        assert item.metadata["experiment_name"] == experiment_name
        assert item.metadata["dataset_id"] == dataset.id
        assert item.metadata["dataset_item_id"] == item.experiment_item_id

        assert [s.name for s in item.scores or []] == ["factuality"]

    experiment = get_experiment(
        result.experiment_id, is_ready=has_score("average_length")
    )
    assert experiment.name == result.run_name
    assert experiment.description == "Test on Langfuse dataset"
    assert experiment.dataset_id == dataset.id
    assert experiment.item_count == 2
    assert [s.name for s in experiment.scores or []] == ["average_length"]
    assert experiment.scores[0].subject.kind == "experiment"
    assert experiment.scores[0].subject.id == result.experiment_id


def test_run_experiment_on_versioned_dataset_records_item_version():
    """Pinned dataset versions are recorded as the experiment item version."""
    langfuse_client = get_client()
    dataset_name = "versioned-dataset-" + create_uuid()
    langfuse_client.create_dataset(name=dataset_name)
    langfuse_client.create_dataset_item(
        dataset_name=dataset_name, input="Germany", expected_output="Berlin"
    )

    version = datetime.now(timezone.utc).replace(microsecond=0) + timedelta(seconds=1)
    time.sleep(1.5)
    dataset = langfuse_client.get_dataset(dataset_name, version=version)
    assert len(dataset.items) == 1

    result = dataset.run_experiment(
        name="Versioned " + create_uuid()[:8], task=mock_task
    )
    langfuse_client.flush()

    items = get_experiment_items(result.experiment_id, expected_count=1)
    assert len(items) == 1
    assert items[0].experiment_item_version is not None
    assert items[0].experiment_item_version.astimezone(timezone.utc) == version


def test_same_run_name_on_dataset_reuses_experiment_id():
    """Re-running with the same run name on a dataset groups into one experiment."""
    langfuse_client = get_client()
    dataset_name = "rerun-dataset-" + create_uuid()
    langfuse_client.create_dataset(name=dataset_name)
    langfuse_client.create_dataset_item(
        dataset_name=dataset_name, input="Germany", expected_output="Berlin"
    )
    dataset = langfuse_client.get_dataset(dataset_name)
    run_name = "Shared run " + create_uuid()[:8]

    first = dataset.run_experiment(name="Rerun", run_name=run_name, task=mock_task)
    second = dataset.run_experiment(name="Rerun", run_name=run_name, task=mock_task)
    langfuse_client.flush()

    assert first.experiment_id == second.experiment_id
    items = get_experiment_items(first.experiment_id, expected_count=2)
    assert {item.trace_id for item in items} == {
        r.trace_id for r in first.item_results + second.item_results
    }


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

    assert len(result.item_results) == 1
    assert len(result.run_evaluations) == 1

    langfuse_client.flush()

    experiment = get_experiment(
        result.experiment_id, is_ready=has_score("persistence_run_test")
    )
    assert experiment.name == "Score persistence test"
    run_scores = {s.name: s for s in experiment.scores or []}
    assert run_scores["persistence_run_test"].value == 0.9

    trace_scores = get_trace_scores(result.item_results[0].trace_id, expected_count=1)
    assert [(s.name, s.value) for s in trace_scores] == [("persistence_test", 0.85)]
    assert trace_scores[0].subject.kind == "observation"


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

    assert result1.experiment_id != result2.experiment_id

    api = get_api()
    experiments = poll(
        lambda: (
            api.experiments.list(
                from_start_time=_read_window_start(), dataset_id=dataset.id
            ).data
        ),
        lambda data: len(data) >= 2,
    )
    assert {e.id: e.name for e in experiments} == {
        result1.experiment_id: "Experiment 1",
        result2.experiment_id: "Experiment 2",
    }


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

    langfuse_client.flush()

    # Verify scores are persisted via API with correct data types
    for i, item_result in enumerate(result.item_results):
        trace_id = item_result.trace_id
        assert trace_id is not None, f"Item {i} should have a trace_id"

        scores = get_trace_scores(trace_id, expected_count=1)
        assert len(scores) == 1
        assert scores[0].data_type == "BOOLEAN"
        assert scores[0].value is expected_results[i]

    experiment = get_experiment(
        result.experiment_id, is_ready=has_score("all_items_pass")
    )
    run_score = next(s for s in experiment.scores or [] if s.name == "all_items_pass")
    assert run_score.data_type == "BOOLEAN"
    assert run_score.value is False


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
