"""Langfuse experiment functionality for running and evaluating tasks on datasets.

This module provides the core experiment functionality for the Langfuse Python SDK,
allowing users to run experiments on datasets with automatic tracing, evaluation,
and result formatting.
"""

import asyncio
import logging
from typing import (
    TYPE_CHECKING,
    Any,
    Awaitable,
    Dict,
    List,
    Optional,
    Protocol,
    TypedDict,
    Union,
)

if TYPE_CHECKING:
    from langfuse._client.datasets import DatasetItemClient


class LocalExperimentItem(TypedDict, total=False):
    """Structure for experiment data items.

    Args:
        input: The input data to pass to the task function
        expected_output: Optional expected output for evaluation purposes
        metadata: Optional metadata for the experiment item
    """

    input: Any
    expected_output: Any
    metadata: Optional[Dict[str, Any]]


ExperimentItem = Union[LocalExperimentItem, DatasetItemClient]
ExperimentData = Union[List[LocalExperimentItem], List[DatasetItemClient]]


class Evaluation(TypedDict, total=False):
    """Structure for evaluation results.

    Args:
        name: Name of the evaluation metric
        value: The evaluation score/value (numeric or string)
        comment: Optional comment explaining the evaluation
        metadata: Optional metadata for the evaluation
    """

    name: str
    value: Union[int, float, str, bool]
    comment: Optional[str]
    metadata: Optional[Dict[str, Any]]


class ExperimentItemResult(TypedDict):
    """Result structure for individual experiment items.

    Args:
        item: The original experiment item that was processed
        output: The actual output produced by the task
        evaluations: List of evaluation results for this item
        trace_id: Langfuse trace ID for this item's execution
        dataset_run_id: Dataset run ID if this item was part of a Langfuse dataset
    """

    item: ExperimentItem
    output: Any
    evaluations: List[Evaluation]
    trace_id: Optional[str]
    dataset_run_id: Optional[str]


class ExperimentResult(TypedDict):
    """Complete result structure for experiment execution.

    Args:
        item_results: Results from processing each individual data item
        run_evaluations: Results from run-level evaluators
        dataset_run_id: ID of the dataset run (if using Langfuse datasets)
        dataset_run_url: URL to view the dataset run in Langfuse UI
    """

    item_results: List[ExperimentItemResult]
    run_evaluations: List[Evaluation]
    dataset_run_id: Optional[str]
    dataset_run_url: Optional[str]


class TaskFunction(Protocol):
    """Protocol for experiment task functions."""

    def __call__(
        self,
        *,
        item: ExperimentItem,
        **kwargs: Dict[str, Any],
    ) -> Union[Any, Awaitable[Any]]:
        """Execute the task on an experiment item.

        Args:
            item: The experiment or dataset item to process

        Returns:
            The task output (can be sync or async)
        """
        ...


class EvaluatorFunction(Protocol):
    """Protocol for item-level evaluator functions."""

    def __call__(
        self,
        *,
        input: Any,
        output: Any,
        expected_output: Any = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Dict[str, Any],
    ) -> Union[
        Evaluation, List[Evaluation], Awaitable[Union[Evaluation, List[Evaluation]]]
    ]:
        """Evaluate a task output.

        Args:
            input: The original input to the task
            output: The output produced by the task
            expected_output: The expected output (if available)
            metadata: Optional metadata from the experiment item

        Returns:
            Single evaluation or list of evaluations (can be sync or async)
        """
        ...


class RunEvaluatorFunction(Protocol):
    """Protocol for run-level evaluator functions."""

    def __call__(
        self,
        *,
        item_results: List[ExperimentItemResult],
        **kwargs: Dict[str, Any],
    ) -> Union[
        Evaluation, List[Evaluation], Awaitable[Union[Evaluation, List[Evaluation]]]
    ]:
        """Evaluate the entire experiment run.

        Args:
            item_results: Results from all processed experiment items

        Returns:
            Single evaluation or list of evaluations (can be sync or async)
        """
        ...


def format_experiment_results(
    item_results: List[ExperimentItemResult],
    run_evaluations: List[Evaluation],
    experiment_name: str,
    experiment_description: Optional[str] = None,
    dataset_run_url: Optional[str] = None,
    include_item_results: bool = False,
) -> str:
    """Format experiment results for display.

    Args:
        item_results: Results from processing each item
        run_evaluations: Results from run-level evaluators
        experiment_name: Name of the experiment
        experiment_description: Optional description of the experiment
        dataset_run_url: Optional URL to dataset run in Langfuse UI
        include_item_results: Whether to include individual item details

    Returns:
        Formatted string representation of the results
    """
    if not item_results:
        return "No experiment results to display."

    output = ""

    # Individual results
    if include_item_results:
        for i, result in enumerate(item_results):
            output += f"\n{i + 1}. Item {i + 1}:\n"

            # Input, expected, and actual
            item_input = None
            if isinstance(result["item"], dict):
                item_input = result["item"].get("input")
            elif hasattr(result["item"], "input"):
                item_input = result["item"].input

            if item_input is not None:
                output += f"   Input:    {_format_value(item_input)}\n"

            expected_output = None
            if isinstance(result["item"], dict):
                expected_output = result["item"].get("expected_output")
            elif hasattr(result["item"], "expected_output"):
                expected_output = result["item"].expected_output

            if expected_output is not None:
                output += f"   Expected: {_format_value(expected_output)}\n"
            output += f"   Actual:   {_format_value(result['output'])}\n"

            # Scores
            if result["evaluations"]:
                output += "   Scores:\n"
                for evaluation in result["evaluations"]:
                    score = evaluation["value"]
                    if isinstance(score, (int, float)):
                        score = f"{score:.3f}"
                    output += f"     • {evaluation['name']}: {score}"
                    if evaluation.get("comment"):
                        output += f"\n       💭 {evaluation['comment']}"
                    output += "\n"

            # Trace link
            if result.get("trace_id"):
                # Note: We'd need the langfuse client to generate the actual URL
                output += f"\n   Trace ID: {result['trace_id']}\n"
    else:
        output += f"Individual Results: Hidden ({len(item_results)} items)\n"
        output += "💡 Set include_item_results=True to view them\n"

    # Experiment Overview
    output += f"\n{'─' * 50}\n"
    output += f"📊 {experiment_name}"
    if experiment_description:
        output += f" - {experiment_description}"

    output += f"\n{len(item_results)} items"

    # Get unique evaluation names
    evaluation_names = set()
    for result in item_results:
        for evaluation in result["evaluations"]:
            evaluation_names.add(evaluation["name"])

    if evaluation_names:
        output += "\nEvaluations:"
        for eval_name in evaluation_names:
            output += f"\n  • {eval_name}"
        output += "\n"

    # Average scores
    if evaluation_names:
        output += "\nAverage Scores:"
        for eval_name in evaluation_names:
            scores = []
            for result in item_results:
                for evaluation in result["evaluations"]:
                    if evaluation["name"] == eval_name and isinstance(
                        evaluation["value"], (int, float)
                    ):
                        scores.append(evaluation["value"])

            if scores:
                avg = sum(scores) / len(scores)
                output += f"\n  • {eval_name}: {avg:.3f}"
        output += "\n"

    # Run evaluations
    if run_evaluations:
        output += "\nRun Evaluations:"
        for run_eval in run_evaluations:
            score = run_eval["value"]
            if isinstance(score, (int, float)):
                score = f"{score:.3f}"
            output += f"\n  • {run_eval['name']}: {score}"
            if run_eval.get("comment"):
                output += f"\n    💭 {run_eval['comment']}"
        output += "\n"

    if dataset_run_url:
        output += f"\n🔗 Dataset Run:\n   {dataset_run_url}"

    return output


def _format_value(value: Any) -> str:
    """Format a value for display."""
    if isinstance(value, str):
        return value[:50] + "..." if len(value) > 50 else value
    return str(value)


async def _run_evaluator(
    evaluator: Union[EvaluatorFunction, RunEvaluatorFunction], **kwargs: Any
) -> List[Evaluation]:
    """Run an evaluator function and normalize the result."""
    try:
        result = evaluator(**kwargs)

        # Handle async evaluators
        if asyncio.iscoroutine(result):
            result = await result

        # Normalize to list
        if isinstance(result, dict):
            return [result]

        elif isinstance(result, list):
            return result

        else:
            return []

    except Exception as e:
        evaluator_name = getattr(evaluator, "__name__", "unknown_evaluator")
        logging.getLogger("langfuse").error(f"Evaluator {evaluator_name} failed: {e}")
        return []


async def _run_task(task: TaskFunction, item: ExperimentItem) -> Any:
    """Run a task function and handle sync/async."""
    result = task(item=item)

    # Handle async tasks
    if asyncio.iscoroutine(result):
        result = await result

    return result
