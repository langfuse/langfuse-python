"""Batch evaluation functionality for Langfuse.

This module provides comprehensive batch evaluation capabilities for running evaluations
on observations fetched from Langfuse via the v2 observations API
(`GET /api/public/v2/observations`). It includes type definitions, protocols, result
classes, and the implementation for large-scale evaluation workflows with error
handling, retry logic, and resume capability.
"""

import asyncio
import json
import time
from typing import (
    TYPE_CHECKING,
    Any,
    Awaitable,
    Dict,
    List,
    Optional,
    Protocol,
    Tuple,
    Union,
)

from langfuse.api import ObservationV2
from langfuse.experiment import Evaluation, EvaluatorFunction
from langfuse.logger import langfuse_logger as logger

if TYPE_CHECKING:
    from langfuse._client.client import Langfuse

DEFAULT_BATCH_EVALUATION_FIELDS = "core,basic,io,metadata"
"""Default v2 observation field groups fetched for batch evaluation.

Includes ``io`` (raw ``input``/``output`` strings) and ``metadata`` so that mappers
work without extra configuration. Available groups: core, basic, time, io,
metadata, model, usage, prompt, metrics, trace_context.
"""


class EvaluatorInputs:
    """Input data structure for evaluators, returned by mapper functions.

    This class provides a strongly-typed container for transforming `ObservationV2`
    objects returned by the v2 observations API into the standardized format
    expected by evaluator functions.

    Attributes:
        input: The input data that was provided to generate the output being evaluated,
            for example the observation's input.
        output: The actual output that was produced and needs to be evaluated,
            for example the generation output or span result.
        expected_output: Optional ground truth or expected result for comparison.
            Used by evaluators to assess correctness. May be None if no ground truth
            is available for the entity being evaluated.
        metadata: Optional structured metadata providing additional context for evaluation.
            Can include information about the entity, execution context, user attributes,
            or any other relevant data that evaluators might use.

    Examples:
        Simple observation mapper:
        ```python
        from langfuse import EvaluatorInputs

        def simple_mapper(*, item):
            return EvaluatorInputs(
                input=item.input,  # raw string as returned by the API
                output=item.output,
                expected_output=None,  # No ground truth available
                metadata={"trace_id": item.trace_id, "user_id": item.user_id},
            )
        ```

        Mapper that decodes JSON input/output:
        ```python
        import json

        def observation_mapper(*, item):
            return EvaluatorInputs(
                input=json.loads(item.input) if item.input else None,
                output=json.loads(item.output) if item.output else None,
                expected_output=None,
                metadata={"observation_type": item.type, "name": item.name},
            )
        ```

    Note:
        All arguments must be passed as keywords when instantiating this class.
    """

    def __init__(
        self,
        *,
        input: Any,
        output: Any,
        expected_output: Any = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Initialize EvaluatorInputs with the provided data.

        Args:
            input: The input data for evaluation.
            output: The output data to be evaluated.
            expected_output: Optional ground truth for comparison.
            metadata: Optional additional context for evaluation.

        Note:
            All arguments must be provided as keywords.
        """
        self.input = input
        self.output = output
        self.expected_output = expected_output
        self.metadata = metadata


class MapperFunction(Protocol):
    """Protocol defining the interface for mapper functions in batch evaluation.

    Mapper functions transform `ObservationV2` objects from the v2 observations API
    into the standardized EvaluatorInputs format that evaluators expect.

    Mapper functions must:
    - Accept a single keyword argument `item` (an `ObservationV2`)
    - Return an EvaluatorInputs instance with input, output, expected_output, metadata
    - Can be either synchronous or asynchronous
    - Should handle missing or malformed data gracefully

    Notes on `ObservationV2`:
    - `input` and `output` are raw strings exactly as stored; the SDK does not
      parse them. Use `json.loads` in the mapper if you need structured data.
    - Only the requested field groups (see the `fields` argument of
      `Langfuse.run_batched_evaluation`) are populated; fields of other groups
      are None.
    - `metadata` values longer than 200 characters are truncated by the API.
    - Price fields (`input_price`, `output_price`, `total_price`) are strings.
    """

    def __call__(
        self,
        *,
        item: ObservationV2,
        **kwargs: Dict[str, Any],
    ) -> Union[EvaluatorInputs, Awaitable[EvaluatorInputs]]:
        """Transform an observation into evaluator inputs.

        This method defines how to extract evaluation-relevant data from the
        observation. The implementation should map its fields to the standardized
        input/output/expected_output/metadata structure.

        Args:
            item: The `ObservationV2` to transform.

        Returns:
            EvaluatorInputs: A structured container with:
                - input: The input data that generated the output
                - output: The output to be evaluated
                - expected_output: Optional ground truth for comparison
                - metadata: Optional additional context

            Can return either a direct EvaluatorInputs instance or an awaitable
            (for async mappers that need to fetch additional data).

        Examples:
            Basic observation mapper:
            ```python
            def map_basic(*, item):
                return EvaluatorInputs(
                    input=item.input,
                    output=item.output,
                    expected_output=None,
                    metadata={"trace_id": item.trace_id, "user": item.user_id}
                )
            ```

            Observation mapper with conditional logic:
            ```python
            import json

            def map_observation(*, item):
                if item.type == "GENERATION":
                    input_data = json.loads(item.input) if item.input else None
                else:
                    input_data = item.input

                return EvaluatorInputs(
                    input=input_data,
                    output=item.output,
                    expected_output=None,
                    metadata={"obs_id": item.id, "type": item.type}
                )
            ```

            Async mapper (if additional processing needed):
            ```python
            async def map_async(*, item):
                processed_output = await some_async_transformation(item.output)

                return EvaluatorInputs(
                    input=item.input,
                    output=processed_output,
                    expected_output=None,
                    metadata={"trace_id": item.trace_id}
                )
            ```
        """
        ...


class CompositeEvaluatorFunction(Protocol):
    """Protocol defining the interface for composite evaluator functions.

    Composite evaluators create aggregate scores from multiple item-level evaluations.
    This is commonly used to compute weighted averages, combined metrics, or other
    composite assessments based on individual evaluation results.

    Composite evaluators:
    - Accept the same inputs as item-level evaluators (input, output, expected_output, metadata)
      plus the list of evaluations
    - Return either a single Evaluation, a list of Evaluations, or a dict
    - Can be either synchronous or asynchronous
    - Have access to both raw item data and evaluation results
    """

    def __call__(
        self,
        *,
        input: Optional[Any] = None,
        output: Optional[Any] = None,
        expected_output: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None,
        evaluations: List[Evaluation],
        **kwargs: Dict[str, Any],
    ) -> Union[
        Evaluation,
        List[Evaluation],
        Dict[str, Any],
        Awaitable[Evaluation],
        Awaitable[List[Evaluation]],
        Awaitable[Dict[str, Any]],
    ]:
        r"""Create a composite evaluation from item-level evaluation results.

        This method combines multiple evaluation scores into a single composite metric.
        Common use cases include weighted averages, pass/fail decisions based on multiple
        criteria, or custom scoring logic that considers multiple dimensions.

        Args:
            input: The input data that was provided to the system being evaluated.
            output: The output generated by the system being evaluated.
            expected_output: The expected/reference output for comparison (if available).
            metadata: Additional metadata about the evaluation context.
            evaluations: List of evaluation results from item-level evaluators.
                Each evaluation contains name, value, comment, and metadata.

        Returns:
            Can return any of:
            - Evaluation: A single composite evaluation result
            - List[Evaluation]: Multiple composite evaluations
            - Dict: A dict that will be converted to an Evaluation
                - name: Identifier for the composite metric (e.g., "composite_score")
                - value: The computed composite value
                - comment: Optional explanation of how the score was computed
                - metadata: Optional details about the composition logic

            Can return either a direct Evaluation instance or an awaitable
            (for async composite evaluators).

        Examples:
            Simple weighted average:
            ```python
            def weighted_composite(*, input, output, expected_output, metadata, evaluations):
                weights = {
                    "accuracy": 0.5,
                    "relevance": 0.3,
                    "safety": 0.2
                }

                total_score = 0.0
                total_weight = 0.0

                for eval in evaluations:
                    if eval.name in weights and isinstance(eval.value, (int, float)):
                        total_score += eval.value * weights[eval.name]
                        total_weight += weights[eval.name]

                final_score = total_score / total_weight if total_weight > 0 else 0.0

                return Evaluation(
                    name="composite_score",
                    value=final_score,
                    comment=f"Weighted average of {len(evaluations)} metrics"
                )
            ```

            Pass/fail composite based on thresholds:
            ```python
            def pass_fail_composite(*, input, output, expected_output, metadata, evaluations):
                # Must pass all criteria
                thresholds = {
                    "accuracy": 0.7,
                    "safety": 0.9,
                    "relevance": 0.6
                }

                passes = True
                failing_metrics = []

                for metric, threshold in thresholds.items():
                    eval_result = next((e for e in evaluations if e.name == metric), None)
                    if eval_result and isinstance(eval_result.value, (int, float)):
                        if eval_result.value < threshold:
                            passes = False
                            failing_metrics.append(metric)

                return Evaluation(
                    name="passes_all_checks",
                    value=passes,
                    comment=f"Failed: {', '.join(failing_metrics)}" if failing_metrics else "All checks passed",
                    data_type="BOOLEAN"
                )
            ```

            Async composite with external scoring:
            ```python
            async def llm_composite(*, input, output, expected_output, metadata, evaluations):
                # Use LLM to synthesize multiple evaluation results
                eval_summary = "\n".join(
                    f"- {e.name}: {e.value}" for e in evaluations
                )

                prompt = f"Given these evaluation scores:\n{eval_summary}\n"
                prompt += f"For the output: {output}\n"
                prompt += "Provide an overall quality score from 0-1."

                response = await openai.chat.completions.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": prompt}]
                )

                score = float(response.choices[0].message.content.strip())

                return Evaluation(
                    name="llm_composite_score",
                    value=score,
                    comment="LLM-synthesized composite score"
                )
            ```

            Context-aware composite:
            ```python
            def context_composite(*, input, output, expected_output, metadata, evaluations):
                # Adjust weighting based on metadata
                base_weights = {"accuracy": 0.5, "speed": 0.3, "cost": 0.2}

                # If metadata indicates high importance, prioritize accuracy
                if metadata and metadata.get('importance') == 'high':
                    weights = {"accuracy": 0.7, "speed": 0.2, "cost": 0.1}
                else:
                    weights = base_weights

                total = sum(
                    e.value * weights.get(e.name, 0)
                    for e in evaluations
                    if isinstance(e.value, (int, float))
                )

                return Evaluation(
                    name="weighted_composite",
                    value=total,
                    comment="Context-aware weighted composite"
                )
            ```
        """
        ...


class EvaluatorStats:
    """Statistics for a single evaluator's performance during batch evaluation.

    This class tracks detailed metrics about how a specific evaluator performed
    across all items in a batch evaluation run. It helps identify evaluator issues,
    understand reliability, and optimize evaluation pipelines.

    Attributes:
        name: The name of the evaluator function (extracted from __name__).
        total_runs: Total number of times the evaluator was invoked.
        successful_runs: Number of times the evaluator completed successfully.
        failed_runs: Number of times the evaluator raised an exception or failed.
        total_scores_created: Total number of evaluation scores created by this evaluator.
            Can be higher than successful_runs if the evaluator returns multiple scores.

    Examples:
        Accessing evaluator stats from batch evaluation result:
        ```python
        result = client.run_batched_evaluation(...)

        for stats in result.evaluator_stats:
            print(f"Evaluator: {stats.name}")
            print(f"  Success rate: {stats.successful_runs / stats.total_runs:.1%}")
            print(f"  Scores created: {stats.total_scores_created}")

            if stats.failed_runs > 0:
                print(f"  ⚠️  Failed {stats.failed_runs} times")
        ```

        Identifying problematic evaluators:
        ```python
        result = client.run_batched_evaluation(...)

        # Find evaluators with high failure rates
        for stats in result.evaluator_stats:
            failure_rate = stats.failed_runs / stats.total_runs
            if failure_rate > 0.1:  # More than 10% failures
                print(f"⚠️  {stats.name} has {failure_rate:.1%} failure rate")
                print(f"    Consider debugging or removing this evaluator")
        ```

    Note:
        All arguments must be passed as keywords when instantiating this class.
    """

    def __init__(
        self,
        *,
        name: str,
        total_runs: int = 0,
        successful_runs: int = 0,
        failed_runs: int = 0,
        total_scores_created: int = 0,
    ):
        """Initialize EvaluatorStats with the provided metrics.

        Args:
            name: The evaluator function name.
            total_runs: Total number of evaluator invocations.
            successful_runs: Number of successful completions.
            failed_runs: Number of failures.
            total_scores_created: Total scores created by this evaluator.

        Note:
            All arguments must be provided as keywords.
        """
        self.name = name
        self.total_runs = total_runs
        self.successful_runs = successful_runs
        self.failed_runs = failed_runs
        self.total_scores_created = total_scores_created


class BatchEvaluationResumeToken:
    """Token for resuming an interrupted or limited batch evaluation run.

    The v2 observations API returns observations ordered by start time, newest
    first, and paginates with an opaque cursor. The token stores the cursor of
    the next page that has not been processed yet, so a resumed run continues
    exactly where the previous one stopped, without re-evaluating or skipping
    items, even if new observations were ingested in the meantime.

    A token is returned when a run stops because a batch fetch failed after all
    retries (`completed=False`) or because `max_items` was reached while more
    items exist (`has_more_items=True`).

    Attributes:
        filter: The original JSON filter string used to query items. Pass the
            same filter when resuming.
        cursor: Cursor of the next page to fetch. None if no page was fetched yet.
        last_processed_timestamp: ISO 8601 start time of the oldest processed
            observation. Only used to resume when `cursor` is None, by fetching
            observations that started strictly before this timestamp.
        last_processed_id: The ID of the last processed observation, for reference.
        items_processed: Number of items successfully processed so far, including
            items processed by the runs this token was resumed from.

    Examples:
        Resuming a run that stopped early:
        ```python
        result = client.run_batched_evaluation(
            mapper=my_mapper,
            evaluators=[evaluator1, evaluator2],
            filter=my_filter,
            max_items=10000,
        )

        if result.resume_token:
            result = client.run_batched_evaluation(
                mapper=my_mapper,
                evaluators=[evaluator1, evaluator2],
                filter=my_filter,
                resume_from=result.resume_token,
            )
        ```

        Persisting a token between processes:
        ```python
        import json

        token = result.resume_token
        with open("resume_token.json", "w") as f:
            json.dump(vars(token), f)

        with open("resume_token.json") as f:
            token = BatchEvaluationResumeToken(**json.load(f))
        ```

    Note:
        All arguments must be passed as keywords when instantiating this class.
    """

    def __init__(
        self,
        *,
        filter: Optional[str],
        last_processed_timestamp: str,
        last_processed_id: str,
        items_processed: int,
        cursor: Optional[str] = None,
    ):
        """Initialize BatchEvaluationResumeToken with the provided state.

        Args:
            filter: The original JSON filter string.
            last_processed_timestamp: ISO 8601 start time of the oldest processed item.
            last_processed_id: ID of last processed item.
            items_processed: Count of items processed before interruption.
            cursor: Cursor of the next page to fetch.

        Note:
            All arguments must be provided as keywords.
        """
        self.filter = filter
        self.cursor = cursor
        self.last_processed_timestamp = last_processed_timestamp
        self.last_processed_id = last_processed_id
        self.items_processed = items_processed


class BatchEvaluationResult:
    r"""Complete result structure for batch evaluation execution.

    This class encapsulates comprehensive statistics and metadata about a batch
    evaluation run, including counts, evaluator-specific metrics, timing information,
    error details, and resume capability.

    Attributes:
        total_items_fetched: Total number of items fetched from the API.
        total_items_processed: Number of items successfully evaluated.
        total_items_failed: Number of items that failed during evaluation.
        total_scores_created: Total scores created by all item-level evaluators.
        total_composite_scores_created: Scores created by the composite evaluator.
        total_evaluations_failed: Number of individual evaluator failures across all items.
        evaluator_stats: List of per-evaluator statistics (success/failure rates, scores created).
        resume_token: Token for continuing the run. Set when a batch fetch failed
            (`completed=False`) or when `max_items` was reached while more items
            exist (`has_more_items=True`); None otherwise.
        completed: False if the run stopped because a batch fetch failed, True otherwise
            (including when it stopped at `max_items`).
        duration_seconds: Total time taken to execute the batch evaluation.
        failed_item_ids: List of observation IDs for items that failed evaluation.
        error_summary: Dictionary mapping error types to occurrence counts.
        has_more_items: True if max_items limit was reached but more items exist.
        item_evaluations: Dictionary mapping observation IDs to their evaluation results (both regular and composite).

    Examples:
        Basic result inspection:
        ```python
        result = client.run_batched_evaluation(...)

        print(f"Processed: {result.total_items_processed}/{result.total_items_fetched}")
        print(f"Scores created: {result.total_scores_created}")
        print(f"Duration: {result.duration_seconds:.2f}s")
        print(f"Success rate: {result.total_items_processed / result.total_items_fetched:.1%}")
        ```

        Detailed analysis with evaluator stats:
        ```python
        result = client.run_batched_evaluation(...)

        print(f"\n📊 Batch Evaluation Results")
        print(f"{'='*50}")
        print(f"Items processed: {result.total_items_processed}")
        print(f"Items failed: {result.total_items_failed}")
        print(f"Scores created: {result.total_scores_created}")

        if result.total_composite_scores_created > 0:
            print(f"Composite scores: {result.total_composite_scores_created}")

        print(f"\n📈 Evaluator Performance:")
        for stats in result.evaluator_stats:
            success_rate = stats.successful_runs / stats.total_runs if stats.total_runs > 0 else 0
            print(f"\n  {stats.name}:")
            print(f"    Success rate: {success_rate:.1%}")
            print(f"    Scores created: {stats.total_scores_created}")
            if stats.failed_runs > 0:
                print(f"    ⚠️  Failures: {stats.failed_runs}")

        if result.error_summary:
            print(f"\n⚠️  Errors encountered:")
            for error_type, count in result.error_summary.items():
                print(f"    {error_type}: {count}")
        ```

        Handling incomplete runs:
        ```python
        result = client.run_batched_evaluation(...)

        if not result.completed:
            print("⚠️  Evaluation incomplete!")

            if result.resume_token:
                print(f"Processed {result.resume_token.items_processed} items before failure")
                print("Pass it as resume_from to continue")

        if result.has_more_items:
            print(f"ℹ️  More items available beyond max_items limit")
        ```

        Performance monitoring:
        ```python
        result = client.run_batched_evaluation(...)

        items_per_second = result.total_items_processed / result.duration_seconds
        avg_scores_per_item = result.total_scores_created / result.total_items_processed

        print(f"Performance metrics:")
        print(f"  Throughput: {items_per_second:.2f} items/second")
        print(f"  Avg scores/item: {avg_scores_per_item:.2f}")
        print(f"  Total duration: {result.duration_seconds:.2f}s")

        if result.total_evaluations_failed > 0:
            failure_rate = result.total_evaluations_failed / (
                result.total_items_processed * len(result.evaluator_stats)
            )
            print(f"  Evaluation failure rate: {failure_rate:.1%}")
        ```

    Note:
        All arguments must be passed as keywords when instantiating this class.
    """

    def __init__(
        self,
        *,
        total_items_fetched: int,
        total_items_processed: int,
        total_items_failed: int,
        total_scores_created: int,
        total_composite_scores_created: int,
        total_evaluations_failed: int,
        evaluator_stats: List[EvaluatorStats],
        resume_token: Optional[BatchEvaluationResumeToken],
        completed: bool,
        duration_seconds: float,
        failed_item_ids: List[str],
        error_summary: Dict[str, int],
        has_more_items: bool,
        item_evaluations: Dict[str, List["Evaluation"]],
    ):
        """Initialize BatchEvaluationResult with comprehensive statistics.

        Args:
            total_items_fetched: Total items fetched from API.
            total_items_processed: Items successfully evaluated.
            total_items_failed: Items that failed evaluation.
            total_scores_created: Scores from item-level evaluators.
            total_composite_scores_created: Scores from composite evaluator.
            total_evaluations_failed: Individual evaluator failures.
            evaluator_stats: Per-evaluator statistics.
            resume_token: Token for resuming (None if completed).
            completed: Whether all items were processed.
            duration_seconds: Total execution time.
            failed_item_ids: IDs of failed items.
            error_summary: Error types and counts.
            has_more_items: Whether more items exist beyond max_items.
            item_evaluations: Dictionary mapping item IDs to their evaluation results.

        Note:
            All arguments must be provided as keywords.
        """
        self.total_items_fetched = total_items_fetched
        self.total_items_processed = total_items_processed
        self.total_items_failed = total_items_failed
        self.total_scores_created = total_scores_created
        self.total_composite_scores_created = total_composite_scores_created
        self.total_evaluations_failed = total_evaluations_failed
        self.evaluator_stats = evaluator_stats
        self.resume_token = resume_token
        self.completed = completed
        self.duration_seconds = duration_seconds
        self.failed_item_ids = failed_item_ids
        self.error_summary = error_summary
        self.has_more_items = has_more_items
        self.item_evaluations = item_evaluations

    def __str__(self) -> str:
        """Return a formatted string representation of the batch evaluation results.

        Returns:
            A multi-line string with a summary of the evaluation results.
        """
        lines = []
        lines.append("=" * 60)
        lines.append("Batch Evaluation Results")
        lines.append("=" * 60)

        # Summary statistics
        lines.append(f"\nStatus: {'Completed' if self.completed else 'Incomplete'}")
        lines.append(f"Duration: {self.duration_seconds:.2f}s")
        lines.append(f"\nItems fetched: {self.total_items_fetched}")
        lines.append(f"Items processed: {self.total_items_processed}")

        if self.total_items_failed > 0:
            lines.append(f"Items failed: {self.total_items_failed}")

        # Success rate
        if self.total_items_fetched > 0:
            success_rate = self.total_items_processed / self.total_items_fetched * 100
            lines.append(f"Success rate: {success_rate:.1f}%")

        # Scores created
        lines.append(f"\nScores created: {self.total_scores_created}")
        if self.total_composite_scores_created > 0:
            lines.append(f"Composite scores: {self.total_composite_scores_created}")

        total_scores = self.total_scores_created + self.total_composite_scores_created
        lines.append(f"Total scores: {total_scores}")

        # Evaluator statistics
        if self.evaluator_stats:
            lines.append("\nEvaluator Performance:")
            for stats in self.evaluator_stats:
                lines.append(f"  {stats.name}:")
                if stats.total_runs > 0:
                    success_rate = (
                        stats.successful_runs / stats.total_runs * 100
                        if stats.total_runs > 0
                        else 0
                    )
                    lines.append(
                        f"    Runs: {stats.successful_runs}/{stats.total_runs} "
                        f"({success_rate:.1f}% success)"
                    )
                    lines.append(f"    Scores created: {stats.total_scores_created}")
                    if stats.failed_runs > 0:
                        lines.append(f"    Failed runs: {stats.failed_runs}")

        # Performance metrics
        if self.total_items_processed > 0 and self.duration_seconds > 0:
            items_per_sec = self.total_items_processed / self.duration_seconds
            lines.append("\nPerformance:")
            lines.append(f"  Throughput: {items_per_sec:.2f} items/second")
            if self.total_scores_created > 0:
                avg_scores = self.total_scores_created / self.total_items_processed
                lines.append(f"  Avg scores per item: {avg_scores:.2f}")

        # Errors and warnings
        if self.error_summary:
            lines.append("\nErrors encountered:")
            for error_type, count in self.error_summary.items():
                lines.append(f"  {error_type}: {count}")

        # Incomplete run information
        if not self.completed:
            lines.append("\nWarning: Evaluation incomplete")
            if self.resume_token:
                lines.append(
                    f"  Last processed: {self.resume_token.last_processed_timestamp}"
                )
                lines.append(f"  Items processed: {self.resume_token.items_processed}")
                lines.append("  Use resume_from parameter to continue")

        if self.has_more_items:
            lines.append("\nNote: More items available beyond max_items limit")

        lines.append("=" * 60)
        return "\n".join(lines)


class BatchEvaluationRunner:
    """Handles batch evaluation execution for a Langfuse client.

    This class encapsulates all the logic for fetching items, running evaluators,
    creating scores, and managing the evaluation lifecycle. It provides a clean
    separation of concerns from the main Langfuse client class.

    The runner uses a streaming/pipeline approach to process items in batches,
    avoiding loading the entire dataset into memory. This makes it suitable for
    evaluating large numbers of items.

    Attributes:
        client: The Langfuse client instance used for API calls and score creation.
    """

    def __init__(self, client: "Langfuse"):
        """Initialize the batch evaluation runner.

        Args:
            client: The Langfuse client instance.
        """
        self.client = client

    async def run_async(
        self,
        *,
        mapper: MapperFunction,
        evaluators: List[EvaluatorFunction],
        filter: Optional[str] = None,
        fetch_batch_size: int = 50,
        fields: Optional[str] = DEFAULT_BATCH_EVALUATION_FIELDS,
        max_items: Optional[int] = None,
        max_concurrency: int = 5,
        composite_evaluator: Optional[CompositeEvaluatorFunction] = None,
        metadata: Optional[Dict[str, Any]] = None,
        _add_observation_scores_to_trace: bool = False,
        max_retries: int = 3,
        verbose: bool = False,
        resume_from: Optional[BatchEvaluationResumeToken] = None,
    ) -> BatchEvaluationResult:
        """Run batch evaluation asynchronously.

        This is the main implementation method that orchestrates the entire batch
        evaluation process: fetching items from `GET /api/public/v2/observations`
        with cursor pagination, mapping, evaluating, creating scores, and tracking
        statistics.

        Args:
            mapper: Function to transform `ObservationV2` items to evaluator inputs.
            evaluators: List of evaluation functions to run on each item.
            filter: JSON filter string (v2 observations filter schema).
            fetch_batch_size: Number of items to fetch per API call (max 1000).
            fields: Comma-separated v2 observation field groups to fetch.
            max_items: Maximum number of items to process (None = all).
            max_concurrency: Maximum number of concurrent evaluations.
            composite_evaluator: Optional function to create composite scores.
            metadata: Metadata to add to all created scores.
            _add_observation_scores_to_trace: Private option to duplicate
                observation-level scores onto the parent trace.
            max_retries: Maximum retries for failed batch fetches.
            verbose: If True, log progress to console.
            resume_from: Resume token from a previous run. If `filter` is omitted,
                the token's filter is reused.

        Returns:
            BatchEvaluationResult with comprehensive statistics.

        Raises:
            ValueError: If the filter is not a JSON array, or the resume token was
                created for a different filter.
        """
        start_time = time.time()

        # The token's cursor only continues the query that produced it.
        if resume_from is not None:
            if filter is None:
                filter = resume_from.filter
            elif filter != resume_from.filter:
                raise ValueError(
                    "Resume token was created for a different filter. Pass the "
                    "same filter, or omit it to reuse the token's filter."
                )

        effective_filter = self._build_filter(filter=filter, resume_from=resume_from)

        total_items_fetched = 0
        total_items_processed = 0
        total_items_failed = 0
        total_scores_created = 0
        total_composite_scores_created = 0
        total_evaluations_failed = 0
        failed_item_ids: List[str] = []
        error_summary: Dict[str, int] = {}
        item_evaluations: Dict[str, List[Evaluation]] = {}
        previously_processed = resume_from.items_processed if resume_from else 0

        evaluator_stats_dict = {
            getattr(evaluator, "__name__", "unknown_evaluator"): EvaluatorStats(
                name=getattr(evaluator, "__name__", "unknown_evaluator")
            )
            for evaluator in evaluators
        }

        semaphore = asyncio.Semaphore(max_concurrency)

        cursor: Optional[str] = resume_from.cursor if resume_from else None
        has_more = True
        last_item_timestamp = (
            resume_from.last_processed_timestamp if resume_from else ""
        )
        last_item_id = resume_from.last_processed_id if resume_from else ""
        # The start-time fallback is inclusive so tied items are not lost; skip
        # the one item the token says was already processed.
        resumed_item_id = (
            resume_from.last_processed_id
            if resume_from is not None
            and resume_from.cursor is None
            and resume_from.last_processed_timestamp
            else None
        )
        batch_number = 0

        if verbose:
            logger.info("Starting batch evaluation on observations")
            if fields:
                logger.info("Fetching observation fields: %s", fields)
            if resume_from:
                logger.info(
                    "Resuming after %s (%s items already processed)",
                    resume_from.last_processed_timestamp or "start",
                    resume_from.items_processed,
                )

        def build_resume_token() -> BatchEvaluationResumeToken:
            return BatchEvaluationResumeToken(
                filter=filter,
                cursor=cursor,
                last_processed_timestamp=last_item_timestamp,
                last_processed_id=last_item_id,
                items_processed=previously_processed + total_items_processed,
            )

        while True:
            if max_items is not None and total_items_fetched >= max_items:
                if verbose:
                    logger.info("Reached max_items limit (%s)", max_items)
                break

            # Clamping the page size keeps the cursor aligned with the last
            # processed item, so resuming after max_items skips nothing.
            limit = fetch_batch_size
            if max_items is not None:
                limit = min(limit, max_items - total_items_fetched)

            try:
                items, next_cursor = await self._fetch_batch_with_retry(
                    filter=effective_filter,
                    cursor=cursor,
                    limit=limit,
                    max_retries=max_retries,
                    fields=fields,
                )
            except Exception as e:
                logger.error(
                    "Failed to fetch batch after %s retries: %s", max_retries, e
                )

                return self._build_result(
                    total_items_fetched=total_items_fetched,
                    total_items_processed=total_items_processed,
                    total_items_failed=total_items_failed,
                    total_scores_created=total_scores_created,
                    total_composite_scores_created=total_composite_scores_created,
                    total_evaluations_failed=total_evaluations_failed,
                    evaluator_stats_dict=evaluator_stats_dict,
                    resume_token=build_resume_token(),
                    completed=False,
                    start_time=start_time,
                    failed_item_ids=failed_item_ids,
                    error_summary=error_summary,
                    has_more_items=False,
                    item_evaluations=item_evaluations,
                )

            batch_number += 1
            total_items_fetched += len(items)

            if verbose:
                logger.info("Fetched batch %s (%s items)", batch_number, len(items))

            async def process_item(
                item: ObservationV2,
            ) -> Tuple[str, Union[Tuple[int, int, int, List[Evaluation]], Exception]]:
                async with semaphore:
                    try:
                        result = await self._process_batch_evaluation_item(
                            item=item,
                            mapper=mapper,
                            evaluators=evaluators,
                            composite_evaluator=composite_evaluator,
                            metadata=metadata,
                            _add_observation_scores_to_trace=_add_observation_scores_to_trace,
                            evaluator_stats_dict=evaluator_stats_dict,
                        )
                        return (item.id, result)
                    except Exception as e:
                        return (item.id, e)

            items_to_process = [item for item in items if item.id != resumed_item_id]

            results = await asyncio.gather(
                *[process_item(item) for item in items_to_process]
            )

            for item, (item_id, result) in zip(items_to_process, results):
                if isinstance(result, Exception):
                    total_items_failed += 1
                    failed_item_ids.append(item_id)
                    error_type = type(result).__name__
                    error_summary[error_type] = error_summary.get(error_type, 0) + 1
                    logger.warning("Item %s failed: %s", item_id, result)
                else:
                    total_items_processed += 1
                    scores_created, composite_created, evals_failed, evaluations = (
                        result
                    )
                    total_scores_created += scores_created
                    total_composite_scores_created += composite_created
                    total_evaluations_failed += evals_failed
                    item_evaluations[item_id] = evaluations

            if items:
                last_item_timestamp = items[-1].start_time.isoformat()
                last_item_id = items[-1].id

            if verbose:
                if max_items is not None and max_items > 0:
                    logger.info(
                        "Progress: %s/%s items (%.1f%%), %s scores created",
                        total_items_processed,
                        max_items,
                        total_items_processed / max_items * 100,
                        total_scores_created,
                    )
                else:
                    logger.info(
                        "Progress: %s items processed, %s scores created",
                        total_items_processed,
                        total_scores_created,
                    )

            cursor = next_cursor
            if cursor is None or not items:
                has_more = False
                break

        if verbose:
            logger.info("Flushing scores to Langfuse...")
        self.client.flush()

        if verbose:
            logger.info(
                "Batch evaluation complete: %s items processed in %.2fs",
                total_items_processed,
                time.time() - start_time,
            )

        return self._build_result(
            total_items_fetched=total_items_fetched,
            total_items_processed=total_items_processed,
            total_items_failed=total_items_failed,
            total_scores_created=total_scores_created,
            total_composite_scores_created=total_composite_scores_created,
            total_evaluations_failed=total_evaluations_failed,
            evaluator_stats_dict=evaluator_stats_dict,
            resume_token=build_resume_token() if has_more else None,
            completed=True,
            start_time=start_time,
            failed_item_ids=failed_item_ids,
            error_summary=error_summary,
            has_more_items=has_more,
            item_evaluations=item_evaluations,
        )

    async def _fetch_batch_with_retry(
        self,
        *,
        filter: Optional[str],
        cursor: Optional[str],
        limit: int,
        max_retries: int,
        fields: Optional[str],
    ) -> Tuple[List[ObservationV2], Optional[str]]:
        """Fetch one page of observations from the v2 observations API.

        Args:
            filter: JSON filter string for querying.
            cursor: Cursor returned with the previous page; None for the first page.
            limit: Number of items to request.
            max_retries: Maximum number of retry attempts.
            fields: Comma-separated v2 field groups to include.

        Returns:
            Tuple of the page's observations and the cursor for the next page
            (None when there are no more pages).

        Raises:
            Exception: If all retry attempts fail.
        """
        response = await asyncio.to_thread(
            self.client.api.observations.get_many,
            fields=fields,
            limit=limit,
            cursor=cursor,
            filter=filter,
            request_options={"max_retries": max_retries},
        )

        return list(response.data), response.meta.cursor

    async def _process_batch_evaluation_item(
        self,
        item: ObservationV2,
        mapper: MapperFunction,
        evaluators: List[EvaluatorFunction],
        composite_evaluator: Optional[CompositeEvaluatorFunction],
        metadata: Optional[Dict[str, Any]],
        _add_observation_scores_to_trace: bool,
        evaluator_stats_dict: Dict[str, EvaluatorStats],
    ) -> Tuple[int, int, int, List[Evaluation]]:
        """Process a single item: map, evaluate, create scores.

        Args:
            item: The observation to evaluate.
            mapper: Function to transform item to evaluator inputs.
            evaluators: List of evaluator functions.
            composite_evaluator: Optional composite evaluator function.
            metadata: Additional metadata to add to scores.
            _add_observation_scores_to_trace: Whether to duplicate
                observation-level scores at trace level.
            evaluator_stats_dict: Dictionary tracking evaluator statistics.

        Returns:
            Tuple of (scores_created, composite_scores_created, evaluations_failed, all_evaluations).

        Raises:
            Exception: If mapping fails or item processing encounters fatal error.
        """
        if not item.trace_id:
            raise ValueError(f"Observation {item.id} has no trace_id")

        scores_created = 0
        composite_scores_created = 0
        evaluations_failed = 0

        evaluator_inputs = await self._run_mapper(mapper, item)

        evaluations: List[Evaluation] = []
        for evaluator in evaluators:
            evaluator_name = getattr(evaluator, "__name__", "unknown_evaluator")
            stats = evaluator_stats_dict[evaluator_name]
            stats.total_runs += 1

            try:
                eval_results = await self._run_evaluator_internal(
                    evaluator,
                    input=evaluator_inputs.input,
                    output=evaluator_inputs.output,
                    expected_output=evaluator_inputs.expected_output,
                    metadata=evaluator_inputs.metadata,
                )

                stats.successful_runs += 1
                stats.total_scores_created += len(eval_results)
                evaluations.extend(eval_results)

            except Exception as e:
                stats.failed_runs += 1
                evaluations_failed += 1
                logger.warning(
                    "Evaluator %s failed on item %s: %s",
                    evaluator_name,
                    item.id,
                    e,
                )

        for evaluation in evaluations:
            scores_created += self._create_score(
                item=item,
                evaluation=evaluation,
                additional_metadata=metadata,
                add_observation_score_to_trace=_add_observation_scores_to_trace,
            )

        if composite_evaluator and evaluations:
            try:
                composite_evals = await self._run_composite_evaluator(
                    composite_evaluator,
                    input=evaluator_inputs.input,
                    output=evaluator_inputs.output,
                    expected_output=evaluator_inputs.expected_output,
                    metadata=evaluator_inputs.metadata,
                    evaluations=evaluations,
                )

                for composite_eval in composite_evals:
                    composite_scores_created += self._create_score(
                        item=item,
                        evaluation=composite_eval,
                        additional_metadata=metadata,
                        add_observation_score_to_trace=_add_observation_scores_to_trace,
                    )

                evaluations.extend(composite_evals)

            except Exception as e:
                logger.warning("Composite evaluator failed on item %s: %s", item.id, e)

        return (
            scores_created,
            composite_scores_created,
            evaluations_failed,
            evaluations,
        )

    async def _run_evaluator_internal(
        self,
        evaluator: EvaluatorFunction,
        **kwargs: Any,
    ) -> List[Evaluation]:
        """Run an evaluator function and normalize the result.

        Unlike experiment._run_evaluator, this version raises exceptions
        so we can track failures in our statistics.

        Args:
            evaluator: The evaluator function to run.
            **kwargs: Arguments to pass to the evaluator.

        Returns:
            List of Evaluation objects.

        Raises:
            Exception: If evaluator raises an exception (not caught).
        """
        result = evaluator(**kwargs)

        if asyncio.iscoroutine(result):
            result = await result

        if isinstance(result, (dict, Evaluation)):
            return [result]  # type: ignore
        elif isinstance(result, list):
            return result
        else:
            return []

    async def _run_mapper(
        self,
        mapper: MapperFunction,
        item: ObservationV2,
    ) -> EvaluatorInputs:
        """Run mapper function (handles both sync and async mappers).

        Args:
            mapper: The mapper function to run.
            item: The observation to map.

        Returns:
            EvaluatorInputs instance.

        Raises:
            Exception: If mapper raises an exception.
        """
        result = mapper(item=item)
        if asyncio.iscoroutine(result):
            return await result  # type: ignore
        return result  # type: ignore

    async def _run_composite_evaluator(
        self,
        composite_evaluator: CompositeEvaluatorFunction,
        input: Optional[Any],
        output: Optional[Any],
        expected_output: Optional[Any],
        metadata: Optional[Dict[str, Any]],
        evaluations: List[Evaluation],
    ) -> List[Evaluation]:
        """Run composite evaluator function (handles both sync and async).

        Args:
            composite_evaluator: The composite evaluator function.
            input: The input data provided to the system.
            output: The output generated by the system.
            expected_output: The expected/reference output.
            metadata: Additional metadata about the evaluation context.
            evaluations: List of item-level evaluations.

        Returns:
            List of Evaluation objects (normalized from single or list return).

        Raises:
            Exception: If composite evaluator raises an exception.
        """
        result = composite_evaluator(
            input=input,
            output=output,
            expected_output=expected_output,
            metadata=metadata,
            evaluations=evaluations,
        )
        if asyncio.iscoroutine(result):
            result = await result

        if isinstance(result, (dict, Evaluation)):
            return [result]  # type: ignore
        elif isinstance(result, list):
            return result
        else:
            return []

    def _create_score(
        self,
        *,
        item: ObservationV2,
        evaluation: Evaluation,
        additional_metadata: Optional[Dict[str, Any]],
        add_observation_score_to_trace: bool = False,
    ) -> int:
        """Create a score on the evaluated observation, optionally duplicated onto its trace.

        Args:
            item: The evaluated observation.
            evaluation: The evaluation result to create a score from.
            additional_metadata: Additional metadata to merge with evaluation metadata.
            add_observation_score_to_trace: Whether to duplicate observation
                score on parent trace as well.

        Returns:
            Number of score events created.
        """
        score_metadata = {
            **(evaluation.metadata or {}),
            **(additional_metadata or {}),
        }
        score_kwargs: Dict[str, Any] = {
            "name": evaluation.name,
            "value": evaluation.value,
            "comment": evaluation.comment,
            "metadata": score_metadata,
            "data_type": evaluation.data_type,
            "config_id": evaluation.config_id,
        }

        self.client.create_score(
            observation_id=item.id, trace_id=item.trace_id, **score_kwargs
        )
        if not add_observation_score_to_trace:
            return 1

        self.client.create_score(trace_id=item.trace_id, **score_kwargs)
        return 2

    @staticmethod
    def _build_filter(
        *,
        filter: Optional[str],
        resume_from: Optional[BatchEvaluationResumeToken],
    ) -> Optional[str]:
        """Combine the user filter with the resume constraint.

        Constraints are added as JSON filter conditions rather than query
        parameters because the API drops a query-parameter filter whenever the
        JSON filter has a condition on the same column.

        Args:
            filter: The user-provided JSON filter string (a JSON array).
            resume_from: Optional resume token.

        Returns:
            The JSON filter string to send, or None if there are no conditions.

        Raises:
            ValueError: If the filter is not a JSON array.
        """
        conditions: List[Any] = []
        if filter:
            try:
                parsed = json.loads(filter)
            except json.JSONDecodeError as e:
                raise ValueError(f"filter must be a JSON array: {e}") from e
            if not isinstance(parsed, list):
                raise ValueError(
                    f"filter must be a JSON array of conditions, got {type(parsed).__name__}"
                )
            conditions.extend(parsed)

        # Results are ordered by start time descending, so items that remain
        # after the last processed one started before it. The cursor resumes
        # exactly; the timestamp is the fallback for tokens without a cursor.
        if (
            resume_from is not None
            and resume_from.cursor is None
            and resume_from.last_processed_timestamp
        ):
            conditions.append(
                {
                    "type": "datetime",
                    "column": "startTime",
                    "operator": "<=",
                    "value": resume_from.last_processed_timestamp,
                }
            )

        return json.dumps(conditions) if conditions else None

    def _build_result(
        self,
        total_items_fetched: int,
        total_items_processed: int,
        total_items_failed: int,
        total_scores_created: int,
        total_composite_scores_created: int,
        total_evaluations_failed: int,
        evaluator_stats_dict: Dict[str, EvaluatorStats],
        resume_token: Optional[BatchEvaluationResumeToken],
        completed: bool,
        start_time: float,
        failed_item_ids: List[str],
        error_summary: Dict[str, int],
        has_more_items: bool,
        item_evaluations: Dict[str, List[Evaluation]],
    ) -> BatchEvaluationResult:
        """Build the final BatchEvaluationResult.

        Args:
            total_items_fetched: Total items fetched.
            total_items_processed: Items successfully processed.
            total_items_failed: Items that failed.
            total_scores_created: Scores from item evaluators.
            total_composite_scores_created: Scores from composite evaluator.
            total_evaluations_failed: Individual evaluator failures.
            evaluator_stats_dict: Per-evaluator statistics.
            resume_token: Resume token if the run can be continued.
            completed: Whether evaluation completed without a fetch failure.
            start_time: Start time (unix timestamp).
            failed_item_ids: IDs of failed items.
            error_summary: Error type counts.
            has_more_items: Whether more items exist beyond max_items.
            item_evaluations: Dictionary mapping item IDs to their evaluation results.

        Returns:
            BatchEvaluationResult instance.
        """
        duration = time.time() - start_time

        return BatchEvaluationResult(
            total_items_fetched=total_items_fetched,
            total_items_processed=total_items_processed,
            total_items_failed=total_items_failed,
            total_scores_created=total_scores_created,
            total_composite_scores_created=total_composite_scores_created,
            total_evaluations_failed=total_evaluations_failed,
            evaluator_stats=list(evaluator_stats_dict.values()),
            resume_token=resume_token,
            completed=completed,
            duration_seconds=duration,
            failed_item_ids=failed_item_ids,
            error_summary=error_summary,
            has_more_items=has_more_items,
            item_evaluations=item_evaluations,
        )
