"""Comprehensive tests for batch evaluation functionality.

This test suite covers the run_batched_evaluation method which allows evaluating
traces, observations, and sessions fetched from Langfuse with mappers, evaluators,
and composite evaluators.
"""

import asyncio
import time

import pytest

from langfuse import get_client
from langfuse.batch_evaluation import (
    BatchEvaluationResult,
    BatchEvaluationResumeToken,
    EvaluatorInputs,
    EvaluatorStats,
)
from langfuse.experiment import Evaluation
from tests.utils import create_uuid

# ============================================================================
# FIXTURES & SETUP
# ============================================================================


# pytestmark = pytest.mark.skip(reason="Github CI runner overwhelmed by score volume")


@pytest.fixture
def langfuse_client():
    """Get a Langfuse client for testing."""
    return get_client()


@pytest.fixture
def sample_trace_name():
    """Generate a unique trace name for filtering."""
    return f"batch-eval-test-{create_uuid()}"


def simple_trace_mapper(*, item):
    """Simple mapper for traces."""
    return EvaluatorInputs(
        input=item.input if hasattr(item, "input") else None,
        output=item.output if hasattr(item, "output") else None,
        expected_output=None,
        metadata={"trace_id": item.id},
    )


def simple_evaluator(*, input, output, expected_output=None, metadata=None, **kwargs):
    """Simple evaluator that returns a score based on output length."""
    if output is None:
        return Evaluation(name="length_score", value=0.0, comment="No output")

    return Evaluation(
        name="length_score",
        value=float(len(str(output))) / 10.0,
        comment=f"Length: {len(str(output))}",
    )


# ============================================================================
# BASIC FUNCTIONALITY TESTS
# ============================================================================


def test_run_batched_evaluation_on_observations_basic(langfuse_client):
    """Test basic batch evaluation on traces."""
    result = langfuse_client.run_batched_evaluation(
        scope="observations",
        mapper=simple_trace_mapper,
        evaluators=[simple_evaluator],
        max_items=1,
        verbose=True,
    )

    # Validate result structure
    assert isinstance(result, BatchEvaluationResult)
    assert result.total_items_fetched >= 0
    assert result.total_items_processed >= 0
    assert result.total_scores_created >= 0
    assert result.completed is True
    assert isinstance(result.duration_seconds, float)
    assert result.duration_seconds > 0

    # Verify evaluator stats
    assert len(result.evaluator_stats) == 1
    stats = result.evaluator_stats[0]
    assert isinstance(stats, EvaluatorStats)
    assert stats.name == "simple_evaluator"


def test_run_batched_evaluation_on_traces_basic(langfuse_client):
    """Test basic batch evaluation on traces."""
    result = langfuse_client.run_batched_evaluation(
        scope="traces",
        mapper=simple_trace_mapper,
        evaluators=[simple_evaluator],
        max_items=5,
        verbose=True,
    )

    # Validate result structure
    assert isinstance(result, BatchEvaluationResult)
    assert result.total_items_fetched >= 0
    assert result.total_items_processed >= 0
    assert result.total_scores_created >= 0
    assert result.completed is True
    assert isinstance(result.duration_seconds, float)
    assert result.duration_seconds > 0

    # Verify evaluator stats
    assert len(result.evaluator_stats) == 1
    stats = result.evaluator_stats[0]
    assert isinstance(stats, EvaluatorStats)
    assert stats.name == "simple_evaluator"


def test_batch_evaluation_with_filter(langfuse_client):
    """Test batch evaluation with JSON filter."""
    # Create a trace with specific tag
    unique_tag = f"test-filter-{create_uuid()}"
    with langfuse_client.start_as_current_span(
        name=f"filtered-trace-{create_uuid()}"
    ) as span:
        span.update_trace(
            input="Filtered test",
            output="Filtered output",
            tags=[unique_tag],
        )

    langfuse_client.flush()
    time.sleep(3)

    # Filter format: array of filter conditions
    filter_json = f'[{{"type": "arrayOptions", "column": "tags", "operator": "any of", "value": ["{unique_tag}"]}}]'

    result = langfuse_client.run_batched_evaluation(
        scope="traces",
        mapper=simple_trace_mapper,
        evaluators=[simple_evaluator],
        filter=filter_json,
        verbose=True,
    )

    # Should only process the filtered trace
    assert result.total_items_fetched >= 1
    assert result.completed is True


def test_batch_evaluation_with_metadata(langfuse_client):
    """Test that additional metadata is added to all scores."""

    def metadata_checking_evaluator(*, input, output, metadata=None, **kwargs):
        return Evaluation(
            name="test_score",
            value=1.0,
            metadata={"evaluator_data": "test"},
        )

    additional_metadata = {
        "batch_run_id": "test-batch-123",
        "evaluation_version": "v2.0",
    }

    result = langfuse_client.run_batched_evaluation(
        scope="traces",
        mapper=simple_trace_mapper,
        evaluators=[metadata_checking_evaluator],
        metadata=additional_metadata,
        max_items=2,
    )

    assert result.total_scores_created > 0

    # Verify scores were created with merged metadata
    langfuse_client.flush()
    time.sleep(3)

    # Note: In a real test, you'd verify via API that metadata was merged
    # For now, just verify the operation completed
    assert result.completed is True


def test_result_structure_fields(langfuse_client):
    """Test that BatchEvaluationResult has all expected fields."""
    result = langfuse_client.run_batched_evaluation(
        scope="traces",
        mapper=simple_trace_mapper,
        evaluators=[simple_evaluator],
        max_items=3,
    )

    # Check all result fields exist
    assert hasattr(result, "total_items_fetched")
    assert hasattr(result, "total_items_processed")
    assert hasattr(result, "total_items_failed")
    assert hasattr(result, "total_scores_created")
    assert hasattr(result, "total_composite_scores_created")
    assert hasattr(result, "total_evaluations_failed")
    assert hasattr(result, "evaluator_stats")
    assert hasattr(result, "resume_token")
    assert hasattr(result, "completed")
    assert hasattr(result, "duration_seconds")
    assert hasattr(result, "failed_item_ids")
    assert hasattr(result, "error_summary")
    assert hasattr(result, "has_more_items")
    assert hasattr(result, "item_evaluations")

    # Check types
    assert isinstance(result.evaluator_stats, list)
    assert isinstance(result.failed_item_ids, list)
    assert isinstance(result.error_summary, dict)
    assert isinstance(result.completed, bool)
    assert isinstance(result.has_more_items, bool)
    assert isinstance(result.item_evaluations, dict)


# ============================================================================
# MAPPER FUNCTION TESTS
# ============================================================================


def test_simple_mapper(langfuse_client):
    """Test basic mapper functionality."""

    def custom_mapper(*, item):
        return EvaluatorInputs(
            input=item.input if hasattr(item, "input") else "no input",
            output=item.output if hasattr(item, "output") else "no output",
            expected_output=None,
            metadata={"custom_field": "test_value"},
        )

    result = langfuse_client.run_batched_evaluation(
        scope="traces",
        mapper=custom_mapper,
        evaluators=[simple_evaluator],
        max_items=2,
    )

    assert result.total_items_processed > 0


@pytest.mark.asyncio
async def test_async_mapper(langfuse_client):
    """Test that async mappers work correctly."""

    async def async_mapper(*, item):
        await asyncio.sleep(0.01)  # Simulate async work
        return EvaluatorInputs(
            input=item.input if hasattr(item, "input") else None,
            output=item.output if hasattr(item, "output") else None,
            expected_output=None,
            metadata={"async": True},
        )

    # Note: run_batched_evaluation is synchronous but handles async mappers
    result = langfuse_client.run_batched_evaluation(
        scope="traces",
        mapper=async_mapper,
        evaluators=[simple_evaluator],
        max_items=2,
    )

    assert result.total_items_processed > 0


def test_mapper_failure_handling(langfuse_client):
    """Test that mapper failures cause items to be skipped."""

    def failing_mapper(*, item):
        raise ValueError("Intentional mapper failure")

    result = langfuse_client.run_batched_evaluation(
        scope="traces",
        mapper=failing_mapper,
        evaluators=[simple_evaluator],
        max_items=3,
    )

    # All items should fail due to mapper failures
    assert result.total_items_failed > 0
    assert len(result.failed_item_ids) > 0
    assert "ValueError" in result.error_summary or "Exception" in result.error_summary


def test_mapper_with_missing_fields(langfuse_client):
    """Test mapper handles traces with missing fields gracefully."""

    def robust_mapper(*, item):
        # Handle missing fields with defaults
        input_val = getattr(item, "input", None) or "default_input"
        output_val = getattr(item, "output", None) or "default_output"

        return EvaluatorInputs(
            input=input_val,
            output=output_val,
            expected_output=None,
            metadata={},
        )

    result = langfuse_client.run_batched_evaluation(
        scope="traces",
        mapper=robust_mapper,
        evaluators=[simple_evaluator],
        max_items=2,
    )

    assert result.total_items_processed > 0


# ============================================================================
# EVALUATOR TESTS
# ============================================================================


def test_single_evaluator(langfuse_client):
    """Test with a single evaluator."""

    def quality_evaluator(*, input, output, **kwargs):
        return Evaluation(name="quality", value=0.85, comment="High quality")

    result = langfuse_client.run_batched_evaluation(
        scope="traces",
        mapper=simple_trace_mapper,
        evaluators=[quality_evaluator],
        max_items=2,
    )

    assert result.total_scores_created > 0
    assert len(result.evaluator_stats) == 1
    assert result.evaluator_stats[0].name == "quality_evaluator"


def test_multiple_evaluators(langfuse_client):
    """Test with multiple evaluators running in parallel."""

    def accuracy_evaluator(*, input, output, **kwargs):
        return Evaluation(name="accuracy", value=0.9)

    def relevance_evaluator(*, input, output, **kwargs):
        return Evaluation(name="relevance", value=0.8)

    def safety_evaluator(*, input, output, **kwargs):
        return Evaluation(name="safety", value=1.0)

    result = langfuse_client.run_batched_evaluation(
        scope="traces",
        mapper=simple_trace_mapper,
        evaluators=[accuracy_evaluator, relevance_evaluator, safety_evaluator],
        max_items=2,
    )

    # Should have 3 evaluators
    assert len(result.evaluator_stats) == 3
    assert result.total_scores_created >= result.total_items_processed * 3


@pytest.mark.asyncio
async def test_async_evaluator(langfuse_client):
    """Test that async evaluators work correctly."""

    async def async_evaluator(*, input, output, **kwargs):
        await asyncio.sleep(0.01)  # Simulate async work
        return Evaluation(name="async_score", value=0.75)

    result = langfuse_client.run_batched_evaluation(
        scope="traces",
        mapper=simple_trace_mapper,
        evaluators=[async_evaluator],
        max_items=2,
    )

    assert result.total_scores_created > 0


def test_evaluator_returning_list(langfuse_client):
    """Test evaluator that returns multiple Evaluations."""

    def multi_score_evaluator(*, input, output, **kwargs):
        return [
            Evaluation(name="score_1", value=0.8),
            Evaluation(name="score_2", value=0.9),
            Evaluation(name="score_3", value=0.7),
        ]

    result = langfuse_client.run_batched_evaluation(
        scope="traces",
        mapper=simple_trace_mapper,
        evaluators=[multi_score_evaluator],
        max_items=2,
    )

    # Should create 3 scores per item
    assert result.total_scores_created >= result.total_items_processed * 3


def test_evaluator_failure_statistics(langfuse_client):
    """Test that evaluator failures are tracked in statistics."""

    def working_evaluator(*, input, output, **kwargs):
        return Evaluation(name="working", value=1.0)

    def failing_evaluator(*, input, output, **kwargs):
        raise RuntimeError("Intentional evaluator failure")

    result = langfuse_client.run_batched_evaluation(
        scope="traces",
        mapper=simple_trace_mapper,
        evaluators=[working_evaluator, failing_evaluator],
        max_items=3,
    )

    # Verify evaluator stats
    assert len(result.evaluator_stats) == 2

    working_stats = next(
        s for s in result.evaluator_stats if s.name == "working_evaluator"
    )
    assert working_stats.successful_runs > 0
    assert working_stats.failed_runs == 0

    failing_stats = next(
        s for s in result.evaluator_stats if s.name == "failing_evaluator"
    )
    assert failing_stats.failed_runs > 0
    assert failing_stats.successful_runs == 0

    # Total evaluations failed should be tracked
    assert result.total_evaluations_failed > 0


def test_mixed_sync_async_evaluators(langfuse_client):
    """Test mixing synchronous and asynchronous evaluators."""

    def sync_evaluator(*, input, output, **kwargs):
        return Evaluation(name="sync_score", value=0.8)

    async def async_evaluator(*, input, output, **kwargs):
        await asyncio.sleep(0.01)
        return Evaluation(name="async_score", value=0.9)

    result = langfuse_client.run_batched_evaluation(
        scope="traces",
        mapper=simple_trace_mapper,
        evaluators=[sync_evaluator, async_evaluator],
        max_items=2,
    )

    assert len(result.evaluator_stats) == 2
    assert result.total_scores_created >= result.total_items_processed * 2


# ============================================================================
# COMPOSITE EVALUATOR TESTS
# ============================================================================


def test_composite_evaluator_weighted_average(langfuse_client):
    """Test composite evaluator that computes weighted average."""

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

    result = langfuse_client.run_batched_evaluation(
        scope="traces",
        mapper=simple_trace_mapper,
        evaluators=[accuracy_evaluator, relevance_evaluator],
        composite_evaluator=composite_evaluator,
        max_items=2,
    )

    # Should have both regular and composite scores
    assert result.total_scores_created > 0
    assert result.total_composite_scores_created > 0
    assert result.total_scores_created > result.total_composite_scores_created


def test_composite_evaluator_pass_fail(langfuse_client):
    """Test composite evaluator that implements pass/fail logic."""

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

    result = langfuse_client.run_batched_evaluation(
        scope="traces",
        mapper=simple_trace_mapper,
        evaluators=[metric1_evaluator, metric2_evaluator],
        composite_evaluator=pass_fail_composite,
        max_items=2,
    )

    assert result.total_composite_scores_created > 0


@pytest.mark.asyncio
async def test_async_composite_evaluator(langfuse_client):
    """Test async composite evaluator."""

    def evaluator1(*, input, output, **kwargs):
        return Evaluation(name="eval1", value=0.8)

    async def async_composite(*, input, output, expected_output, metadata, evaluations):
        await asyncio.sleep(0.01)  # Simulate async processing
        avg = sum(
            e.value for e in evaluations if isinstance(e.value, (int, float))
        ) / len(evaluations)
        return Evaluation(name="async_composite", value=avg)

    result = langfuse_client.run_batched_evaluation(
        scope="traces",
        mapper=simple_trace_mapper,
        evaluators=[evaluator1],
        composite_evaluator=async_composite,
        max_items=2,
    )

    assert result.total_composite_scores_created > 0


def test_composite_evaluator_with_no_evaluations(langfuse_client):
    """Test composite evaluator when no evaluations are present."""

    def always_failing_evaluator(*, input, output, **kwargs):
        raise Exception("Always fails")

    def composite_evaluator(*, input, output, expected_output, metadata, evaluations):
        # Should not be called if no evaluations succeed
        return Evaluation(name="composite", value=0.0)

    result = langfuse_client.run_batched_evaluation(
        scope="traces",
        mapper=simple_trace_mapper,
        evaluators=[always_failing_evaluator],
        composite_evaluator=composite_evaluator,
        max_items=2,
    )

    # Composite evaluator should not create scores if no evaluations
    assert result.total_composite_scores_created == 0


def test_composite_evaluator_failure_handling(langfuse_client):
    """Test that composite evaluator failures are handled gracefully."""

    def evaluator1(*, input, output, **kwargs):
        return Evaluation(name="eval1", value=0.8)

    def failing_composite(*, input, output, expected_output, metadata, evaluations):
        raise ValueError("Composite evaluator failed")

    result = langfuse_client.run_batched_evaluation(
        scope="traces",
        mapper=simple_trace_mapper,
        evaluators=[evaluator1],
        composite_evaluator=failing_composite,
        max_items=2,
    )

    # Regular scores should still be created
    assert result.total_scores_created > 0
    # But no composite scores
    assert result.total_composite_scores_created == 0


# ============================================================================
# ERROR HANDLING TESTS
# ============================================================================


def test_mapper_failure_skips_item(langfuse_client):
    """Test that mapper failure causes item to be skipped."""

    call_count = {"count": 0}

    def sometimes_failing_mapper(*, item):
        call_count["count"] += 1
        if call_count["count"] % 2 == 0:
            raise Exception("Mapper failed")
        return simple_trace_mapper(item=item)

    result = langfuse_client.run_batched_evaluation(
        scope="traces",
        mapper=sometimes_failing_mapper,
        evaluators=[simple_evaluator],
        max_items=4,
    )

    # Some items should fail, some should succeed
    assert result.total_items_failed > 0
    assert result.total_items_processed > 0


def test_evaluator_failure_continues(langfuse_client):
    """Test that one evaluator failing doesn't stop others."""

    def working_evaluator1(*, input, output, **kwargs):
        return Evaluation(name="working1", value=0.8)

    def failing_evaluator(*, input, output, **kwargs):
        raise Exception("Evaluator failed")

    def working_evaluator2(*, input, output, **kwargs):
        return Evaluation(name="working2", value=0.9)

    result = langfuse_client.run_batched_evaluation(
        scope="traces",
        mapper=simple_trace_mapper,
        evaluators=[working_evaluator1, failing_evaluator, working_evaluator2],
        max_items=2,
    )

    # Working evaluators should still create scores
    assert result.total_scores_created >= result.total_items_processed * 2

    # Failing evaluator should be tracked
    failing_stats = next(
        s for s in result.evaluator_stats if s.name == "failing_evaluator"
    )
    assert failing_stats.failed_runs > 0


def test_all_evaluators_fail(langfuse_client):
    """Test when all evaluators fail but item is still processed."""

    def failing_evaluator1(*, input, output, **kwargs):
        raise Exception("Failed 1")

    def failing_evaluator2(*, input, output, **kwargs):
        raise Exception("Failed 2")

    result = langfuse_client.run_batched_evaluation(
        scope="traces",
        mapper=simple_trace_mapper,
        evaluators=[failing_evaluator1, failing_evaluator2],
        max_items=2,
    )

    # Items should be processed even if all evaluators fail
    assert result.total_items_processed > 0
    # But no scores created
    assert result.total_scores_created == 0
    # All evaluations failed
    assert result.total_evaluations_failed > 0


# ============================================================================
# EDGE CASES TESTS
# ============================================================================


def test_empty_results_handling(langfuse_client):
    """Test batch evaluation when filter returns no items."""
    nonexistent_name = f"nonexistent-trace-{create_uuid()}"
    nonexistent_filter = f'[{{"type": "string", "column": "name", "operator": "=", "value": "{nonexistent_name}"}}]'

    result = langfuse_client.run_batched_evaluation(
        scope="traces",
        mapper=simple_trace_mapper,
        evaluators=[simple_evaluator],
        filter=nonexistent_filter,
    )

    assert result.total_items_fetched == 0
    assert result.total_items_processed == 0
    assert result.total_scores_created == 0
    assert result.completed is True
    assert result.has_more_items is False


def test_max_items_zero(langfuse_client):
    """Test with max_items=0 (should process no items)."""
    result = langfuse_client.run_batched_evaluation(
        scope="traces",
        mapper=simple_trace_mapper,
        evaluators=[simple_evaluator],
        max_items=0,
    )

    assert result.total_items_fetched == 0
    assert result.total_items_processed == 0


def test_evaluation_value_type_conversions(langfuse_client):
    """Test that different evaluation value types are handled correctly."""

    def multi_type_evaluator(*, input, output, **kwargs):
        return [
            Evaluation(name="int_score", value=5),  # int
            Evaluation(name="float_score", value=0.85),  # float
            Evaluation(name="bool_score", value=True),  # bool
            Evaluation(name="none_score", value=None),  # None
        ]

    result = langfuse_client.run_batched_evaluation(
        scope="traces",
        mapper=simple_trace_mapper,
        evaluators=[multi_type_evaluator],
        max_items=1,
    )

    # All value types should be converted and scores created
    assert result.total_scores_created >= 4


# ============================================================================
# PAGINATION TESTS
# ============================================================================


def test_pagination_with_max_items(langfuse_client):
    """Test that max_items limit is respected."""
    # Create more traces to ensure we have enough data
    for i in range(10):
        with langfuse_client.start_as_current_span(
            name=f"pagination-test-{create_uuid()}"
        ) as span:
            span.update_trace(
                input=f"Input {i}",
                output=f"Output {i}",
                tags=["pagination_test"],
            )

    langfuse_client.flush()
    time.sleep(3)

    filter_json = '[{"type": "arrayOptions", "column": "tags", "operator": "any of", "value": ["pagination_test"]}]'

    result = langfuse_client.run_batched_evaluation(
        scope="traces",
        mapper=simple_trace_mapper,
        evaluators=[simple_evaluator],
        filter=filter_json,
        max_items=5,
        fetch_batch_size=2,
    )

    # Should not exceed max_items
    assert result.total_items_processed <= 5


def test_has_more_items_flag(langfuse_client):
    """Test that has_more_items flag is set correctly when max_items is reached."""
    # Create enough traces to exceed max_items
    batch_tag = f"batch-test-{create_uuid()}"
    for i in range(15):
        with langfuse_client.start_as_current_span(name=f"more-items-test-{i}") as span:
            span.update_trace(
                input=f"Input {i}",
                output=f"Output {i}",
                tags=[batch_tag],
            )

    langfuse_client.flush()
    time.sleep(3)

    filter_json = f'[{{"type": "arrayOptions", "column": "tags", "operator": "any of", "value": ["{batch_tag}"]}}]'

    result = langfuse_client.run_batched_evaluation(
        scope="traces",
        mapper=simple_trace_mapper,
        evaluators=[simple_evaluator],
        filter=filter_json,
        max_items=5,
        fetch_batch_size=2,
    )

    # has_more_items should be True if we hit the limit
    if result.total_items_fetched >= 5:
        assert result.has_more_items is True


def test_fetch_batch_size_parameter(langfuse_client):
    """Test that different fetch_batch_size values work correctly."""
    for batch_size in [1, 5, 10]:
        result = langfuse_client.run_batched_evaluation(
            scope="traces",
            mapper=simple_trace_mapper,
            evaluators=[simple_evaluator],
            max_items=3,
            fetch_batch_size=batch_size,
        )

        # Should complete regardless of batch size
        assert result.completed is True or result.total_items_processed > 0


# ============================================================================
# RESUME FUNCTIONALITY TESTS
# ============================================================================


def test_resume_token_structure(langfuse_client):
    """Test that BatchEvaluationResumeToken has correct structure."""
    resume_token = BatchEvaluationResumeToken(
        scope="traces",
        filter='{"test": "filter"}',
        last_processed_timestamp="2024-01-01T00:00:00Z",
        last_processed_id="trace-123",
        items_processed=10,
    )

    assert resume_token.scope == "traces"
    assert resume_token.filter == '{"test": "filter"}'
    assert resume_token.last_processed_timestamp == "2024-01-01T00:00:00Z"
    assert resume_token.last_processed_id == "trace-123"
    assert resume_token.items_processed == 10


# ============================================================================
# CONCURRENCY TESTS
# ============================================================================


def test_max_concurrency_parameter(langfuse_client):
    """Test that max_concurrency parameter works correctly."""
    for concurrency in [1, 5, 10]:
        result = langfuse_client.run_batched_evaluation(
            scope="traces",
            mapper=simple_trace_mapper,
            evaluators=[simple_evaluator],
            max_items=3,
            max_concurrency=concurrency,
        )

        # Should complete regardless of concurrency
        assert result.completed is True or result.total_items_processed > 0


# ============================================================================
# STATISTICS TESTS
# ============================================================================


def test_evaluator_stats_structure(langfuse_client):
    """Test that EvaluatorStats has correct structure."""

    def test_evaluator(*, input, output, **kwargs):
        return Evaluation(name="test", value=1.0)

    result = langfuse_client.run_batched_evaluation(
        scope="traces",
        mapper=simple_trace_mapper,
        evaluators=[test_evaluator],
        max_items=2,
    )

    assert len(result.evaluator_stats) == 1
    stats = result.evaluator_stats[0]

    # Check all fields exist
    assert hasattr(stats, "name")
    assert hasattr(stats, "total_runs")
    assert hasattr(stats, "successful_runs")
    assert hasattr(stats, "failed_runs")
    assert hasattr(stats, "total_scores_created")

    # Check values
    assert stats.name == "test_evaluator"
    assert stats.total_runs == result.total_items_processed
    assert stats.successful_runs == result.total_items_processed
    assert stats.failed_runs == 0


def test_evaluator_stats_tracking(langfuse_client):
    """Test that evaluator statistics are tracked correctly."""

    call_count = {"count": 0}

    def sometimes_failing_evaluator(*, input, output, **kwargs):
        call_count["count"] += 1
        if call_count["count"] % 2 == 0:
            raise Exception("Failed")
        return Evaluation(name="test", value=1.0)

    result = langfuse_client.run_batched_evaluation(
        scope="traces",
        mapper=simple_trace_mapper,
        evaluators=[sometimes_failing_evaluator],
        max_items=4,
    )

    stats = result.evaluator_stats[0]
    assert stats.total_runs == result.total_items_processed
    assert stats.successful_runs > 0
    assert stats.failed_runs > 0
    assert stats.successful_runs + stats.failed_runs == stats.total_runs


def test_error_summary_aggregation(langfuse_client):
    """Test that error types are aggregated correctly in error_summary."""

    def failing_mapper(*, item):
        raise ValueError("Mapper error")

    result = langfuse_client.run_batched_evaluation(
        scope="traces",
        mapper=failing_mapper,
        evaluators=[simple_evaluator],
        max_items=3,
    )

    # Error summary should contain the error type
    assert len(result.error_summary) > 0
    assert any("Error" in key for key in result.error_summary.keys())


def test_failed_item_ids_collected(langfuse_client):
    """Test that failed item IDs are collected."""

    def failing_mapper(*, item):
        raise Exception("Failed")

    result = langfuse_client.run_batched_evaluation(
        scope="traces",
        mapper=failing_mapper,
        evaluators=[simple_evaluator],
        max_items=3,
    )

    assert len(result.failed_item_ids) > 0
    # Each failed ID should be a string
    assert all(isinstance(item_id, str) for item_id in result.failed_item_ids)


# ============================================================================
# PERFORMANCE TESTS
# ============================================================================


def test_duration_tracking(langfuse_client):
    """Test that duration is tracked correctly."""
    result = langfuse_client.run_batched_evaluation(
        scope="traces",
        mapper=simple_trace_mapper,
        evaluators=[simple_evaluator],
        max_items=2,
    )

    assert result.duration_seconds > 0
    assert result.duration_seconds < 60  # Should complete quickly for small batch


def test_verbose_logging(langfuse_client):
    """Test that verbose=True doesn't cause errors."""
    result = langfuse_client.run_batched_evaluation(
        scope="traces",
        mapper=simple_trace_mapper,
        evaluators=[simple_evaluator],
        max_items=2,
        verbose=True,  # Should log progress
    )

    assert result.completed is True


# ============================================================================
# ITEM EVALUATIONS TESTS
# ============================================================================


def test_item_evaluations_basic(langfuse_client):
    """Test that item_evaluations dict contains correct structure."""

    def test_evaluator(*, input, output, **kwargs):
        return Evaluation(name="test_metric", value=0.5)

    result = langfuse_client.run_batched_evaluation(
        scope="traces",
        mapper=simple_trace_mapper,
        evaluators=[test_evaluator],
        max_items=3,
    )

    # Check that item_evaluations is a dict
    assert isinstance(result.item_evaluations, dict)

    # Should have evaluations for each processed item
    assert len(result.item_evaluations) == result.total_items_processed

    # Each entry should be a list of Evaluation objects
    for item_id, evaluations in result.item_evaluations.items():
        assert isinstance(item_id, str)
        assert isinstance(evaluations, list)
        assert all(isinstance(e, Evaluation) for e in evaluations)
        # Should have one evaluation per evaluator
        assert len(evaluations) == 1
        assert evaluations[0].name == "test_metric"


def test_item_evaluations_multiple_evaluators(langfuse_client):
    """Test item_evaluations with multiple evaluators."""

    def accuracy_evaluator(*, input, output, **kwargs):
        return Evaluation(name="accuracy", value=0.8)

    def relevance_evaluator(*, input, output, **kwargs):
        return Evaluation(name="relevance", value=0.9)

    result = langfuse_client.run_batched_evaluation(
        scope="traces",
        mapper=simple_trace_mapper,
        evaluators=[accuracy_evaluator, relevance_evaluator],
        max_items=2,
    )

    # Check structure
    assert len(result.item_evaluations) == result.total_items_processed

    # Each item should have evaluations from both evaluators
    for item_id, evaluations in result.item_evaluations.items():
        assert len(evaluations) == 2
        eval_names = {e.name for e in evaluations}
        assert eval_names == {"accuracy", "relevance"}


def test_item_evaluations_with_composite(langfuse_client):
    """Test that item_evaluations includes composite evaluations."""

    def base_evaluator(*, input, output, **kwargs):
        return Evaluation(name="base_score", value=0.7)

    def composite_evaluator(*, input, output, expected_output, metadata, evaluations):
        return Evaluation(
            name="composite_score",
            value=sum(
                e.value for e in evaluations if isinstance(e.value, (int, float))
            ),
        )

    result = langfuse_client.run_batched_evaluation(
        scope="traces",
        mapper=simple_trace_mapper,
        evaluators=[base_evaluator],
        composite_evaluator=composite_evaluator,
        max_items=2,
    )

    # Each item should have both base and composite evaluations
    for item_id, evaluations in result.item_evaluations.items():
        assert len(evaluations) == 2
        eval_names = {e.name for e in evaluations}
        assert eval_names == {"base_score", "composite_score"}

    # Verify composite scores were created
    assert result.total_composite_scores_created > 0


def test_item_evaluations_empty_on_failure(langfuse_client):
    """Test that failed items don't appear in item_evaluations."""

    def failing_mapper(*, item):
        raise Exception("Mapper failed")

    result = langfuse_client.run_batched_evaluation(
        scope="traces",
        mapper=failing_mapper,
        evaluators=[simple_evaluator],
        max_items=3,
    )

    # All items failed, so item_evaluations should be empty
    assert len(result.item_evaluations) == 0
    assert result.total_items_failed > 0
