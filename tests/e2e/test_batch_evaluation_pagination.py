"""Exercise trace pagination through a real Langfuse server."""

import json

import pytest

from langfuse import get_client, propagate_attributes
from langfuse.batch_evaluation import EvaluatorInputs
from langfuse.experiment import Evaluation
from tests.support.utils import create_uuid, wait_for_result


def _seed_multi_root_corpus() -> tuple[str, set[str]]:
    client = get_client()
    environment = f"batch-pages-{create_uuid().replace('-', '')[:12]}"
    trace_ids = [create_uuid().replace("-", "") for _ in range(3)]
    # The v2 endpoint orders by descending start time. The last trace has
    # enough sibling roots to fill an entire later page with duplicates.
    with propagate_attributes(environment=environment):
        for trace_id, root_count in zip(trace_ids, [1, 1, 5]):
            for index in range(root_count):
                with client.start_as_current_observation(
                    name=f"pagination-{index}",
                    trace_context={"trace_id": trace_id},
                    input="pagination-input",
                    output="pagination-output",
                ):
                    pass
    client.flush()

    conditions = [
        {
            "type": "string",
            "column": "environment",
            "operator": "=",
            "value": environment,
        }
    ]
    response = wait_for_result(
        lambda: client.api.observations.get_many(
            filter=json.dumps(conditions), limit=100
        ),
        is_result_ready=lambda result: len(result.data) == 7,
        timeout_seconds=90,
    )
    assert len(response.data) == 7, "the regression corpus was not persisted"
    return json.dumps(conditions), set(trace_ids)


@pytest.mark.parametrize("page_size", [1, 2])
def test_trace_pagination_continues_after_a_page_of_duplicate_roots(page_size):
    client = get_client()
    filter_json, expected_trace_ids = _seed_multi_root_corpus()
    root_conditions = json.loads(filter_json) + [
        {
            "type": "boolean",
            "column": "isRootObservation",
            "operator": "=",
            "value": True,
        }
    ]

    # Prove the real server supplied an empty-after-collapse page with a
    # cursor, followed by a previously unseen trace. Do not fabricate pages.
    cursor = None
    raw_seen = set()
    duplicate_page_with_cursor = False
    later_unseen_trace = False
    page_trace_ids = []
    while True:
        response = client.api.observations.get_many(
            filter=json.dumps(root_conditions), limit=page_size, cursor=cursor
        )
        current_ids = {row.trace_id for row in response.data}
        page_trace_ids.append([row.trace_id for row in response.data])
        next_cursor = response.meta.cursor if response.meta else None
        if duplicate_page_with_cursor and current_ids - raw_seen:
            later_unseen_trace = True
        if current_ids and not current_ids - raw_seen and next_cursor is not None:
            duplicate_page_with_cursor = True
        raw_seen.update(current_ids)
        if next_cursor is None:
            break
        cursor = next_cursor

    assert raw_seen == expected_trace_ids
    assert duplicate_page_with_cursor and later_unseen_trace, page_trace_ids
    print(f"real root pages: {page_trace_ids}")

    mapped_trace_ids = []

    def mapper(*, item):
        mapped_trace_ids.append(item.trace_id)
        return EvaluatorInputs(input=item.input, output=item.output)

    def evaluator(**kwargs):
        return Evaluation(name="pagination-completeness", value=1.0)

    result = client.run_batched_evaluation(
        scope="traces",
        filter=filter_json,
        mapper=mapper,
        evaluators=[evaluator],
        fetch_batch_size=page_size,
    )
    client.flush()

    assert set(mapped_trace_ids) == expected_trace_ids, (
        "batch evaluation stopped at a duplicate-only page and missed later traces"
    )
    assert len(mapped_trace_ids) == len(expected_trace_ids)
    assert result.total_items_processed == len(expected_trace_ids)
    assert result.total_scores_created == len(expected_trace_ids)
    assert result.completed
    assert not result.has_more_items
