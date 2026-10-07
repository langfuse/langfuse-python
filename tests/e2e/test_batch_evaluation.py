"""End-to-end tests for run_batched_evaluation against a Langfuse server.

Every run is restricted to a corpus seeded by this module (filtered by a unique
tag), so assertions do not depend on other data in the project. Runner logic
that does not need a server is covered in tests/unit/test_batch_evaluation.py.
"""

import json
from dataclasses import dataclass
from typing import Any, List

import pytest

from langfuse import get_client, propagate_attributes
from langfuse.api import ObservationV2
from langfuse.batch_evaluation import EvaluatorInputs
from langfuse.experiment import Evaluation
from tests.support.utils import create_uuid, get_api, wait_for_result

TRACE_COUNT = 4


@dataclass
class Corpus:
    tag: str
    filter: str
    trace_ids: List[str]
    root_ids: List[str]
    child_ids: List[str]


def _tag_filter(tag: str) -> str:
    return json.dumps(
        [
            {
                "type": "arrayOptions",
                "column": "tags",
                "operator": "any of",
                "value": [tag],
            }
        ]
    )


def _wait_for_observations(filter_json: str, expected_count: int) -> List[Any]:
    api = get_api(retry=False)
    response = wait_for_result(
        lambda: api.observations.get_many(filter=filter_json, limit=100),
        is_result_ready=lambda r: len(r.data) >= expected_count,
    )
    return list(response.data)


def _wait_for_scores(**kwargs: Any) -> List[Any]:
    api = get_api(retry=False)
    response = wait_for_result(
        lambda: api.scores_v3.get_many_v3(fields="details,subject", **kwargs),
        is_result_ready=lambda r: len(r.data) > 0,
    )
    return list(response.data)


@pytest.fixture(scope="module")
def corpus() -> Corpus:
    langfuse = get_client()
    tag = f"batch-eval-{create_uuid()}"
    trace_ids, root_ids, child_ids = [], [], []

    for index in range(TRACE_COUNT):
        with langfuse.start_as_current_observation(name=f"{tag}-root") as root:
            with propagate_attributes(tags=[tag]):
                with langfuse.start_as_current_observation(
                    as_type="generation",
                    name=f"{tag}-child",
                    input={"question": index},
                    output=f"child answer {index}",
                ) as child:
                    child_ids.append(child.id)
                root.update(input={"question": index}, output=f"answer {index}")
            trace_ids.append(root.trace_id)
            root_ids.append(root.id)

    langfuse.flush()
    _wait_for_observations(_tag_filter(tag), 2 * TRACE_COUNT)

    return Corpus(
        tag=tag,
        filter=_tag_filter(tag),
        trace_ids=trace_ids,
        root_ids=root_ids,
        child_ids=child_ids,
    )


def io_mapper(*, item: ObservationV2) -> EvaluatorInputs:
    return EvaluatorInputs(
        input=item.input,
        output=item.output,
        metadata={"trace_id": item.trace_id},
    )


def length_evaluator(*, output, **kwargs):
    return Evaluation(name="length", value=float(len(output or "")))


def test_evaluates_every_observation(corpus):
    seen: List[ObservationV2] = []

    def recording_mapper(*, item):
        seen.append(item)
        return io_mapper(item=item)

    result = get_client().run_batched_evaluation(
        mapper=recording_mapper,
        evaluators=[length_evaluator],
        filter=corpus.filter,
    )

    assert result.completed is True
    assert result.has_more_items is False
    assert result.resume_token is None
    assert result.total_items_fetched == 2 * TRACE_COUNT
    assert result.total_items_processed == 2 * TRACE_COUNT
    assert result.total_scores_created == 2 * TRACE_COUNT
    assert set(result.item_evaluations) == set(corpus.root_ids + corpus.child_ids)

    by_id = {item.id: item for item in seen}
    child = by_id[corpus.child_ids[0]]
    assert isinstance(child.input, str)
    assert json.loads(child.input) == {"question": 0}
    assert child.output == "child answer 0"


def test_observation_scores_are_attached_to_observations(corpus):
    score_name = f"obs-score-{create_uuid()}"
    child_filter = json.loads(corpus.filter) + [
        {
            "type": "string",
            "column": "id",
            "operator": "=",
            "value": corpus.child_ids[1],
        }
    ]

    def observation_evaluator(**kwargs):
        return Evaluation(name=score_name, value=0.5, comment="ok")

    result = get_client().run_batched_evaluation(
        mapper=io_mapper,
        evaluators=[observation_evaluator],
        filter=json.dumps(child_filter),
        metadata={"run": "e2e"},
    )

    assert result.total_items_processed == 1

    scores = _wait_for_scores(
        trace_id=corpus.trace_ids[1],
        observation_id=corpus.child_ids[1],
        name=score_name,
    )
    assert len(scores) == 1
    assert scores[0].subject.kind == "observation"
    assert scores[0].subject.id == corpus.child_ids[1]
    assert scores[0].subject.trace_id == corpus.trace_ids[1]
    assert scores[0].comment == "ok"
    assert scores[0].metadata == {"run": "e2e"}


def test_max_items_then_resume_covers_corpus_exactly_once(corpus):
    langfuse = get_client()
    run_kwargs: Any = {
        "mapper": io_mapper,
        "evaluators": [length_evaluator],
        "filter": corpus.filter,
        "fetch_batch_size": 2,
    }

    first = langfuse.run_batched_evaluation(max_items=3, **run_kwargs)

    assert first.total_items_fetched == 3
    assert first.has_more_items is True
    assert first.resume_token is not None
    assert first.resume_token.cursor is not None

    second = langfuse.run_batched_evaluation(
        resume_from=first.resume_token, **run_kwargs
    )

    assert second.completed is True
    assert second.resume_token is None
    assert set(first.item_evaluations).isdisjoint(second.item_evaluations)
    assert set(first.item_evaluations) | set(second.item_evaluations) == set(
        corpus.root_ids + corpus.child_ids
    )


def test_fields_control_populated_field_groups(corpus):
    seen: List[ObservationV2] = []

    def recording_mapper(*, item):
        seen.append(item)
        return EvaluatorInputs(input=None, output=None)

    get_client().run_batched_evaluation(
        mapper=recording_mapper,
        evaluators=[length_evaluator],
        filter=corpus.filter,
        fields="core,basic",
        max_items=1,
    )

    assert len(seen) == 1
    assert seen[0].input is None
    assert seen[0].output is None


def test_composite_evaluator_and_failures(corpus):
    def failing_evaluator(**kwargs):
        raise RuntimeError("intentional")

    def composite(*, evaluations, **kwargs):
        return Evaluation(name="composite", value=float(len(evaluations)))

    result = get_client().run_batched_evaluation(
        mapper=io_mapper,
        evaluators=[length_evaluator, failing_evaluator],
        composite_evaluator=composite,
        filter=corpus.filter,
    )

    item_count = 2 * TRACE_COUNT
    assert result.total_items_processed == item_count
    assert result.total_scores_created == item_count
    assert result.total_composite_scores_created == item_count
    assert result.total_evaluations_failed == item_count
    stats = {s.name: s for s in result.evaluator_stats}
    assert stats["failing_evaluator"].failed_runs == item_count


def test_mapper_failures_are_reported_per_item(corpus):
    def failing_mapper(*, item):
        raise ValueError("intentional")

    result = get_client().run_batched_evaluation(
        mapper=failing_mapper,
        evaluators=[length_evaluator],
        filter=corpus.filter,
    )

    assert result.completed is True
    assert result.total_items_failed == 2 * TRACE_COUNT
    assert set(result.failed_item_ids) == set(corpus.root_ids + corpus.child_ids)
    assert result.error_summary == {"ValueError": 2 * TRACE_COUNT}


def test_filter_without_matches_completes_empty():
    result = get_client().run_batched_evaluation(
        mapper=io_mapper,
        evaluators=[length_evaluator],
        filter=_tag_filter(f"nonexistent-{create_uuid()}"),
    )

    assert result.completed is True
    assert result.total_items_fetched == 0
    assert result.has_more_items is False
    assert result.resume_token is None
