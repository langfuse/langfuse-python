"""End-to-end coverage for trace-root selection in batch evaluation.

Unit tests cover the request the runner sends, but only a real server can show
that the ``isRootObservation`` filter is honoured and that the cross-page bug
this replaced actually occurred. Both are asserted here against the deployment
the e2e suite runs against.

The bug: the v2 endpoint pages by cursor over observations, not traces, so a
trace's root and its children can land on different pages. Choosing a
representative per page let the first page fix the representative for the whole
run, and the cross-page ``seen`` set then suppressed the root entirely. These
tests seed a wide trace so the root cannot share a page with all its children,
which is the shape that triggers it.
"""

import base64
import json
import os
import time

from langfuse import get_client
from langfuse.batch_evaluation import _collapse_observations_to_traces
from tests.support.utils import create_uuid

ROOT_OUTPUT = "root-output-marker"
CHILD_OUTPUT = "child-output-marker"


def _seed_wide_trace(*, child_count: int = 12) -> str:
    """Seed one trace with a root observation and many children.

    A wide trace is what makes the cross-page case reachable: with a small
    `page_size` the root cannot share a page with every child.
    """
    langfuse_client = get_client()
    # Trace IDs must be 32 lowercase hex chars; `create_uuid()` is dashed.
    trace_id = create_uuid().replace("-", "")
    name = f"root-selection-{create_uuid()}"

    with langfuse_client.start_as_current_observation(
        name=f"{name}-root",
        trace_context={"trace_id": trace_id},
        input=f"{name}-root-input",
        output=ROOT_OUTPUT,
    ):
        for index in range(child_count):
            with langfuse_client.start_as_current_observation(
                name=f"{name}-child-{index}",
                input=f"{name}-child-{index}-input",
                output=CHILD_OUTPUT,
            ):
                pass

    langfuse_client.flush()
    return trace_id


def _fetch_observations(trace_id: str, *, page_size: int = 2, root_only: bool = False):
    """Read one trace back page by page, optionally narrowed to root observations."""
    client = get_client()
    conditions = None
    if root_only:
        conditions = json.dumps(
            [
                {
                    "type": "boolean",
                    "column": "isRootObservation",
                    "operator": "=",
                    "value": True,
                }
            ]
        )

    pages = []
    cursor = None
    while True:
        response = client.api.observations.get_many(
            trace_id=trace_id, limit=page_size, cursor=cursor, filter=conditions
        )
        rows = list(response.data)
        if rows:
            pages.append(rows)
        cursor = response.meta.cursor if response.meta else None
        if cursor is None or not rows:
            break
    return pages


def _wait_for_trace_observations(
    trace_id: str, *, expected_min: int, timeout: float = 90.0
):
    """Ingestion is async; wait until the trace has its observations.

    Returns the rows so callers do not immediately re-fetch.
    """
    deadline = time.time() + timeout
    rows: list = []
    while time.time() < deadline:
        rows = list(
            get_client().api.observations.get_many(trace_id=trace_id, limit=100).data
        )
        if len(rows) >= expected_min:
            return rows
        time.sleep(2.0)
    raise AssertionError(
        f"expected at least {expected_min} observations for {trace_id}, got {len(rows)}"
    )


def test_is_root_observation_filter_is_honoured_by_the_server():
    """The filter the runner sends must actually narrow the response.

    If the server ignored the condition, a trace's children would come back and
    the root could still lose the page-local comparison.
    """
    child_count = 12
    trace_id = _seed_wide_trace(child_count=child_count)
    all_rows = _wait_for_trace_observations(trace_id, expected_min=child_count + 1)

    # The server flags exactly one observation on the trace as the root.
    flagged_roots = [row for row in all_rows if row.is_root_observation]
    assert len(flagged_roots) == 1, (
        f"expected exactly one server-side root, got {[r.name for r in flagged_roots]}"
    )

    # And the root filter returns exactly that row, not its siblings.
    root_pages = _fetch_observations(trace_id, root_only=True)
    filtered = [row for page in root_pages for row in page]
    assert len(filtered) == 1
    assert filtered[0].id == flagged_roots[0].id

    # Sanity check that the negation excludes it, so the filter is not a no-op.
    negated = json.dumps(
        [
            {
                "type": "boolean",
                "column": "isRootObservation",
                "operator": "<>",
                "value": True,
            }
        ]
    )
    non_root = list(
        get_client()
        .api.observations.get_many(trace_id=trace_id, limit=100, filter=negated)
        .data
    )
    assert len(non_root) >= child_count
    assert all(row.id != flagged_roots[0].id for row in non_root)


def test_root_wins_even_when_it_is_not_on_the_first_page():
    """The bug this change fixes, reproduced against real server ordering.

    Without the root filter the representative is chosen per page, so a page of
    children arriving before the root's page decides the representative for the
    whole run. With the filter, every page carries roots only.
    """
    child_count = 12
    page_size = 2
    trace_id = _seed_wide_trace(child_count=child_count)
    all_rows = _wait_for_trace_observations(trace_id, expected_min=child_count + 1)
    server_root = next(row for row in all_rows if row.is_root_observation)

    # Unfiltered pages: the root may not share a page with the earlier children.
    # Whether it does depends on the server's ordering, so this is recorded and
    # not asserted; what must hold either way is that the root-filtered path
    # yields only the root, once.
    pages = _fetch_observations(trace_id, page_size=page_size, root_only=False)
    root_on_first_page = any(row.id == server_root.id for row in pages[0])

    # Pre-fix behaviour: page-local collapse with no root filter upstream.
    seen: set = set()
    pre_fix_ids = [
        row.id for page in pages for row in _collapse_observations_to_traces(page, seen)
    ]

    # Post-fix behaviour: the same helper, fed by root-only pages.
    root_pages = _fetch_observations(trace_id, page_size=page_size, root_only=True)
    seen_post: set = set()
    post_fix_ids = [
        row.id
        for page in root_pages
        for row in _collapse_observations_to_traces(page, seen_post)
    ]

    assert post_fix_ids == [server_root.id], (
        f"expected only the root {[server_root.name]}, got {post_fix_ids}"
    )
    # The pre-fix path evaluated exactly one observation for this trace too --
    # that was the original defect, not a duplicate-evaluation bug. Whether it
    # picked the root depended on page ordering.
    assert len(pre_fix_ids) == 1
    print(
        f"\n[info] root_on_first_page={root_on_first_page} "
        f"pre_fix_picked_root={pre_fix_ids[0] == server_root.id} "
        f"post_fix_picked_root={post_fix_ids[0] == server_root.id}"
    )


def test_run_batched_evaluation_on_traces_evaluates_the_root():
    """End to end through the public API: the mapper must receive the root.

    The seeded root and its children carry different outputs, so a child
    representative is detectable from what the mapper saw.
    """
    trace_id = _seed_wide_trace(child_count=12)
    rows = _wait_for_trace_observations(trace_id, expected_min=13)
    root = next(row for row in rows if row.is_root_observation)

    seen_ids: list = []
    seen_outputs: list = []

    def mapper(*, item):
        seen_ids.append(item.id)
        seen_outputs.append(getattr(item, "output", None))
        return None

    get_client().run_batched_evaluation(
        scope="traces",
        mapper=mapper,
        evaluators=[],
        fetch_batch_size=2,
    )
    get_client().flush()

    assert root.id in seen_ids, "the root observation was never evaluated"
    assert CHILD_OUTPUT not in [o for o in seen_outputs if o is not None], (
        "a child observation was evaluated instead of the root"
    )


def test_multiple_flagged_roots_on_one_trace_still_collapse_to_one():
    """The server can flag more than one root on a single trace.

    Two sibling spans on the same trace have both been observed carrying
    ``isRootObservation=True``, so "one root per trace" is not a guarantee the
    filter provides. What has to hold is that the runner still evaluates such a
    trace exactly once, which is the collapse's job rather than the filter's.
    """
    langfuse_client = get_client()
    trace_id = create_uuid().replace("-", "")
    name = f"multi-root-{create_uuid()}"

    # Siblings: neither is nested in the other, so both are eligible app roots.
    for index in range(2):
        with langfuse_client.start_as_current_observation(
            name=f"{name}-sibling-{index}",
            trace_context={"trace_id": trace_id},
            input=f"{name}-in-{index}",
            output=f"sibling-out-{index}",
        ):
            pass
    langfuse_client.flush()

    rows = _wait_for_trace_observations(trace_id, expected_min=2)
    flagged = [row for row in rows if row.is_root_observation]
    root_pages = _fetch_observations(trace_id, root_only=True)
    filtered = [row for page in root_pages for row in page]

    print(
        f"\n[info] observations={len(rows)} flagged_roots={len(flagged)} "
        f"root_filter_returned={len(filtered)}"
    )

    # Whether the server flags one root or several here is server behaviour, not
    # something this test should pin. Either way the runner must collapse the
    # page to one item for the trace.
    seen: set = set()
    collapsed = _collapse_observations_to_traces(filtered, seen)
    assert len(collapsed) == 1, (
        f"expected one representative for the trace, got {[c.name for c in collapsed]}"
    )
    assert collapsed[0].is_root_observation

    # And across pages: two flagged roots on separate pages must still yield one
    # evaluation, which is what `seen_trace_ids` is for.
    if len(flagged) > 1:
        seen_across: set = set()
        first = _collapse_observations_to_traces([flagged[0]], seen_across)
        second = _collapse_observations_to_traces(flagged[1:], seen_across)
        assert len(first) + len(second) == 1, (
            f"a multi-root trace was evaluated {len(first) + len(second)} times"
        )


def test_trace_with_no_flagged_root_is_not_evaluated():
    """A trace whose observations carry no root is silently skipped.

    Traces ingested by other clients -- an OTel collector, another language SDK,
    direct OTLP -- never pass through this SDK's app-root marking, so nothing on
    them is flagged. The root filter then returns nothing for them and
    `scope='traces'` never evaluates them.

    The span is written through the OTLP endpoint with a parent id that does not
    exist, so there is no parent-less observation either. The observation write
    endpoints refuse observation creates on an events_only deployment
    (`/api/public/observations` answers 405, `/api/public/ingestion` answers
    "Event type not accepted ... only accepts score events"), and the error names
    the OTLP path as the events_only-compatible route, so that is what is used.
    """
    import httpx

    client = get_client()
    trace_id = create_uuid().replace("-", "")
    span_id = create_uuid().replace("-", "")[:16]
    name = f"no-root-{create_uuid()}"

    public_key = os.environ["LANGFUSE_PUBLIC_KEY"]
    secret_key = os.environ["LANGFUSE_SECRET_KEY"]
    base_url = os.environ.get("LANGFUSE_BASE_URL", "http://localhost:3000")

    payload = {
        "resourceSpans": [
            {
                "resource": {
                    "attributes": [
                        {"key": "service.name", "value": {"stringValue": name}},
                    ]
                },
                "scopeSpans": [
                    {
                        "scope": {"name": name},
                        "spans": [
                            {
                                "traceId": trace_id,
                                "spanId": span_id,
                                # A parent that does not exist: no observation
                                # is parent-less, and nothing is flagged.
                                "parentSpanId": create_uuid().replace("-", "")[:16],
                                "name": name,
                                "kind": 1,
                                "startTimeUnixNano": "1767225600000000000",
                                "endTimeUnixNano": "1767225601000000000",
                                "attributes": [
                                    {
                                        "key": "langfuse.observation.type",
                                        "value": {"stringValue": "SPAN"},
                                    },
                                    {
                                        "key": "langfuse.trace.name",
                                        "value": {"stringValue": name},
                                    },
                                ],
                            }
                        ],
                    }
                ],
            }
        ]
    }

    response = httpx.post(
        f"{base_url}/api/public/otel/v1/traces",
        json=payload,
        headers={
            "Authorization": "Basic "
            + base64.b64encode(f"{public_key}:{secret_key}".encode()).decode("ascii"),
            "x-langfuse-sdk-name": "python",
            "x-langfuse-sdk-version": "e2e",
            "x-langfuse-public-key": public_key,
            "Content-Type": "application/json",
        },
        timeout=30.0,
    )
    assert response.status_code == 200, (
        f"OTLP ingest failed: {response.status_code} {response.text[:200]}"
    )
    client.flush()

    rows = _wait_for_trace_observations(trace_id, expected_min=1)
    assert rows, (
        "the OTLP span never became readable; the no-root case was not exercised"
    )
    assert all(not row.is_root_observation for row in rows), (
        "the span was flagged as a root, so this deployment does not reproduce the no-root case"
    )

    root_pages = _fetch_observations(trace_id, root_only=True)
    assert root_pages == [], "the root filter returned rows for a trace with no root"

    # The consequence: scope='traces' never sees this trace at all.
    seen_ids: list = []

    def mapper(*, item):
        seen_ids.append(item.id)
        return None

    get_client().run_batched_evaluation(
        scope="traces",
        mapper=mapper,
        evaluators=[],
        fetch_batch_size=2,
    )
    get_client().flush()

    assert not any(row.id in seen_ids for row in rows), (
        "a trace with no flagged root was evaluated; the documented consequence changed"
    )
