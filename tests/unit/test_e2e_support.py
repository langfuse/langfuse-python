from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from langfuse.api import ObservationV2
from langfuse.api.commons.errors.not_found_error import NotFoundError
from tests.support.retry import retry_until_ready
from tests.support.utils import (
    TraceSnapshot,
    get_api,
    get_observations,
    normalize_observation,
    user_metadata,
    wait_for_observations,
    wait_for_scores,
    wait_for_trace_snapshot,
)

START = datetime(2024, 1, 1, tzinfo=timezone.utc)


def _observation(index: int = 0, **fields) -> ObservationV2:
    defaults = {
        "id": f"obs-{index}",
        "trace_id": "trace-123",
        "start_time": START + timedelta(seconds=index),
        "project_id": "project",
        "type": "SPAN",
        "is_root_observation": False,
    }
    return ObservationV2(**{**defaults, **fields})


def _page(data, cursor=None):
    return SimpleNamespace(data=data, meta=SimpleNamespace(cursor=cursor))


def _install_client(monkeypatch, **services):
    monkeypatch.setattr("tests.support.retry.sleep", lambda _: None)
    client = SimpleNamespace(**services)
    monkeypatch.setattr("tests.support.utils.LangfuseAPI", lambda **_: client)


def test_get_api_retries_not_found(monkeypatch):
    attempts = {"count": 0}

    def get_many(**kwargs):
        attempts["count"] += 1

        if attempts["count"] < 3:
            raise NotFoundError(
                body={
                    "error": "LangfuseNotFoundError",
                    "message": "Observations not found within authorized project",
                }
            )

        return _page([kwargs["trace_id"]])

    _install_client(monkeypatch, observations=SimpleNamespace(get_many=get_many))

    response = get_api().observations.get_many(trace_id="trace-123")

    assert response.data == ["trace-123"]
    assert attempts["count"] == 3


def test_get_api_retries_filtered_lists(monkeypatch):
    attempts = {"count": 0}

    def get_many(**kwargs):
        attempts["count"] += 1
        return _page([] if attempts["count"] < 3 else [kwargs["name"]])

    _install_client(monkeypatch, observations=SimpleNamespace(get_many=get_many))

    response = get_api().observations.get_many(name="ready-observation")

    assert response.data == ["ready-observation"]
    assert attempts["count"] == 3


def test_get_api_retry_can_be_disabled(monkeypatch):
    attempts = {"count": 0}

    def get_many(**kwargs):
        attempts["count"] += 1
        return _page([])

    _install_client(monkeypatch, observations=SimpleNamespace(get_many=get_many))

    response = get_api(retry=False).observations.get_many(name="missing")

    assert response.data == []
    assert attempts["count"] == 1


def test_normalize_observation_parses_io_and_maps_empty_strings_to_none():
    observation = normalize_observation(
        _observation(
            input='{"question": "hi"}',
            output="plain text",
            name="",
            session_id="",
            user_id="user-1",
        )
    )

    assert observation.input == {"question": "hi"}
    assert observation.output == "plain text"
    assert observation.name is None
    assert observation.session_id is None
    assert observation.user_id == "user-1"


def test_user_metadata_drops_server_added_keys():
    observation = _observation(
        metadata={
            "key": "value",
            "scope.name": "langfuse-sdk",
            "resourceAttributes.service.name": "test",
        }
    )

    assert user_metadata(observation) == {"key": "value"}


def test_get_observations_follows_cursor_and_sorts_by_start_time(monkeypatch):
    calls = []
    pages = {
        None: _page([_observation(2), _observation(0)], cursor="next"),
        "next": _page([_observation(1)]),
    }

    def get_many(**kwargs):
        calls.append(kwargs)
        return pages[kwargs["cursor"]]

    _install_client(monkeypatch, observations=SimpleNamespace(get_many=get_many))

    observations = get_observations(trace_id="trace-123")

    assert [o.id for o in observations] == ["obs-0", "obs-1", "obs-2"]
    assert [call["cursor"] for call in calls] == [None, "next"]
    assert all(call["trace_id"] == "trace-123" for call in calls)


def test_wait_for_observations_polls_until_min_count(monkeypatch):
    attempts = {"count": 0}

    def get_many(**kwargs):
        attempts["count"] += 1
        return _page([_observation(i) for i in range(attempts["count"])])

    _install_client(monkeypatch, observations=SimpleNamespace(get_many=get_many))

    observations = wait_for_observations("trace-123", min_count=3)

    assert len(observations) == 3
    assert attempts["count"] == 3


def test_wait_for_trace_snapshot_waits_for_root_and_scores(monkeypatch):
    attempts = {"observations": 0, "scores": 0}

    def get_many(**kwargs):
        attempts["observations"] += 1
        observations = [_observation(1, name="child")]
        if attempts["observations"] >= 2:
            observations.append(_observation(0, name="root", is_root_observation=True))
        return _page(observations)

    def get_many_v3(**kwargs):
        attempts["scores"] += 1
        return _page(["score"] if attempts["scores"] >= 3 else [])

    _install_client(
        monkeypatch,
        observations=SimpleNamespace(get_many=get_many),
        scores_v3=SimpleNamespace(get_many_v3=get_many_v3),
    )

    snapshot = wait_for_trace_snapshot(
        "trace-123",
        min_scores=1,
        is_result_ready=lambda trace: trace.root.name == "root",
    )

    assert snapshot.root.name == "root"
    assert snapshot.scores == ["score"]
    assert attempts["scores"] == 3


def test_trace_snapshot_aggregates_trace_attributes_like_the_platform():
    snapshot = TraceSnapshot(
        id="trace-123",
        observations=[
            _observation(
                0,
                name="root",
                trace_name="root",
                is_root_observation=True,
                input={"q": 1},
                metadata={"root_key": "root", "scope.name": "sdk"},
                tags=["b"],
            ),
            _observation(
                1,
                name="child",
                trace_name="explicit-name",
                session_id="session-1",
                user_id="user-1",
                tags=["a"],
                public=True,
            ),
            _observation(2, name="grandchild", session_id="session-2"),
        ],
    )

    assert snapshot.name == "explicit-name"
    assert snapshot.session_id == "session-2"
    assert snapshot.user_id == "user-1"
    assert snapshot.tags == ["a", "b"]
    assert snapshot.public is True
    assert snapshot.input == {"q": 1}
    assert snapshot.metadata == {"root_key": "root"}


def test_trace_snapshot_name_falls_back_to_root_trace_name():
    snapshot = TraceSnapshot(
        id="trace-123",
        observations=[
            _observation(0, name="root", trace_name="root", is_root_observation=True),
            _observation(1, name="child", trace_name="root"),
        ],
    )

    assert snapshot.name == "root"
    assert snapshot.session_id is None
    assert snapshot.public is False


def test_wait_for_scores_follows_cursor_and_polls(monkeypatch):
    attempts = {"count": 0}

    def get_many_v3(**kwargs):
        attempts["count"] += 1
        if attempts["count"] < 2:
            return _page([])
        if kwargs["cursor"] is None:
            return _page(["score-1"], cursor="next")
        return _page(["score-2"])

    _install_client(monkeypatch, scores_v3=SimpleNamespace(get_many_v3=get_many_v3))

    scores = wait_for_scores(min_count=2, trace_id="trace-123")

    assert scores == ["score-1", "score-2"]


def test_retry_until_ready_clears_stale_error_after_success(monkeypatch):
    monkeypatch.setattr("tests.support.retry.sleep", lambda _: None)

    monotonic_values = iter([0.0, 0.0, 0.05, 0.06, 0.11, 0.11])
    monkeypatch.setattr("tests.support.retry.monotonic", lambda: next(monotonic_values))

    attempts = {"count": 0}

    def operation():
        attempts["count"] += 1

        if attempts["count"] == 1:
            raise NotFoundError(
                body={
                    "error": "LangfuseNotFoundError",
                    "message": "Trace trace-123 not found within authorized project",
                }
            )

        return {"id": "trace-123", "attempt": attempts["count"], "observations": []}

    trace = retry_until_ready(
        operation,
        is_result_ready=lambda _: False,
        timeout_seconds=0.1,
        interval_seconds=0,
    )

    assert trace["id"] == "trace-123"
    assert trace["attempt"] == 3
