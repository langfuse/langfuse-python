"""Delivery of queued scores to POST /api/public/scores."""

import json
import logging
import threading
import time
from datetime import datetime, timezone
from queue import Queue
from typing import Any, Callable, Dict, List, Sequence

import httpx
import pytest
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
from opentelemetry.sdk.trace.sampling import ALWAYS_OFF, ParentBased, TraceIdRatioBased

from langfuse import Langfuse
from langfuse._client.resource_manager import LangfuseResourceManager
from langfuse._task_manager import score_consumer as score_consumer_module
from langfuse._task_manager.score_consumer import ScoreConsumer
from langfuse.api import LangfuseAPI

ACCEPTED = (202, {"message": "Accepted"})


class NoOpSpanExporter(SpanExporter):
    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        pass


Responder = Callable[[httpx.Request], httpx.Response]


def _responder(*responses: Any) -> Responder:
    """Answer requests in order; an exception instance is raised instead."""
    remaining = list(responses)

    def respond(request: httpx.Request) -> httpx.Response:
        response = remaining.pop(0) if len(remaining) > 1 else remaining[0]
        if isinstance(response, Exception):
            raise response
        status, body = response
        return httpx.Response(status, json=body)

    return respond


class Recorder:
    def __init__(self, respond: Responder) -> None:
        self.requests: List[httpx.Request] = []
        self._respond = respond

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self._respond(request)

    @property
    def bodies(self) -> List[Any]:
        return [json.loads(request.content) for request in self.requests]


def _score(name: str, **fields: Any) -> Dict[str, Any]:
    return {
        "id": f"id-{name}",
        "traceId": "a" * 32,
        "name": name,
        "value": 1.0,
        **fields,
    }


@pytest.fixture(autouse=True)
def no_backoff_sleep(monkeypatch):
    monkeypatch.setattr("backoff._sync.time.sleep", lambda _: None)


def _consumer(recorder: Recorder, queue: Queue, **kwargs: Any) -> ScoreConsumer:
    api = LangfuseAPI(
        base_url="http://langfuse.test",
        username="pk",
        password="sk",
        httpx_client=httpx.Client(transport=httpx.MockTransport(recorder)),
    )
    kwargs.setdefault("flush_interval", 0.05)
    return ScoreConsumer(score_queue=queue, identifier=0, api_client=api, **kwargs)


def _drain(consumer: ScoreConsumer, queue: Queue) -> None:
    while queue.unfinished_tasks:
        consumer.upload()


def test_sends_queued_scores_as_one_list(caplog):
    recorder = Recorder(_responder(ACCEPTED))
    queue: Queue = Queue()
    for name in ("a", "b", "c"):
        queue.put(_score(name))

    with caplog.at_level(logging.ERROR, logger="langfuse"):
        _drain(_consumer(recorder, queue), queue)

    assert [r.url.path for r in recorder.requests] == ["/api/public/scores"]
    assert [score["name"] for score in recorder.bodies[0]] == ["a", "b", "c"]
    assert caplog.records == []


def test_splits_batches_at_flush_at():
    recorder = Recorder(_responder(ACCEPTED))
    queue: Queue = Queue()
    for name in ("a", "b", "c"):
        queue.put(_score(name))

    _drain(_consumer(recorder, queue, flush_at=2), queue)

    assert [[s["name"] for s in body] for body in recorder.bodies] == [
        ["a", "b"],
        ["c"],
    ]


def test_splits_batches_at_byte_limit_without_reordering(monkeypatch):
    recorder = Recorder(_responder(ACCEPTED))
    queue: Queue = Queue()
    scores = [_score(name, comment="x" * 100) for name in ("a", "b", "c")]
    for score in scores:
        queue.put(score)
    item_size = len(json.dumps(scores[0]).encode())
    # Room for two items, their separator and the list brackets, not three.
    monkeypatch.setattr(
        score_consumer_module, "MAX_BATCH_SIZE_BYTES", 2 * item_size + 3
    )

    _drain(_consumer(recorder, queue), queue)

    assert [[s["name"] for s in body] for body in recorder.bodies] == [
        ["a", "b"],
        ["c"],
    ]
    assert all(len(r.content) <= 2 * item_size + 3 for r in recorder.requests)


def test_drops_score_larger_than_byte_limit(monkeypatch, caplog):
    recorder = Recorder(_responder(ACCEPTED))
    queue: Queue = Queue()
    queue.put(_score("huge", comment="x" * 500))
    queue.put(_score("small"))
    monkeypatch.setattr(score_consumer_module, "MAX_BATCH_SIZE_BYTES", 200)

    with caplog.at_level(logging.ERROR, logger="langfuse"):
        _drain(_consumer(recorder, queue), queue)

    assert [[s["name"] for s in body] for body in recorder.bodies] == [["small"]]
    assert "huge" in caplog.text


def test_partial_success_is_logged_and_not_retried(caplog):
    recorder = Recorder(
        _responder(
            (
                207,
                {
                    "accepted": 1,
                    "rejected": 1,
                    "errors": [{"message": "b: value must be numeric"}],
                },
            )
        )
    )
    queue: Queue = Queue()
    queue.put(_score("a"))
    queue.put(_score("b", value="not-a-number"))

    with caplog.at_level(logging.ERROR, logger="langfuse"):
        _drain(_consumer(recorder, queue), queue)

    assert len(recorder.requests) == 1
    assert "accepted 1 and rejected 1 of 2 scores" in caplog.text
    assert "b: value must be numeric" in caplog.text


def test_single_score_response_is_accepted(caplog):
    recorder = Recorder(_responder((200, {"id": "id-a"})))
    queue: Queue = Queue()
    queue.put(_score("a"))

    with caplog.at_level(logging.ERROR, logger="langfuse"):
        _drain(_consumer(recorder, queue), queue)

    assert len(recorder.requests) == 1
    assert caplog.records == []


@pytest.mark.parametrize(
    "failure",
    [
        pytest.param((500, {"message": "boom"}), id="server-error"),
        pytest.param((429, {"message": "slow down"}), id="rate-limited"),
        pytest.param(httpx.ConnectError("connection refused"), id="connect-error"),
        pytest.param(httpx.ReadTimeout("timed out"), id="timeout"),
    ],
)
def test_retries_transient_failures(failure, caplog):
    recorder = Recorder(_responder(failure, ACCEPTED))
    queue: Queue = Queue()
    queue.put(_score("a"))

    with caplog.at_level(logging.ERROR, logger="langfuse"):
        _drain(_consumer(recorder, queue), queue)

    assert len(recorder.requests) == 2
    assert recorder.bodies[0] == recorder.bodies[1]
    assert caplog.records == []


def test_gives_up_after_max_retries(caplog):
    recorder = Recorder(_responder((503, {"message": "unavailable"})))
    queue: Queue = Queue()
    queue.put(_score("a"))

    with caplog.at_level(logging.ERROR, logger="langfuse"):
        _drain(_consumer(recorder, queue, max_retries=3), queue)

    assert len(recorder.requests) == 3
    assert queue.unfinished_tasks == 0
    assert "Service unavailable" in caplog.text


@pytest.mark.parametrize("status", [400, 401, 403, 404])
def test_does_not_retry_client_errors(status, caplog):
    recorder = Recorder(_responder((status, {"message": "nope"})))
    queue: Queue = Queue()
    queue.put(_score("a"))

    with caplog.at_level(logging.ERROR, logger="langfuse"):
        _drain(_consumer(recorder, queue), queue)

    assert len(recorder.requests) == 1
    assert queue.unfinished_tasks == 0
    assert caplog.records


def test_pause_sends_carried_over_score(monkeypatch):
    first_request_started = threading.Event()
    release_first_request = threading.Event()
    accept = _responder(ACCEPTED)

    def respond(request: httpx.Request) -> httpx.Response:
        if not first_request_started.is_set():
            first_request_started.set()
            release_first_request.wait(timeout=5)
        return accept(request)

    recorder = Recorder(respond)
    queue: Queue = Queue()
    scores = [_score(name) for name in ("a", "b")]
    for score in scores:
        queue.put(score)
    monkeypatch.setattr(
        score_consumer_module,
        "MAX_BATCH_SIZE_BYTES",
        len(json.dumps(scores[0]).encode()) + 2,
    )
    consumer = _consumer(recorder, queue)

    consumer.start()
    assert first_request_started.wait(timeout=5)
    # "b" did not fit into the in-flight batch and waits as carry-over.
    consumer.pause()
    release_first_request.set()
    consumer.join(timeout=5)

    assert not consumer.is_alive()
    assert [[s["name"] for s in body] for body in recorder.bodies] == [["a"], ["b"]]


def _sampled_provider(rate: float) -> TracerProvider:
    # An explicit provider, because the global one is set once per process.
    return TracerProvider(sampler=TraceIdRatioBased(rate))


@pytest.fixture
def client_factory():
    clients: List[Langfuse] = []

    def create(recorder: Recorder, **kwargs: Any) -> Langfuse:
        with LangfuseResourceManager._lock:
            LangfuseResourceManager._instances.clear()
        client = Langfuse(
            public_key="pk-score-consumer",
            secret_key="sk-score-consumer",
            base_url="http://langfuse.test",
            httpx_client=httpx.Client(transport=httpx.MockTransport(recorder)),
            span_exporter=NoOpSpanExporter(),
            **kwargs,
        )
        clients.append(client)
        return client

    yield create

    for client in clients:
        client.shutdown()
    with LangfuseResourceManager._lock:
        LangfuseResourceManager._instances.clear()


def test_create_score_posts_to_scores_endpoint(client_factory):
    recorder = Recorder(_responder(ACCEPTED))
    client = client_factory(recorder, environment="staging")
    timestamp = datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)

    client.create_score(
        score_id="score-1",
        name="accuracy",
        value=0.9,
        trace_id="a" * 32,
        observation_id="b" * 16,
        data_type="NUMERIC",
        comment="fine",
        config_id="config-1",
        metadata="not a dict",
        timestamp=timestamp,
    )
    client.create_score(name="session-score", value="good", session_id="session-1")
    client.flush()

    assert {r.url.path for r in recorder.requests} == {"/api/public/scores"}
    sent = [score for body in recorder.bodies for score in body]
    assert sent[0] == {
        "id": "score-1",
        "traceId": "a" * 32,
        "observationId": "b" * 16,
        "name": "accuracy",
        "value": 0.9,
        "dataType": "NUMERIC",
        "comment": "fine",
        "configId": "config-1",
        "environment": "staging",
        "metadata": "not a dict",
        "timestamp": "2026-01-02T03:04:05Z",
    }
    assert sent[1]["sessionId"] == "session-1"
    assert sent[1]["name"] == "session-score"
    assert "traceId" not in sent[1]
    assert sent[1]["timestamp"].endswith("Z")


@pytest.fixture
def non_utc_local_timezone(monkeypatch):
    # Makes treating naive values as local time observable.
    monkeypatch.setenv("TZ", "America/New_York")
    time.tzset()
    yield
    monkeypatch.undo()
    time.tzset()


def test_naive_score_timestamp_is_sent_as_utc(client_factory, non_utc_local_timezone):
    recorder = Recorder(_responder(ACCEPTED))
    client = client_factory(recorder)

    client.create_score(
        name="naive",
        value=1.0,
        session_id="session-1",
        timestamp=datetime(2026, 1, 2, 3, 4, 5),
    )
    client.flush()

    sent = [score for body in recorder.bodies for score in body]
    assert sent[0]["timestamp"] == "2026-01-02T03:04:05Z"


def test_scores_follow_trace_sampling(client_factory):
    recorder = Recorder(_responder(ACCEPTED))
    client = client_factory(recorder, tracer_provider=_sampled_provider(0.5))

    # The sampler keeps a trace when its low 64 bits fall below rate * 2^64.
    client.create_score(name="sampled", value=1.0, trace_id="0" * 32)
    client.create_score(name="dropped", value=1.0, trace_id="f" * 32)
    client.create_score(name="session", value=1.0, session_id="session-1")
    client.create_score(name="dataset-run", value=1.0, dataset_run_id="run-1")
    client.flush()

    sent = [score["name"] for body in recorder.bodies for score in body]
    assert sent == ["sampled", "session", "dataset-run"]


def test_scores_for_sampled_out_traces_are_dropped(client_factory):
    recorder = Recorder(_responder(ACCEPTED))
    client = client_factory(recorder, tracer_provider=_sampled_provider(0.0))

    client.create_score(name="dropped", value=1.0, trace_id="0" * 32)
    with client.start_as_current_observation(name="span") as span:
        span.score(name="dropped-span", value=1.0)
        span.score_trace(name="dropped-trace", value=1.0)
    client.flush()

    assert recorder.requests == []


def test_scores_are_kept_when_sampler_decision_depends_on_more_than_trace_id(
    client_factory,
):
    recorder = Recorder(_responder(ACCEPTED))
    provider = TracerProvider(sampler=ParentBased(ALWAYS_OFF))
    client = client_factory(recorder, tracer_provider=provider)

    client.create_score(name="kept", value=1.0, trace_id="0" * 32)
    client.flush()

    sent = [score["name"] for body in recorder.bodies for score in body]
    assert sent == ["kept"]
