"""Delivery of queued scores to POST /api/public/scores."""

import json
import logging
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from decimal import Decimal
from fractions import Fraction
from queue import Queue
from typing import Any, Callable, Dict, List, Sequence

import httpx
import pytest
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
from opentelemetry.sdk.trace.sampling import ALWAYS_OFF, ParentBased, TraceIdRatioBased

from langfuse import Langfuse
from langfuse._client import resource_manager as resource_manager_module
from langfuse._client.resource_manager import LangfuseResourceManager
from langfuse._task_manager import score_consumer as score_consumer_module
from langfuse._task_manager.score_consumer import ScoreConsumer
from langfuse._utils.serializer import EventSerializer
from langfuse.api import CreateScoreRequest, LangfuseAPI

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


def test_reads_flush_at_from_env(monkeypatch):
    monkeypatch.setenv("LANGFUSE_FLUSH_AT", "2")
    recorder = Recorder(_responder(ACCEPTED))
    queue: Queue = Queue()
    for name in ("a", "b", "c"):
        queue.put(_score(name))

    _drain(_consumer(recorder, queue), queue)

    assert [[s["name"] for s in body] for body in recorder.bodies] == [
        ["a", "b"],
        ["c"],
    ]


def test_reads_flush_interval_from_env(monkeypatch):
    monkeypatch.setenv("LANGFUSE_FLUSH_INTERVAL", "0.05")
    recorder = Recorder(_responder(ACCEPTED))
    consumer = ScoreConsumer(
        score_queue=Queue(),
        identifier=0,
        api_client=LangfuseAPI(
            base_url="http://langfuse.test",
            username="pk",
            password="sk",
            httpx_client=httpx.Client(transport=httpx.MockTransport(recorder)),
        ),
    )

    started = time.monotonic()
    consumer.upload()

    # The default window is 1 second.
    assert time.monotonic() - started < 0.5


def test_constructor_flush_settings_win_over_env(monkeypatch):
    monkeypatch.setenv("LANGFUSE_FLUSH_AT", "1")
    recorder = Recorder(_responder(ACCEPTED))
    queue: Queue = Queue()
    for name in ("a", "b", "c"):
        queue.put(_score(name))

    _drain(_consumer(recorder, queue, flush_at=3), queue)

    assert [[s["name"] for s in body] for body in recorder.bodies] == [["a", "b", "c"]]


def test_wake_sends_pending_batch_before_flush_interval():
    recorder = Recorder(_responder(ACCEPTED))
    queue: Queue = Queue()
    consumer = _consumer(recorder, queue, flush_interval=30)
    consumer.start()
    queue.put(_score("a"))

    consumer.wake()
    finished = threading.Event()
    threading.Thread(target=lambda: (queue.join(), finished.set()), daemon=True).start()

    assert finished.wait(timeout=5)
    assert [[s["name"] for s in body] for body in recorder.bodies] == [["a"]]
    assert consumer.is_alive()
    consumer.pause()
    consumer.join(timeout=5)


def test_wake_on_full_queue_does_not_acknowledge_a_score():
    queue: Queue = Queue(maxsize=1)
    consumer = _consumer(Recorder(_responder(ACCEPTED)), queue)
    queue.put(_score("a"))

    consumer.wake()

    assert queue.unfinished_tasks == 1
    assert queue.qsize() == 1


class CountingQueue(Queue):
    def __init__(self) -> None:
        super().__init__()
        self.task_done_calls = 0

    def task_done(self) -> None:
        self.task_done_calls += 1
        super().task_done()


def test_consumer_never_acknowledges_wake_signals(monkeypatch):
    first = _score("a")
    item_size = len(json.dumps(first).encode())
    # Room for one score per batch, so "b" is carried over into the next one.
    monkeypatch.setattr(
        score_consumer_module, "MAX_BATCH_SIZE_BYTES", item_size + 3 + 50
    )
    monkeypatch.setattr(score_consumer_module, "EventSerializer", _FailingSerializer)
    recorder = Recorder(_responder(ACCEPTED))
    queue = CountingQueue()
    consumer = _consumer(recorder, queue)
    real_scores = [
        first,
        _score("b"),
        _score("too-large", comment="x" * 500),
        _score("not-serializable"),
        _score("c"),
    ]
    for score in real_scores:
        queue.put(score)
        consumer.wake()
    while queue.qsize() > 1:
        consumer.upload()
    # The last wake signal is still queued, and "c" may be carried over: the
    # shutdown path takes both.
    consumer.pause()
    consumer.run()

    assert queue.unfinished_tasks == 0
    wakes = len(real_scores) + 1
    assert queue.task_done_calls == len(real_scores) + wakes
    assert [s["name"] for body in recorder.bodies for s in body] == ["a", "b", "c"]


class _FailingSerializer(EventSerializer):
    def encode(self, o: Any) -> str:
        if isinstance(o, dict) and o.get("name") == "not-serializable":
            raise TypeError("cannot serialize")
        return super().encode(o)


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
        # The global provider may carry a sampler set by another test.
        kwargs.setdefault("tracer_provider", TracerProvider())
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


def test_flush_sends_pending_score_without_waiting_for_flush_interval(
    client_factory,
):
    recorder = Recorder(_responder(ACCEPTED))
    client = client_factory(recorder, flush_interval=5)

    client.create_score(name="a", value=1.0, trace_id="a" * 32)
    started = time.monotonic()
    client.flush()
    elapsed = time.monotonic() - started

    assert [s["name"] for body in recorder.bodies for s in body] == ["a"]
    assert elapsed < 1


def test_flush_after_shutdown_returns(client_factory):
    client = client_factory(Recorder(_responder(ACCEPTED)))
    client.shutdown()

    flush = threading.Thread(target=client.flush, daemon=True)
    flush.start()
    flush.join(timeout=2)

    assert not flush.is_alive()


def test_score_after_shutdown_is_dropped_and_flush_returns(client_factory, caplog):
    recorder = Recorder(_responder(ACCEPTED))
    client = client_factory(recorder)
    client.shutdown()
    score_queue = client._resources._score_queue
    enqueued: List[Any] = []
    put = score_queue.put

    def recording_put(item: Any, *args: Any, **kwargs: Any) -> None:
        enqueued.append(item)
        put(item, *args, **kwargs)

    score_queue.put = recording_put  # type: ignore[method-assign]

    with caplog.at_level(logging.WARNING, logger="langfuse"):
        client.create_score(name="late", value=1.0, trace_id="a" * 32)

    assert enqueued == []
    assert any("shut down" in r.getMessage() for r in caplog.records)

    flush = threading.Thread(target=client.flush, daemon=True)
    flush.start()
    flush.join(timeout=2)

    assert not flush.is_alive()
    assert recorder.requests == []


def test_flush_returns_when_consumer_stops_after_liveness_check(client_factory):
    client = client_factory(Recorder(_responder(ACCEPTED)))
    consumer = client._resources._score_consumers[0]
    wake = consumer.wake

    def stop_then_wake() -> None:
        # Shutdown stops the consumer between flush()'s liveness check and its
        # wake signal, so nothing acknowledges the signal.
        consumer.running = False
        wake()
        consumer.join(timeout=5)
        wake()

    consumer.wake = stop_then_wake  # type: ignore[method-assign]

    flush = threading.Thread(target=client.flush, daemon=True)
    flush.start()
    flush.join(timeout=2)
    flush_hung = flush.is_alive()

    assert not consumer.is_alive()
    assert not flush_hung
    # The orphaned signal stays queued but is not an unfinished task.
    assert client._resources._score_queue.qsize() == 1


@pytest.mark.filterwarnings("error::pytest.PytestUnhandledThreadExceptionWarning")
def test_flush_racing_shutdown_does_not_hang(client_factory):
    client = client_factory(Recorder(_responder(ACCEPTED)), flush_interval=5)
    for name in ("a", "b", "c"):
        client.create_score(name=name, value=1.0, trace_id="a" * 32)

    threads = [threading.Thread(target=client.flush, daemon=True) for _ in range(4)]
    threads.append(threading.Thread(target=client.shutdown, daemon=True))
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)

    assert not any(thread.is_alive() for thread in threads)
    assert client._resources._score_queue.empty() or not any(
        c.is_alive() for c in client._resources._score_consumers
    )


@pytest.mark.filterwarnings("error::pytest.PytestUnhandledThreadExceptionWarning")
@pytest.mark.timeout(60)
@pytest.mark.parametrize(
    ("queue_size", "rounds"), [(100_000, 25), (1, 5)], ids=["roomy", "often-full"]
)
def test_concurrent_flush_returns_only_after_own_score_is_sent(
    client_factory, monkeypatch, queue_size, rounds
):
    monkeypatch.setattr(resource_manager_module, "Queue", lambda _: Queue(queue_size))
    posted = set()
    lock = threading.Lock()
    accept = _responder(ACCEPTED)

    def respond(request: httpx.Request) -> httpx.Response:
        time.sleep(0.001)
        with lock:
            posted.update(s["name"] for s in json.loads(request.content))
        return accept(request)

    # A full queue drops the wake signal, so flush() then waits up to
    # flush_interval; keep it short.
    client = client_factory(Recorder(respond), flush_at=3, flush_interval=0.05)
    score_queue = client._resources._score_queue
    enqueued = set()
    put = score_queue.put

    def recording_put(item: Any, block: bool = True, timeout: Any = None) -> None:
        put(item, block, timeout)
        if isinstance(item, dict):
            with lock:
                enqueued.add(item["name"])

    score_queue.put = recording_put  # type: ignore[method-assign]
    unsent_after_flush = []

    def worker(thread_id: int) -> None:
        for i in range(rounds):
            name = f"{thread_id}-{i}"
            client.create_score(name=name, value=1.0, trace_id="a" * 32)
            client.flush()
            with lock:
                if name in enqueued and name not in posted:
                    unsent_after_flush.append(name)

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    assert not any(thread.is_alive() for thread in threads)
    assert unsent_after_flush == []
    assert enqueued
    assert enqueued <= posted
    assert score_queue.unfinished_tasks == 0
    assert client._resources._score_consumers[0].is_alive()


def test_flush_does_not_wait_for_dropped_scores(client_factory, monkeypatch):
    monkeypatch.setattr(score_consumer_module, "MAX_BATCH_SIZE_BYTES", 10)
    recorder = Recorder(_responder(ACCEPTED))
    client = client_factory(recorder, flush_interval=5)
    client.create_score(name="too-large", value=1.0, trace_id="a" * 32)

    flush = threading.Thread(target=client.flush, daemon=True)
    flush.start()
    flush.join(timeout=1)

    assert not flush.is_alive()
    assert recorder.requests == []


_GEVENT_FLUSH_SCRIPT = """
from gevent import monkey

monkey.patch_all()

import queue

import gevent.queue
import httpx

assert issubclass(queue.Queue, gevent.queue.Queue), queue.Queue

from langfuse import Langfuse

posted = []

def respond(request):
    posted.extend(item["name"] for item in __import__("json").loads(request.content))
    return httpx.Response(202, json={"message": "Accepted"})

client = Langfuse(
    public_key="pk-gevent",
    secret_key="sk-gevent",
    base_url="http://langfuse.test",
    flush_interval=5,
    httpx_client=httpx.Client(transport=httpx.MockTransport(respond)),
)
client.create_score(name="first", value=1, trace_id="a" * 32)
client.flush()
assert posted == ["first"], posted

client.create_score(name="second", value=1, trace_id="a" * 32)
client.shutdown()
assert posted == ["first", "second"], posted
client.flush()
print("ok")
"""


_EXIT_FLUSH_SCRIPT = """
import json
import sys

import httpx

from langfuse import Langfuse

def respond(request):
    for item in json.loads(request.content):
        print("posted", item["name"], flush=True)
    return httpx.Response(202, json={"message": "Accepted"})

client = Langfuse(
    public_key="pk-exit",
    secret_key="sk-exit",
    base_url="http://langfuse.test",
    flush_interval=5,
    httpx_client=httpx.Client(transport=httpx.MockTransport(respond)),
)
client.create_score(name="at-exit", value=1, trace_id="a" * 32)
"""


def test_exit_without_shutdown_sends_pending_scores():
    result = subprocess.run(
        [sys.executable, "-c", _EXIT_FLUSH_SCRIPT],
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == ["posted at-exit"]
    assert "Traceback" not in result.stderr


def test_flush_and_shutdown_work_under_gevent_monkey_patching():
    pytest.importorskip("gevent")

    result = subprocess.run(
        [sys.executable, "-c", _GEVENT_FLUSH_SCRIPT],
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


def test_create_score_payload_matches_request_model_serialization(client_factory):
    recorder = Recorder(_responder(ACCEPTED))
    client = client_factory(recorder, environment="staging")
    timestamp = datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
    cases: List[Dict[str, Any]] = [
        dict(
            name="numeric",
            value=0.9,
            trace_id="a" * 32,
            observation_id="b" * 16,
            data_type="NUMERIC",
            comment="fine",
        ),
        dict(name="int-value", value=1, trace_id="a" * 32),
        dict(name="boolean", value=True, trace_id="a" * 32, data_type="BOOLEAN"),
        dict(name="decimal-value", value=Decimal("0.75"), trace_id="a" * 32),
        dict(name="fraction-value", value=Fraction(3, 4), trace_id="a" * 32),
        dict(
            name="categorical",
            value="good",
            session_id="session-1",
            data_type="CATEGORICAL",
            config_id="config-1",
        ),
        dict(
            name="text",
            value="free text",
            dataset_run_id="run-1",
            data_type="TEXT",
            environment="override",
        ),
        dict(
            name="correction",
            value="corrected output",
            trace_id="a" * 32,
            observation_id="b" * 16,
            data_type="CORRECTION",
        ),
        dict(
            name="dict-metadata",
            value=0.5,
            trace_id="a" * 32,
            metadata={"evaluator": "judge", "nested": {"steps": [1, 2]}},
        ),
        dict(name="list-metadata", value=0.5, trace_id="a" * 32, metadata=["a", 1]),
    ]

    for i, case in enumerate(cases):
        client.create_score(score_id=f"score-{i}", timestamp=timestamp, **case)
    client.flush()

    expected = []
    for i, case in enumerate(cases):
        fields = dict(case)
        metadata = fields.pop("metadata", None)
        fields.setdefault("environment", "staging")
        request = CreateScoreRequest(id=f"score-{i}", timestamp=timestamp, **fields)
        payload = {
            k: v for k, v in request.dict(by_alias=True).items() if v is not None
        }
        if metadata is not None:
            payload["metadata"] = metadata
        expected.append(json.dumps(payload, cls=EventSerializer))
    # Compare JSON text, because 1 == 1.0 == True as Python values.
    sent = [json.dumps(score) for body in recorder.bodies for score in body]
    assert sent == expected


def test_create_score_sends_numpy_values_as_numbers(client_factory):
    np = pytest.importorskip("numpy")
    recorder = Recorder(_responder(ACCEPTED))
    client = client_factory(recorder)

    # e.g. an evaluator returning np.mean(hits) > 0.5 or np.float32(score)
    for value in (np.bool_(True), np.float32(0.5), np.int64(3)):
        client.create_score(name="numpy", value=value, trace_id="a" * 32)
    client.flush()

    sent = [score["value"] for body in recorder.bodies for score in body]
    assert sent == [1.0, 0.5, 3.0]
    assert all(type(value) is float for value in sent)


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
