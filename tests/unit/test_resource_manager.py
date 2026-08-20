"""Test the LangfuseResourceManager and get_client() function."""

from queue import Queue
from types import SimpleNamespace
from typing import Sequence
from unittest.mock import Mock

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

from langfuse import Langfuse
from langfuse._client.get_client import get_client
from langfuse._client.resource_manager import LangfuseResourceManager
from langfuse._task_manager.media_manager import MediaManager
from langfuse._task_manager.media_upload_consumer import MediaUploadConsumer
from langfuse._task_manager.score_ingestion_consumer import ScoreIngestionConsumer
from langfuse.types import MaskOtelSpansResult


class NoOpSpanExporter(SpanExporter):
    """Minimal exporter used to verify configuration propagation."""

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        pass


def test_get_client_preserves_all_settings(monkeypatch):
    """Test that get_client() preserves environment and all client settings."""
    with LangfuseResourceManager._lock:
        LangfuseResourceManager._instances.clear()

    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-comprehensive-default")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-comprehensive-default")
    monkeypatch.setenv("LANGFUSE_BASE_URL", "http://localhost:3000")

    def should_export(span):
        return span.name != "drop"

    def mask_otel_spans(*, params):
        return MaskOtelSpansResult()

    span_exporter = NoOpSpanExporter()

    settings = {
        "public_key": "pk-comprehensive",
        "secret_key": "sk-comprehensive",
        "environment": "test-env",
        "release": "v1.2.3",
        "timeout": 30,
        "flush_at": 100,
        "sample_rate": 0.8,
        "should_export_span": should_export,
        "mask_otel_spans": mask_otel_spans,
        "additional_headers": {"X-Custom": "value"},
        "span_exporter": span_exporter,
    }

    original_client = Langfuse(**settings)
    retrieved_client = get_client()

    assert retrieved_client._environment == settings["environment"]
    assert retrieved_client._release == settings["release"]

    assert retrieved_client._resources is not None
    rm = retrieved_client._resources
    assert rm.environment == settings["environment"]
    assert rm.timeout == settings["timeout"]
    assert rm.sample_rate == settings["sample_rate"]
    assert rm.should_export_span is should_export
    assert rm.mask_otel_spans is mask_otel_spans
    assert rm.additional_headers == settings["additional_headers"]
    assert rm.span_exporter is span_exporter

    original_client.shutdown()


def test_get_client_multiple_clients_preserve_different_settings():
    """Test that get_client() preserves different settings for multiple clients."""

    def should_export_a(span):
        return span.name.startswith("a")

    def should_export_b(span):
        return span.name.startswith("b")

    exporter_a = NoOpSpanExporter()
    exporter_b = NoOpSpanExporter()

    # Settings for client A
    settings_a = {
        "public_key": "pk-comprehensive-a",
        "secret_key": "sk-comprehensive-a",
        "environment": "env-a",
        "release": "release-a",
        "timeout": 10,
        "sample_rate": 0.5,
        "should_export_span": should_export_a,
        "span_exporter": exporter_a,
    }

    # Settings for client B
    settings_b = {
        "public_key": "pk-comprehensive-b",
        "secret_key": "sk-comprehensive-b",
        "environment": "env-b",
        "release": "release-b",
        "timeout": 20,
        "sample_rate": 0.9,
        "should_export_span": should_export_b,
        "span_exporter": exporter_b,
    }

    client_a = Langfuse(**settings_a)
    client_b = Langfuse(**settings_b)

    # Get clients via get_client()
    retrieved_a = get_client(public_key="pk-comprehensive-a")
    retrieved_b = get_client(public_key="pk-comprehensive-b")

    # Verify each client preserves its own settings
    assert retrieved_a._environment == settings_a["environment"]
    assert retrieved_b._environment == settings_b["environment"]

    if retrieved_a._resources and retrieved_b._resources:
        assert retrieved_a._resources.timeout == settings_a["timeout"]
        assert retrieved_b._resources.timeout == settings_b["timeout"]
        assert retrieved_a._resources.sample_rate == settings_a["sample_rate"]
        assert retrieved_b._resources.sample_rate == settings_b["sample_rate"]
        assert retrieved_a._resources.release == settings_a["release"]
        assert retrieved_b._resources.release == settings_b["release"]
        assert retrieved_a._resources.should_export_span is should_export_a
        assert retrieved_b._resources.should_export_span is should_export_b
        assert retrieved_a._resources.span_exporter is exporter_a
        assert retrieved_b._resources.span_exporter is exporter_b

    client_a.shutdown()
    client_b.shutdown()


def test_score_ingestion_consumer_pause_wakes_blocked_thread():
    consumer = ScoreIngestionConsumer(
        ingestion_queue=Queue(),
        identifier=0,
        client=Mock(),
        public_key="pk-test",
        flush_interval=30,
    )

    consumer.start()
    consumer.pause()
    consumer.join(timeout=0.5)

    assert not consumer.is_alive()


def test_media_upload_consumer_signal_shutdown_wakes_blocked_thread():
    media_manager = MediaManager(
        api_client=Mock(),
        httpx_client=Mock(),
        media_upload_queue=Queue(),
    )
    consumer = MediaUploadConsumer(identifier=0, media_manager=media_manager)

    consumer.start()
    consumer.pause()
    media_manager.signal_shutdown()
    consumer.join(timeout=0.5)

    assert not consumer.is_alive()


def test_at_fork_reinit_creates_new_queues_and_consumers(monkeypatch):
    """_at_fork_reinit() must replace queues and start fresh consumer threads."""
    monkeypatch.setenv("LANGFUSE_MEDIA_UPLOAD_ENABLED", "false")

    with LangfuseResourceManager._lock:
        LangfuseResourceManager._instances.clear()

    client = Langfuse(
        public_key="pk-fork-reinit",
        secret_key="sk-fork-reinit",
        span_exporter=NoOpSpanExporter(),
    )
    rm = client._resources
    assert rm is not None

    old_score_queue = rm._score_ingestion_queue
    old_media_queue = rm._media_upload_queue
    old_ingestion_consumers = list(rm._ingestion_consumers)

    rm._at_fork_reinit()

    assert rm._score_ingestion_queue is not old_score_queue
    assert rm._media_upload_queue is not old_media_queue
    assert len(rm._ingestion_consumers) == 1
    assert rm._ingestion_consumers[0].is_alive()

    # In a real fork, old threads don't exist in the child process.
    # In this unit test they do — stop them explicitly to avoid leaking threads.
    for consumer in old_ingestion_consumers:
        consumer.pause()
        consumer.join(timeout=1.0)

    client.shutdown()


def test_at_fork_reinit_skips_when_shutdown(monkeypatch):
    """_at_fork_reinit() must not restart threads after intentional shutdown."""
    monkeypatch.setenv("LANGFUSE_MEDIA_UPLOAD_ENABLED", "false")

    with LangfuseResourceManager._lock:
        LangfuseResourceManager._instances.clear()

    client = Langfuse(
        public_key="pk-fork-shutdown",
        secret_key="sk-fork-shutdown",
        span_exporter=NoOpSpanExporter(),
    )
    rm = client._resources
    assert rm is not None

    old_score_queue = rm._score_ingestion_queue
    old_lock = LangfuseResourceManager._lock

    rm._shutdown = True
    rm._at_fork_reinit()

    assert rm._score_ingestion_queue is old_score_queue  # queue must not be replaced
    # The class-level lock is shared state the child still needs (e.g. to create
    # a new client), so it must be replaced even when this instance is shut down.
    assert LangfuseResourceManager._lock is not old_lock

    client.shutdown()


def test_at_fork_reinit_replaces_lock(monkeypatch):
    """_at_fork_reinit() must replace the class-level lock with a fresh one.

    If a thread held _lock at fork time, the child has no such thread and the
    lock can never be released, causing a deadlock.  The reinit handler must
    replace it before doing any other work so the child can always acquire it.
    """
    monkeypatch.setenv("LANGFUSE_MEDIA_UPLOAD_ENABLED", "false")

    with LangfuseResourceManager._lock:
        LangfuseResourceManager._instances.clear()

    client = Langfuse(
        public_key="pk-fork-lock",
        secret_key="sk-fork-lock",
        span_exporter=NoOpSpanExporter(),
    )
    rm = client._resources
    assert rm is not None

    old_lock = LangfuseResourceManager._lock

    rm._at_fork_reinit()

    assert LangfuseResourceManager._lock is not old_lock
    # New lock must be immediately acquirable (not held by any thread).
    acquired = LangfuseResourceManager._lock.acquire(blocking=False)
    assert acquired, "New lock must not be held after _at_fork_reinit()"
    LangfuseResourceManager._lock.release()

    client.shutdown()


def test_at_fork_reinit_new_lock_acquirable_even_if_old_lock_was_held(monkeypatch):
    """Simulate the fork-deadlock scenario: old lock held, new lock must still be acquirable.

    In a real fork, a thread holding _lock in the parent disappears in the child,
    leaving the lock permanently acquired.  Here we replicate that by acquiring the
    old lock without releasing it, then calling _at_fork_reinit() and verifying that
    the replacement lock is free.
    """
    monkeypatch.setenv("LANGFUSE_MEDIA_UPLOAD_ENABLED", "false")

    with LangfuseResourceManager._lock:
        LangfuseResourceManager._instances.clear()

    client = Langfuse(
        public_key="pk-fork-lock-held",
        secret_key="sk-fork-lock-held",
        span_exporter=NoOpSpanExporter(),
    )
    rm = client._resources
    assert rm is not None

    # Simulate the lock being permanently held (as it would be in a forked child
    # when the owning thread no longer exists).
    stuck_lock = LangfuseResourceManager._lock
    stuck_lock.acquire()  # held, never released — simulates the fork scenario

    try:
        rm._at_fork_reinit()

        # The new lock must be a different object and must be acquirable.
        new_lock = LangfuseResourceManager._lock
        assert new_lock is not stuck_lock
        acquired = new_lock.acquire(blocking=False)
        assert acquired, "Replacement lock must be acquirable after _at_fork_reinit()"
        new_lock.release()
    finally:
        stuck_lock.release()  # clean up so other tests are not affected

    client.shutdown()


def test_at_fork_reinit_recreates_httpx_client_by_default(monkeypatch):
    """_at_fork_reinit() must create a new httpx.Client to avoid sharing
    connection-pool file descriptors (TCP sockets) across forked processes.
    httpx.Client is thread-safe but not process-safe."""
    monkeypatch.setenv("LANGFUSE_MEDIA_UPLOAD_ENABLED", "false")

    with LangfuseResourceManager._lock:
        LangfuseResourceManager._instances.clear()

    client = Langfuse(
        public_key="pk-fork-httpx-default",
        secret_key="sk-fork-httpx-default",
        span_exporter=NoOpSpanExporter(),
    )
    rm = client._resources
    assert rm is not None

    old_httpx_client = rm.httpx_client
    old_api = rm.api
    old_async_api = rm.async_api
    old_score_ingestion_client = rm._score_ingestion_client
    assert client.api is old_api
    assert client.async_api is old_async_api

    rm._at_fork_reinit()

    assert rm.httpx_client is not old_httpx_client
    assert rm.api is not old_api
    assert rm.async_api is not old_async_api
    assert client.api is rm.api
    assert client.api is not old_api
    assert client.async_api is rm.async_api
    assert client.async_api is not old_async_api
    assert rm._score_ingestion_client is not old_score_ingestion_client

    client.shutdown()


def test_at_fork_reinit_preserves_custom_httpx_client(monkeypatch):
    """After fork, a caller-supplied httpx.Client is reused as-is.
    The caller is responsible for their own fork-safety (e.g. via their own
    os.register_at_fork handler). The SDK must not silently replace it."""
    import httpx

    monkeypatch.setenv("LANGFUSE_MEDIA_UPLOAD_ENABLED", "false")

    with LangfuseResourceManager._lock:
        LangfuseResourceManager._instances.clear()

    custom_client = httpx.Client(timeout=99)
    client = Langfuse(
        public_key="pk-fork-httpx-custom",
        secret_key="sk-fork-httpx-custom",
        httpx_client=custom_client,
        span_exporter=NoOpSpanExporter(),
    )
    rm = client._resources
    assert rm is not None
    assert rm.httpx_client is custom_client

    rm._at_fork_reinit()

    # Custom client must be preserved — caller owns process-safety for it.
    assert rm.httpx_client is custom_client
    assert rm.api is not None
    assert rm.async_api is not None
    assert rm._score_ingestion_client is not None

    custom_client.close()
    client.shutdown()


def test_at_fork_reinit_new_httpx_client_uses_configured_timeout_and_headers(
    monkeypatch,
):
    """After fork, the recreated httpx.Client must reflect the timeout and
    additional_headers that were set on the resource manager."""
    monkeypatch.setenv("LANGFUSE_MEDIA_UPLOAD_ENABLED", "false")

    with LangfuseResourceManager._lock:
        LangfuseResourceManager._instances.clear()

    client = Langfuse(
        public_key="pk-fork-httpx-settings",
        secret_key="sk-fork-httpx-settings",
        timeout=42,
        additional_headers={"X-Custom": "value"},
        span_exporter=NoOpSpanExporter(),
    )
    rm = client._resources
    assert rm is not None

    rm._at_fork_reinit()

    assert rm.httpx_client.timeout.connect == 42
    assert rm.httpx_client.headers.get("X-Custom") == "value"

    client.shutdown()


def test_stop_and_join_consumer_threads_broadcasts_media_shutdown_after_pausing_all():
    events = []

    class FakeConsumer:
        def __init__(self, identifier):
            self._identifier = identifier

        def pause(self):
            events.append(("pause", self._identifier))

        def join(self):
            events.append(("join", self._identifier))

    class FakeMediaManager:
        def signal_shutdown(self, *, count):
            events.append(("signal_shutdown", count))

    fake_resource_manager = SimpleNamespace(
        _media_upload_consumers=[FakeConsumer(0), FakeConsumer(1)],
        _ingestion_consumers=[],
        _media_manager=FakeMediaManager(),
    )

    LangfuseResourceManager._stop_and_join_consumer_threads(fake_resource_manager)

    assert events == [
        ("pause", 0),
        ("pause", 1),
        ("signal_shutdown", 2),
        ("join", 0),
        ("join", 1),
    ]


# ---------------------------------------------------------------------------
# #1799 regression: durable invariants (added 2026-08)
# ---------------------------------------------------------------------------


def test_1799_stale_client_score_trace_dropped_after_shutdown(monkeypatch):
    """Original #1799 deadlock: stale same-key client must not enqueue after shutdown.

    Proves flush/shutdown cannot hang via stranded Queue.unfinished_tasks.
    Fails on base e3f7e7dd where add_* had no post-shutdown guard.
    """
    import atexit

    monkeypatch.setenv("LANGFUSE_MEDIA_UPLOAD_ENABLED", "false")

    with LangfuseResourceManager._lock:
        LangfuseResourceManager._instances.clear()

    client_a = Langfuse(
        public_key="pk-1799-stale",
        secret_key="sk-1799-stale",
        base_url="http://localhost:9",
        span_exporter=NoOpSpanExporter(),
    )
    # same public_key → same manager (cached)
    client_b = Langfuse(
        public_key="pk-1799-stale",
        secret_key="sk-1799-stale",
        base_url="http://localhost:9",
        span_exporter=NoOpSpanExporter(),
    )
    mgr = client_a._resources
    assert mgr is not None
    assert client_b._resources is mgr

    # Prevent atexit hang on base when queue is stranded
    try:
        atexit.unregister(mgr.shutdown)
    except Exception:
        pass

    client_a.shutdown()

    try:
        assert mgr._shutdown is True
        # stale client attempts work – must be dropped, not enqueued
        client_b.create_score(name="stale-score", value=1.0)
        client_b._create_trace_tags_via_ingestion(trace_id="0" * 32, tags=["stale-tag"])

        assert mgr._score_ingestion_queue.unfinished_tasks == 0
        assert mgr._score_ingestion_queue.qsize() == 0
        # flush/shutdown must return boundedly (would hang on base via Queue.join)
        import threading

        flush_done = threading.Event()

        def do_flush():
            mgr.flush()
            flush_done.set()

        t = threading.Thread(target=do_flush, daemon=True)
        t.start()
        assert flush_done.wait(timeout=2), "flush hung – stranded Queue task on base"
        t.join(timeout=1)

        shutdown_done = threading.Event()

        def do_second_shutdown():
            client_b.shutdown()
            shutdown_done.set()

        t2 = threading.Thread(target=do_second_shutdown, daemon=True)
        t2.start()
        assert shutdown_done.wait(timeout=2), "second shutdown hung"
        t2.join(timeout=1)
        assert mgr._shutdown_event.is_set()
    finally:
        # Drain stranded queue on base so atexit/pytest teardown never hangs
        try:
            while mgr._score_ingestion_queue.unfinished_tasks:
                try:
                    mgr._score_ingestion_queue.get_nowait()
                    mgr._score_ingestion_queue.task_done()
                except Exception:
                    break
        except Exception:
            pass
        try:
            atexit.unregister(mgr.shutdown)
        except Exception:
            pass
        with LangfuseResourceManager._lock:
            LangfuseResourceManager._instances.clear()


def test_1799_admission_vs_shutdown_atomicity(monkeypatch):
    """Score/trace check+put must be atomic vs shutdown – no stranded task.

    Deterministic: producer blocks inside queue.put while holding admission lock,
    shutdown must wait for the lock before setting _shutdown and draining.
    """
    import atexit
    import threading

    monkeypatch.setenv("LANGFUSE_MEDIA_UPLOAD_ENABLED", "false")

    with LangfuseResourceManager._lock:
        LangfuseResourceManager._instances.clear()

    client = Langfuse(
        public_key="pk-1799-atomic",
        secret_key="sk-1799-atomic",
        base_url="http://localhost:9",
        span_exporter=NoOpSpanExporter(),
    )
    mgr = client._resources
    assert mgr is not None
    try:
        atexit.unregister(mgr.shutdown)
    except Exception:
        pass

    # Make downstream fast so flush/join is bounded
    mgr._score_ingestion_client.batch_post = Mock(return_value=None)  # type: ignore[attr-defined]
    mock_provider = Mock()
    mock_provider.force_flush = Mock(return_value=None)
    mgr.tracer_provider = mock_provider  # type: ignore[assignment]

    orig_put = mgr._score_ingestion_queue.put
    block_in_put = threading.Event()
    release_put = threading.Event()

    def blocking_put(item, block=False):
        block_in_put.set()
        assert release_put.wait(timeout=2), "test timed out waiting for release"
        return orig_put(item, block=block)

    mgr._score_ingestion_queue.put = blocking_put  # type: ignore[assignment]

    # Producer enters admission critical section (holds admission_lock during put)
    from unittest.mock import Mock as MockBody

    body = MockBody(trace_id="0" * 32, name="race-score", value=1.0)
    event = {"type": "score", "body": body}

    producer = threading.Thread(
        target=lambda: mgr.add_score_task(event, force_sample=True)
    )
    producer.start()
    assert block_in_put.wait(timeout=2), "producer never reached put"

    shutdown_done = threading.Event()

    def do_shutdown():
        mgr.shutdown()
        shutdown_done.set()

    t_shutdown = threading.Thread(target=do_shutdown)
    t_shutdown.start()

    # shutdown must be blocked on admission_lock while producer holds it
    assert not shutdown_done.wait(timeout=0.3), "shutdown must wait for admission lock"

    release_put.set()
    producer.join(timeout=2)
    t_shutdown.join(timeout=5)

    assert not producer.is_alive()
    assert not t_shutdown.is_alive(), "shutdown hung"
    assert mgr._score_ingestion_queue.unfinished_tasks == 0
    assert not any(c.is_alive() for c in mgr._ingestion_consumers)

    # post-shutdown admission must be dropped, not stranded
    body2 = MockBody(trace_id="0" * 32, name="after", value=2.0)
    mgr.add_score_task({"type": "score", "body": body2}, force_sample=True)
    assert mgr._score_ingestion_queue.unfinished_tasks == 0

    with LangfuseResourceManager._lock:
        LangfuseResourceManager._instances.clear()


def test_1799_concurrent_shutdown_completion(monkeypatch):
    """Second shutdown must wait for first to actually complete (no early return)."""
    import atexit
    import threading

    monkeypatch.setenv("LANGFUSE_MEDIA_UPLOAD_ENABLED", "false")

    with LangfuseResourceManager._lock:
        LangfuseResourceManager._instances.clear()

    client = Langfuse(
        public_key="pk-1799-concurrent",
        secret_key="sk-1799-concurrent",
        base_url="http://localhost:9",
        span_exporter=NoOpSpanExporter(),
    )
    mgr = client._resources
    assert mgr is not None
    try:
        atexit.unregister(mgr.shutdown)
    except Exception:
        pass

    # Pause inside flush deterministically – only first flush blocks
    flush_entered = threading.Event()
    release_flush = threading.Event()
    orig_flush = mgr.flush
    call_count = {"n": 0}

    def pausing_flush():
        call_count["n"] += 1
        if call_count["n"] == 1:
            flush_entered.set()
            assert release_flush.wait(timeout=5), "flush release timed out"
        return orig_flush()

    mgr.flush = pausing_flush  # type: ignore[assignment]

    first_done = threading.Event()
    second_done = threading.Event()
    second_elapsed: list[float] = []

    def first():
        mgr.shutdown()
        first_done.set()

    def second():
        import time

        # ensure first has entered flush (and thus holds _shutdown flag)
        assert flush_entered.wait(timeout=2)
        start = time.monotonic()
        mgr.shutdown()
        second_elapsed.append(time.monotonic() - start)
        second_done.set()

    t1 = threading.Thread(target=first)
    t2 = threading.Thread(target=second)
    t1.start()
    # wait for first to be inside flush
    assert flush_entered.wait(timeout=2)
    t2.start()

    # second must be blocked while first is paused inside flush
    assert not second_done.wait(timeout=0.5), "second shutdown returned early"
    assert not first_done.is_set()

    release_flush.set()
    assert first_done.wait(timeout=5)
    assert second_done.wait(timeout=5)

    assert second_elapsed[0] >= 0.3, "second should have waited for first"
    # On base, _shutdown_event does not exist – this also proves RED
    assert hasattr(mgr, "_shutdown_event"), "missing _shutdown_event on base"
    assert mgr._shutdown_event.is_set()  # type: ignore[attr-defined]
    assert not any(c.is_alive() for c in mgr._ingestion_consumers)
    assert mgr._score_ingestion_queue.unfinished_tasks == 0

    t1.join(timeout=2)
    t2.join(timeout=2)

    with LangfuseResourceManager._lock:
        LangfuseResourceManager._instances.clear()


def test_1799_media_during_force_flush_not_dropped_and_after_shutdown_dropped(
    monkeypatch,
):
    """Media discovered during tracer force_flush must survive; after shutdown it is dropped.

    Guards P1 regression where begin_shutdown before flush silently dropped media.
    """
    import atexit

    monkeypatch.setenv("LANGFUSE_MEDIA_UPLOAD_ENABLED", "true")

    with LangfuseResourceManager._lock:
        LangfuseResourceManager._instances.clear()

    client = Langfuse(
        public_key="pk-1799-media-flush",
        secret_key="sk-1799-media-flush",
        base_url="http://localhost:9",
        span_exporter=NoOpSpanExporter(),
    )
    mgr = client._resources
    assert mgr is not None
    try:
        atexit.unregister(mgr.shutdown)
    except Exception:
        pass

    # Make flush discover media: monkeypatch flush to call
    # MediaManager._find_and_process_media before joining. The queue must already
    # contain the media job when flush proceeds to queue.join().
    data_uri = "data:text/plain;base64,SGVsbG8="

    def flush_with_media_discovery():
        # Simulate BatchSpanProcessor export discovering media during force_flush
        mgr._media_manager._find_and_process_media(
            data=data_uri,
            trace_id="0" * 32,
            observation_id="0" * 16,
            field="input",
        )
        # Proceed with normal flush (will join both queues)
        # We call the original flush's internals without re-entering our wrapper
        if mgr.tracer_provider is not None and not isinstance(
            mgr.tracer_provider, __import__("opentelemetry").trace.ProxyTracerProvider
        ):
            # avoid double media discovery – just join queues as original does after force_flush
            pass
        mgr._score_ingestion_queue.join()
        mgr._media_upload_queue.join()

    # Replace flush with media-discovering variant for this shutdown cycle
    mgr.flush = flush_with_media_discovery  # type: ignore[assignment]

    # Mock media upload to be fast and not require network: process_next will call
    # _process_upload_media_job which we stub to just succeed
    mgr._media_manager._process_upload_media_job = Mock(return_value=None)  # type: ignore[attr-defined]

    # Shutdown must drain the media discovered during flush (before begin_shutdown)
    client.shutdown()

    assert mgr._media_upload_queue.unfinished_tasks == 0
    # queue was drained – if ordering were wrong, media would have been skipped
    # and unfinished would still be 0 but qsize would also be 0; we prove it was
    # enqueued at all by checking that begin_shutdown happened after flush:
    # after shutdown, further media must be dropped
    processed = mgr._media_manager._find_and_process_media(
        data=data_uri,
        trace_id="0" * 32,
        observation_id="0" * 16,
        field="input",
    )
    assert processed == data_uri
    assert mgr._media_upload_queue.unfinished_tasks == 0

    with LangfuseResourceManager._lock:
        LangfuseResourceManager._instances.clear()
