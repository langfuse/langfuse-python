"""Test the LangfuseResourceManager and get_client() function."""

from queue import Queue
from types import SimpleNamespace
from unittest.mock import Mock

from langfuse import Langfuse
from langfuse._client.get_client import get_client
from langfuse._client.resource_manager import LangfuseResourceManager
from langfuse._task_manager.media_manager import MediaManager
from langfuse._task_manager.media_upload_consumer import MediaUploadConsumer
from langfuse._task_manager.score_ingestion_consumer import ScoreIngestionConsumer


def test_get_client_preserves_all_settings(monkeypatch):
    """Test that get_client() preserves environment and all client settings."""
    with LangfuseResourceManager._lock:
        LangfuseResourceManager._instances.clear()

    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-comprehensive-default")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-comprehensive-default")
    monkeypatch.setenv("LANGFUSE_BASE_URL", "http://localhost:3000")

    def should_export(span):
        return span.name != "drop"

    settings = {
        "environment": "test-env",
        "release": "v1.2.3",
        "timeout": 30,
        "flush_at": 100,
        "sample_rate": 0.8,
        "should_export_span": should_export,
        "additional_headers": {"X-Custom": "value"},
        "tracing_enabled": False,
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
    assert rm.additional_headers == settings["additional_headers"]

    original_client.shutdown()


def test_get_client_multiple_clients_preserve_different_settings():
    """Test that get_client() preserves different settings for multiple clients."""

    def should_export_a(span):
        return span.name.startswith("a")

    def should_export_b(span):
        return span.name.startswith("b")

    # Settings for client A
    settings_a = {
        "public_key": "pk-comprehensive-a",
        "secret_key": "sk-comprehensive-a",
        "environment": "env-a",
        "release": "release-a",
        "timeout": 10,
        "sample_rate": 0.5,
        "should_export_span": should_export_a,
        "tracing_enabled": False,
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
        "tracing_enabled": False,
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
