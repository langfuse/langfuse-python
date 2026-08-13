"""Tests for project id resolution behind `get_trace_url`.

The project id lookup is a blocking network request. `get_trace_url` is commonly
called once per row when rendering links, so the lookup must be attempted at most
once per client instance -- on failure as well as on success -- and it must never
raise into the caller.
"""

import threading
from types import SimpleNamespace
from typing import List, Optional
from unittest.mock import Mock

import pytest

from langfuse import Langfuse
from langfuse._client.resource_manager import LangfuseResourceManager

TRACE_ID = "1234567890abcdef1234567890abcdef"


@pytest.fixture
def client(monkeypatch):
    """A client with fake credentials; `api` is replaced per test."""
    with LangfuseResourceManager._lock:
        LangfuseResourceManager._instances.clear()

    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-trace-url")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-trace-url")
    monkeypatch.setenv("LANGFUSE_BASE_URL", "http://localhost:3000")

    langfuse = Langfuse()

    yield langfuse

    with LangfuseResourceManager._lock:
        LangfuseResourceManager._instances.clear()


def _api_returning(project_id):
    api = Mock()
    api.projects.get.return_value = SimpleNamespace(
        data=[SimpleNamespace(id=project_id)] if project_id else []
    )
    return api


def test_project_id_is_fetched_once_on_success(client):
    client.api = _api_returning("project-id")

    first = client.get_trace_url(trace_id=TRACE_ID)
    second = client.get_trace_url(trace_id=TRACE_ID)

    assert first == f"http://localhost:3000/project/project-id/traces/{TRACE_ID}"
    assert second == first
    assert client.api.projects.get.call_count == 1


def test_project_id_is_not_refetched_when_no_project_is_found(client):
    client.api = _api_returning(None)

    assert client.get_trace_url(trace_id=TRACE_ID) is None
    assert client.get_trace_url(trace_id=TRACE_ID) is None
    assert client.api.projects.get.call_count == 1


def test_project_id_lookup_failure_does_not_raise_and_is_not_retried(client):
    api = Mock()
    api.projects.get.side_effect = RuntimeError("401 Unauthorized")
    client.api = api

    assert client.get_trace_url(trace_id=TRACE_ID) is None
    assert client.get_trace_url(trace_id=TRACE_ID) is None
    assert api.projects.get.call_count == 1


def test_concurrent_callers_all_see_the_resolved_project_id(client):
    """A caller must not observe the pre-resolution state of an in-flight lookup.

    The lookup is slow enough here that every other thread arrives while the
    first one is still waiting on the network. Each must block and then see the
    resolved id -- not a `None` that a request in flight is about to fill in.
    """
    started = threading.Event()
    release = threading.Event()

    def slow_get():
        started.set()
        release.wait(timeout=5)
        return SimpleNamespace(data=[SimpleNamespace(id="project-id")])

    api = Mock()
    api.projects.get.side_effect = slow_get
    client.api = api

    results: List[Optional[str]] = []
    results_lock = threading.Lock()

    def call():
        url = client.get_trace_url(trace_id=TRACE_ID)
        with results_lock:
            results.append(url)

    threads = [threading.Thread(target=call) for _ in range(8)]
    for thread in threads:
        thread.start()

    assert started.wait(timeout=5), "the first lookup never started"
    release.set()

    for thread in threads:
        thread.join(timeout=5)
        assert not thread.is_alive()

    expected = f"http://localhost:3000/project/project-id/traces/{TRACE_ID}"
    assert results == [expected] * 8
    assert api.projects.get.call_count == 1


def test_no_project_id_lookup_without_a_trace_id(client):
    client.api = _api_returning("project-id")

    assert client.get_trace_url() is None
    assert client.api.projects.get.call_count == 0
