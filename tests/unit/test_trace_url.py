"""Tests for project id resolution behind `get_trace_url`.

The project id lookup is a blocking network request. `get_trace_url` is commonly
called once per row when rendering links, so the lookup must be attempted at most
once per client instance -- on failure as well as on success -- and it must never
raise into the caller.
"""

from types import SimpleNamespace
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


def test_no_project_id_lookup_without_a_trace_id(client):
    client.api = _api_returning("project-id")

    assert client.get_trace_url() is None
    assert client.api.projects.get.call_count == 0
