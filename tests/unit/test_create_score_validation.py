"""Unit tests for create_score input validation.

Ensures that programmer errors (None name or value) raise ValueError
at the call site rather than being silently swallowed.
"""

import pytest

from langfuse import Langfuse
from langfuse._client.resource_manager import LangfuseResourceManager


@pytest.fixture(autouse=True)
def _clear_singleton():
    yield
    with LangfuseResourceManager._lock:
        LangfuseResourceManager._instances.clear()


@pytest.fixture()
def lf():
    return Langfuse(
        public_key="pk-lf-test",
        secret_key="sk-lf-test",
        host="http://localhost:19999",
    )


def test_create_score_raises_on_none_value(lf):
    with pytest.raises(ValueError, match="Invalid score parameters"):
        lf.create_score(name="accuracy", value=None, trace_id="fake-trace-id")


def test_create_score_raises_on_none_name(lf):
    with pytest.raises(ValueError, match="Invalid score parameters"):
        lf.create_score(name=None, value=0.9, trace_id="fake-trace-id")
