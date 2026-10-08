"""Tests for Langfuse.auth_check() error behavior.

See https://github.com/langfuse/langfuse/issues/906 - auth_check() used to
raise a bare `Exception` when no project was found for the provided
credentials, which is easy to swallow with a broad `except Exception` at the
call site.
"""

from unittest.mock import Mock

import pytest

from langfuse import Langfuse, LangfuseAuthCheckError
from langfuse._client.resource_manager import LangfuseResourceManager


@pytest.fixture
def langfuse():
    langfuse_instance = Langfuse()

    if langfuse_instance._resources is None:
        langfuse_instance._resources = Mock(spec=LangfuseResourceManager)

    langfuse_instance.api = Mock()

    return langfuse_instance


def test_auth_check_raises_langfuse_auth_check_error_when_no_projects(langfuse):
    langfuse.api.projects.get.return_value = Mock(data=[])

    with pytest.raises(LangfuseAuthCheckError):
        langfuse.auth_check()


def test_auth_check_returns_true_when_projects_found(langfuse):
    langfuse.api.projects.get.return_value = Mock(data=[Mock()])

    assert langfuse.auth_check() is True
