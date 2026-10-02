"""Tests for the Langfuse SDK exception hierarchy."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from langfuse import (
    APIError,
    APIErrors,
    AuthError,
    Langfuse,
    LangfuseError,
    RegressionError,
)


class TestExceptionHierarchy:
    def test_all_sdk_errors_derive_from_langfuse_error(self):
        for error_cls in (AuthError, APIError, APIErrors, RegressionError):
            assert issubclass(error_cls, LangfuseError)
            assert issubclass(error_cls, Exception)

    def test_api_error_str_is_unchanged(self):
        error = APIError(401, "Unauthorized", {"code": "invalid_api_key"})

        assert str(error) == "Unauthorized (401): {'code': 'invalid_api_key'}"

    def test_api_errors_str_is_unchanged(self):
        error = APIErrors([APIError(400, "Bad request"), APIError(429, "Rate limited")])

        assert (
            str(error) == "[Langfuse] Bad request (400): None, Rate limited (429): None"
        )


class TestAuthCheck:
    def test_auth_check_raises_auth_error_when_no_projects(self):
        client = Langfuse(public_key="test_pk", secret_key="test_sk")

        with (
            patch.object(
                client.api.projects,
                "get",
                return_value=SimpleNamespace(data=[]),
            ),
            pytest.raises(AuthError, match="no project found"),
        ):
            client.auth_check()

    def test_auth_check_error_is_catchable_as_langfuse_error(self):
        client = Langfuse(public_key="test_pk", secret_key="test_sk")

        with (
            patch.object(
                client.api.projects,
                "get",
                return_value=SimpleNamespace(data=[]),
            ),
            pytest.raises(LangfuseError),
        ):
            client.auth_check()

    def test_auth_check_returns_true_when_projects_exist(self):
        client = Langfuse(public_key="test_pk", secret_key="test_sk")

        with patch.object(
            client.api.projects,
            "get",
            return_value=SimpleNamespace(data=[SimpleNamespace(id="p1")]),
        ):
            assert client.auth_check() is True

    def test_auth_check_maps_fern_unauthorized_error_to_auth_error(self):
        from langfuse.api import UnauthorizedError

        client = Langfuse(public_key="test_pk", secret_key="test_sk")
        fern_error = UnauthorizedError(body={"error": "Unauthorized"})

        with (
            patch.object(client.api.projects, "get", side_effect=fern_error),
            pytest.raises(AuthError, match="invalid credentials") as excinfo,
        ):
            client.auth_check()

        assert isinstance(excinfo.value.__cause__, UnauthorizedError)

    def test_auth_check_propagates_non_auth_fern_errors_unchanged(self):
        from langfuse.api import NotFoundError

        client = Langfuse(public_key="test_pk", secret_key="test_sk")
        fern_error = NotFoundError(body={"error": "Not found"})

        with (
            patch.object(client.api.projects, "get", side_effect=fern_error),
            pytest.raises(NotFoundError),
        ):
            client.auth_check()
