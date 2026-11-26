"""@private"""

import importlib

import pytest

from langfuse import Langfuse
from langfuse._utils.parse_error import generate_error_message_fern
from langfuse.api import (
    AccessDeniedError,
    MethodNotAllowedError,
    NotFoundError,
    ServiceUnavailableError,
    UnauthorizedError,
)
from langfuse.api.core import ApiError


def test_generate_error_message_access_denied_error():
    exception = AccessDeniedError(body={})
    expected_message = "Forbidden. Please check your access control settings. Refer to our RBAC docs: https://langfuse.com/docs/rbac for details."
    assert generate_error_message_fern(exception) == expected_message


def test_generate_error_message_method_not_allowed_error():
    exception = MethodNotAllowedError(body={})
    expected_message = "Unexpected error occurred. Please check your request and contact support: https://langfuse.com/support."
    assert generate_error_message_fern(exception) == expected_message


def test_generate_error_message_not_found_error():
    exception = NotFoundError(body={})
    expected_message = "Internal error occurred. This is an unusual occurrence and we are monitoring it closely. For help, please contact support: https://langfuse.com/support."
    assert generate_error_message_fern(exception) == expected_message


def test_generate_error_message_unauthorized_error():
    exception = UnauthorizedError(body={})
    expected_message = "Unauthorized. The public key, the secret key or the host is wrong. Check LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY and LANGFUSE_BASE_URL (https://cloud.langfuse.com, https://us.cloud.langfuse.com, https://jp.cloud.langfuse.com or https://hipaa.cloud.langfuse.com), and run langfuse.auth_check(). Setup guide: https://langfuse.com/docs/observability/sdk/overview"
    assert generate_error_message_fern(exception) == expected_message


def test_generate_error_message_service_unavailable_error():
    exception = ServiceUnavailableError()
    expected_message = "Service unavailable. This is an unusual occurrence and we are monitoring it closely. For help, please contact support: https://langfuse.com/support."
    assert generate_error_message_fern(exception) == expected_message


def test_generate_error_message_generic():
    exception = ApiError(status_code=503)
    expected_message = "Service unavailable. This is an unusual occurrence and we are monitoring it closely. For help, please contact support: https://langfuse.com/support."
    assert generate_error_message_fern(exception) == expected_message


V2_MIGRATION_GUIDE_URL = (
    "https://langfuse.com/docs/observability/sdk/upgrade-path/python-v2-to-v3"
)


@pytest.mark.parametrize(
    ("name", "replacement"),
    [
        ("trace", "start_as_current_observation"),
        ("span", 'as_type="span"'),
        ("generation", 'as_type="generation"'),
    ],
)
def test_removed_v2_client_methods_point_to_migration_guide(
    langfuse_memory_client: Langfuse, name: str, replacement: str
):
    with pytest.raises(AttributeError) as exc_info:
        getattr(langfuse_memory_client, name)

    message = str(exc_info.value)
    assert f"Langfuse.{name}() was removed in Python SDK v3" in message
    assert replacement in message
    assert V2_MIGRATION_GUIDE_URL in message


def test_unknown_client_attribute_raises_default_attribute_error(
    langfuse_memory_client: Langfuse,
):
    with pytest.raises(AttributeError, match="has no attribute 'does_not_exist'"):
        getattr(langfuse_memory_client, "does_not_exist")

    assert not hasattr(langfuse_memory_client, "does_not_exist")


def test_property_attribute_error_is_not_replaced(monkeypatch):
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    client = Langfuse(public_key=None, secret_key=None)

    with pytest.raises(AttributeError, match="Langfuse client is not initialized"):
        client.api


def test_removed_decorators_module_points_to_migration_guide():
    # ModuleNotFoundError keeps `except ModuleNotFoundError` v2/v3 probes working.
    with pytest.raises(ModuleNotFoundError) as exc_info:
        importlib.import_module("langfuse.decorators")

    message = str(exc_info.value)
    assert "from langfuse import observe" in message
    assert V2_MIGRATION_GUIDE_URL in message
