"""Test the LangfuseResourceManager and get_client() function."""

from langfuse import Langfuse
from langfuse._client.get_client import get_client
from langfuse._client.resource_manager import LangfuseResourceManager


def test_get_client_preserves_all_settings():
    """Test that get_client() preserves environment and all client settings."""
    with LangfuseResourceManager._lock:
        LangfuseResourceManager._instances.clear()

    settings = {
        "environment": "test-env",
        "release": "v1.2.3",
        "timeout": 30,
        "flush_at": 100,
        "sample_rate": 0.8,
        "additional_headers": {"X-Custom": "value"},
    }

    original_client = Langfuse(**settings)
    retrieved_client = get_client()

    assert retrieved_client._environment == settings["environment"]

    assert retrieved_client._resources is not None
    rm = retrieved_client._resources
    assert rm.environment == settings["environment"]
    assert rm.timeout == settings["timeout"]
    assert rm.sample_rate == settings["sample_rate"]
    assert rm.additional_headers == settings["additional_headers"]

    original_client.shutdown()


def test_get_client_multiple_clients_preserve_different_settings():
    """Test that get_client() preserves different settings for multiple clients."""
    # Settings for client A
    settings_a = {
        "public_key": "pk-comprehensive-a",
        "secret_key": "sk-comprehensive-a",
        "environment": "env-a",
        "release": "release-a",
        "timeout": 10,
        "sample_rate": 0.5,
    }

    # Settings for client B
    settings_b = {
        "public_key": "pk-comprehensive-b",
        "secret_key": "sk-comprehensive-b",
        "environment": "env-b",
        "release": "release-b",
        "timeout": 20,
        "sample_rate": 0.9,
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

    client_a.shutdown()
    client_b.shutdown()
