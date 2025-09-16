"""Test environment setting on Langfuse get_client() function."""

import pytest
from langfuse import Langfuse
from langfuse._client.get_client import get_client


def test_get_client_preserves_environment():
    """Test that get_client() preserves the environment when returning existing clients."""
    test_env = "production-test-env"
    original_client = Langfuse(environment=test_env)

    # Verify the original client has the correct environment
    assert original_client._environment == test_env, (
        f"original client environment should be '{test_env}', got '{original_client._environment}'"
    )

    # call get_client() should return a client with the same environment
    retrieved_client = get_client()

    assert retrieved_client._environment == test_env, (
        f"get_client() should return client with environment '{test_env}', got '{retrieved_client._environment}'"
    )

    original_client.shutdown()


def test_get_client_with_multiple_environments():
    """Test get_client() behavior with multiple clients having different environments."""
    env_a = "environment-a"
    env_b = "environment-b"

    client_a = Langfuse(public_key="pk-a", secret_key="sk-a", environment=env_a)
    client_b = Langfuse(public_key="pk-b", secret_key="sk-b", environment=env_b)

    # original clients should have correct environments
    assert client_a._environment == env_a
    assert client_b._environment == env_b

    # Get clients using get_client() with specific public keys
    retrieved_a = get_client(public_key="pk-a")
    retrieved_b = get_client(public_key="pk-b")

    # should have the same environments as the originals
    assert retrieved_a._environment == env_a, (
        f"Expected client A environment to be '{env_a}', got '{retrieved_a._environment}'"
    )
    assert retrieved_b._environment == env_b, (
        f"Expected client B environment to be '{env_b}', got '{retrieved_b._environment}'"
    )

    client_a.shutdown()
    client_b.shutdown()


def test_get_client_single_client_environment():
    """Test that get_client() preserves environment in single-client scenario."""
    # Clean state - ensure no existing clients
    from langfuse._client.resource_manager import LangfuseResourceManager

    with LangfuseResourceManager._lock:
        LangfuseResourceManager._instances.clear()

    test_env = "single-client-env"

    client = Langfuse(environment=test_env)
    assert client._environment == test_env

    # get_client() should return a client with the same environment
    retrieved = get_client()

    # This assertion demonstrates the bug
    assert retrieved._environment == test_env, (
        f"Expected single client scenario to preserve environment '{test_env}', got '{retrieved._environment}'"
    )

    client.shutdown()
