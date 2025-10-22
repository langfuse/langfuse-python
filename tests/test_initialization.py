"""Test suite for Langfuse client initialization with LANGFUSE_HOST and LANGFUSE_BASE_URL.

This test suite verifies that both LANGFUSE_HOST (deprecated) and LANGFUSE_BASE_URL
environment variables work correctly for initializing the Langfuse client.
"""

import os

import pytest

from langfuse import Langfuse


class TestClientInitialization:
    """Tests for Langfuse client initialization with different URL configurations."""

    @pytest.fixture
    def cleanup_env_vars(self):
        """Fixture to clean up environment variables before and after each test."""
        # Store original values
        original_base_url = os.environ.get("LANGFUSE_BASE_URL")
        original_host = os.environ.get("LANGFUSE_HOST")
        original_public_key = os.environ.get("LANGFUSE_PUBLIC_KEY")
        original_secret_key = os.environ.get("LANGFUSE_SECRET_KEY")

        # Remove them for the test
        for key in ["LANGFUSE_BASE_URL", "LANGFUSE_HOST"]:
            if key in os.environ:
                del os.environ[key]

        yield

        # Restore original values
        for key, value in [
            ("LANGFUSE_BASE_URL", original_base_url),
            ("LANGFUSE_HOST", original_host),
            ("LANGFUSE_PUBLIC_KEY", original_public_key),
            ("LANGFUSE_SECRET_KEY", original_secret_key),
        ]:
            if value is not None:
                os.environ[key] = value
            elif key in os.environ:
                del os.environ[key]

    def test_base_url_parameter_takes_precedence(self, cleanup_env_vars):
        """Test that base_url parameter takes highest precedence."""
        os.environ["LANGFUSE_BASE_URL"] = "http://env-base-url.com"
        os.environ["LANGFUSE_HOST"] = "http://env-host.com"

        client = Langfuse(
            base_url="http://param-base-url.com",
            host="http://param-host.com",
            public_key="test_pk",
            secret_key="test_sk",
        )

        assert client._base_url == "http://param-base-url.com"

    def test_env_base_url_takes_precedence_over_host_param(self, cleanup_env_vars):
        """Test that LANGFUSE_BASE_URL env var takes precedence over host parameter."""
        os.environ["LANGFUSE_BASE_URL"] = "http://env-base-url.com"

        client = Langfuse(
            host="http://param-host.com",
            public_key="test_pk",
            secret_key="test_sk",
        )

        assert client._base_url == "http://env-base-url.com"

    def test_host_parameter_fallback(self, cleanup_env_vars):
        """Test that host parameter works as fallback when base_url is not set."""
        client = Langfuse(
            host="http://param-host.com",
            public_key="test_pk",
            secret_key="test_sk",
        )

        assert client._base_url == "http://param-host.com"

    def test_env_host_fallback(self, cleanup_env_vars):
        """Test that LANGFUSE_HOST env var works as fallback."""
        os.environ["LANGFUSE_HOST"] = "http://env-host.com"

        client = Langfuse(
            public_key="test_pk",
            secret_key="test_sk",
        )

        assert client._base_url == "http://env-host.com"

    def test_default_base_url(self, cleanup_env_vars):
        """Test that default base_url is used when nothing is set."""
        client = Langfuse(
            public_key="test_pk",
            secret_key="test_sk",
        )

        assert client._base_url == "https://cloud.langfuse.com"

    def test_base_url_env_var(self, cleanup_env_vars):
        """Test that LANGFUSE_BASE_URL environment variable is used correctly."""
        os.environ["LANGFUSE_BASE_URL"] = "http://test-base-url.com"

        client = Langfuse(
            public_key="test_pk",
            secret_key="test_sk",
        )

        assert client._base_url == "http://test-base-url.com"

    def test_host_env_var(self, cleanup_env_vars):
        """Test that LANGFUSE_HOST environment variable is used correctly (deprecated)."""
        os.environ["LANGFUSE_HOST"] = "http://test-host.com"

        client = Langfuse(
            public_key="test_pk",
            secret_key="test_sk",
        )

        assert client._base_url == "http://test-host.com"

    def test_base_url_parameter(self, cleanup_env_vars):
        """Test that base_url parameter is used correctly."""
        client = Langfuse(
            base_url="http://param-base-url.com",
            public_key="test_pk",
            secret_key="test_sk",
        )

        assert client._base_url == "http://param-base-url.com"

    def test_precedence_order_all_set(self, cleanup_env_vars):
        """Test complete precedence order: base_url param > env > host param > env > default."""
        os.environ["LANGFUSE_BASE_URL"] = "http://env-base-url.com"
        os.environ["LANGFUSE_HOST"] = "http://env-host.com"

        # Case 1: base_url parameter wins
        client1 = Langfuse(
            base_url="http://param-base-url.com",
            host="http://param-host.com",
            public_key="test_pk",
            secret_key="test_sk",
        )
        assert client1._base_url == "http://param-base-url.com"

        # Case 2: LANGFUSE_BASE_URL env var wins when base_url param not set
        client2 = Langfuse(
            host="http://param-host.com",
            public_key="test_pk",
            secret_key="test_sk",
        )
        assert client2._base_url == "http://env-base-url.com"

    def test_precedence_without_base_url(self, cleanup_env_vars):
        """Test precedence when base_url options are not set."""
        os.environ["LANGFUSE_HOST"] = "http://env-host.com"

        # Case 1: host parameter wins
        client1 = Langfuse(
            host="http://param-host.com",
            public_key="test_pk",
            secret_key="test_sk",
        )
        assert client1._base_url == "http://param-host.com"

        # Case 2: LANGFUSE_HOST env var is used
        client2 = Langfuse(
            public_key="test_pk",
            secret_key="test_sk",
        )
        assert client2._base_url == "http://env-host.com"

    def test_url_used_in_api_client(self, cleanup_env_vars):
        """Test that the resolved base_url is correctly passed to API clients."""
        test_url = "http://test-unique-api.com"
        # Use a unique public key to avoid singleton conflicts
        client = Langfuse(
            base_url=test_url,
            public_key=f"test_pk_{test_url}",
            secret_key="test_sk",
        )

        # Check that the API client has the correct base_url
        assert client.api._client_wrapper._base_url == test_url
        assert client.async_api._client_wrapper._base_url == test_url

    def test_url_used_in_trace_url_generation(self, cleanup_env_vars):
        """Test that the resolved base_url is stored correctly for trace URL generation."""
        test_url = "http://test-trace-api.com"
        # Use a unique public key to avoid singleton conflicts
        client = Langfuse(
            base_url=test_url,
            public_key=f"test_pk_{test_url}",
            secret_key="test_sk",
        )

        # Verify that the base_url is stored correctly and will be used for URL generation
        # We can't test the full URL generation without making network calls to get project_id
        # but we can verify the base_url is correctly set
        assert client._base_url == test_url

    def test_both_base_url_and_host_params(self, cleanup_env_vars):
        """Test that base_url parameter takes precedence over host parameter."""
        client = Langfuse(
            base_url="http://base-url.com",
            host="http://host.com",
            public_key="test_pk",
            secret_key="test_sk",
        )

        assert client._base_url == "http://base-url.com"

    def test_both_env_vars_set(self, cleanup_env_vars):
        """Test that LANGFUSE_BASE_URL takes precedence over LANGFUSE_HOST."""
        os.environ["LANGFUSE_BASE_URL"] = "http://base-url.com"
        os.environ["LANGFUSE_HOST"] = "http://host.com"

        client = Langfuse(
            public_key="test_pk",
            secret_key="test_sk",
        )

        assert client._base_url == "http://base-url.com"

    def test_localhost_urls(self, cleanup_env_vars):
        """Test that localhost URLs work correctly."""
        # Test with base_url
        client1 = Langfuse(
            base_url="http://localhost:3000",
            public_key="test_pk",
            secret_key="test_sk",
        )
        assert client1._base_url == "http://localhost:3000"

        # Test with host (deprecated)
        client2 = Langfuse(
            host="http://localhost:3000",
            public_key="test_pk",
            secret_key="test_sk",
        )
        assert client2._base_url == "http://localhost:3000"

        # Test with env var
        os.environ["LANGFUSE_BASE_URL"] = "http://localhost:3000"
        client3 = Langfuse(
            public_key="test_pk",
            secret_key="test_sk",
        )
        assert client3._base_url == "http://localhost:3000"

    def test_trailing_slash_handling(self, cleanup_env_vars):
        """Test that URLs with trailing slashes are handled correctly."""
        # URLs with trailing slashes should work
        client1 = Langfuse(
            base_url="http://test.com/",
            public_key="test_pk",
            secret_key="test_sk",
        )
        # The SDK should accept the URL as-is (API client will handle normalization)
        assert client1._base_url == "http://test.com/"

    def test_urls_with_paths(self, cleanup_env_vars):
        """Test that URLs with paths work correctly."""
        client = Langfuse(
            base_url="http://test.com/api/v1",
            public_key="test_pk",
            secret_key="test_sk",
        )
        assert client._base_url == "http://test.com/api/v1"

    def test_https_and_http_urls(self, cleanup_env_vars):
        """Test that both HTTPS and HTTP URLs work."""
        # HTTPS
        client1 = Langfuse(
            base_url="https://secure.com",
            public_key="test_pk",
            secret_key="test_sk",
        )
        assert client1._base_url == "https://secure.com"

        # HTTP
        client2 = Langfuse(
            base_url="http://insecure.com",
            public_key="test_pk",
            secret_key="test_sk",
        )
        assert client2._base_url == "http://insecure.com"
