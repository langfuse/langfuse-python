"""Test suite for Langfuse client initialization with LANGFUSE_HOST and LANGFUSE_BASE_URL.

This test suite verifies that both LANGFUSE_HOST (deprecated) and LANGFUSE_BASE_URL
environment variables work correctly for initializing the Langfuse client.
"""

import os

import pytest
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

import langfuse._client.span_processor as span_processor_module
from langfuse import Langfuse
from langfuse._client.resource_manager import LangfuseResourceManager


class NoOpSpanExporter(SpanExporter):
    def export(self, spans):
        return SpanExportResult.SUCCESS

    def shutdown(self):
        pass


class TestClientInitialization:
    """Tests for Langfuse client initialization with different URL configurations."""

    @pytest.fixture(autouse=True)
    def cleanup_env_vars(self):
        """Fixture to clean up environment variables and singleton cache before and after each test."""
        # Store original values
        original_values = {
            "LANGFUSE_BASE_URL": os.environ.get("LANGFUSE_BASE_URL"),
            "LANGFUSE_HOST": os.environ.get("LANGFUSE_HOST"),
            "LANGFUSE_PUBLIC_KEY": os.environ.get("LANGFUSE_PUBLIC_KEY"),
            "LANGFUSE_SECRET_KEY": os.environ.get("LANGFUSE_SECRET_KEY"),
        }

        # Remove LANGFUSE_BASE_URL and LANGFUSE_HOST for the test
        # but keep PUBLIC_KEY and SECRET_KEY if they exist
        for key in ["LANGFUSE_BASE_URL", "LANGFUSE_HOST"]:
            if key in os.environ:
                del os.environ[key]

        yield

        # Clear the singleton cache to prevent test pollution
        with LangfuseResourceManager._lock:
            LangfuseResourceManager._instances.clear()

        # Restore original values - always remove any test values first
        for key in [
            "LANGFUSE_BASE_URL",
            "LANGFUSE_HOST",
            "LANGFUSE_PUBLIC_KEY",
            "LANGFUSE_SECRET_KEY",
        ]:
            if key in os.environ:
                del os.environ[key]

        # Then restore original values
        for key, value in original_values.items():
            if value is not None:
                os.environ[key] = value

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

    def test_env_host_fallback_logs_no_warning(self, cleanup_env_vars, caplog):
        """Test that LANGFUSE_HOST keeps working without a deprecation warning."""
        caplog.set_level("WARNING", logger="langfuse")
        os.environ["LANGFUSE_HOST"] = "http://env-host.com"

        client = Langfuse(public_key="test_pk", secret_key="test_sk")

        assert client._base_url == "http://env-host.com"
        assert not any("deprecated" in record.message for record in caplog.records)

    @pytest.mark.parametrize(
        ("missing_key", "provided_kwargs"),
        [
            ("public_key", {"secret_key": "test_sk"}),
            ("secret_key", {"public_key": "test_pk"}),
        ],
    )
    def test_missing_credentials_warning_mentions_local_option(
        self, cleanup_env_vars, monkeypatch, caplog, missing_key, provided_kwargs
    ):
        """Test that the missing-credentials warning points to the local option."""
        monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
        monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
        caplog.set_level("WARNING", logger="langfuse")

        Langfuse(**provided_kwargs)

        warnings = [
            record.getMessage()
            for record in caplog.records
            if f"initialized without {missing_key}" in record.getMessage()
        ]
        assert len(warnings) == 1
        assert "Client will be disabled" in warnings[0]
        assert "langfuse-lite" in warnings[0]
        assert "LANGFUSE_BASE_URL=http://localhost:3000" in warnings[0]

    def test_default_base_url(self, cleanup_env_vars):
        """Test that default base_url is used when nothing is set."""
        client = Langfuse(
            public_key="test_pk",
            secret_key="test_sk",
        )

        assert client._base_url == "https://cloud.langfuse.com"

    def test_zero_sample_rate_parameter_is_preserved(
        self, cleanup_env_vars, monkeypatch
    ):
        """Test that an explicit zero sample rate overrides the environment."""
        monkeypatch.setenv("LANGFUSE_SAMPLE_RATE", "1.0")

        client = Langfuse(
            public_key="test_pk_zero_sample_rate",
            secret_key="test_sk",
            tracing_enabled=False,
            sample_rate=0.0,
        )

        assert client._resources is not None
        assert client._resources.sample_rate == 0.0

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

    @pytest.mark.parametrize(
        "source", ["base_url", "LANGFUSE_BASE_URL", "host", "LANGFUSE_HOST"]
    )
    def test_trailing_slash_is_stripped(self, cleanup_env_vars, monkeypatch, source):
        """Test that trailing slashes are stripped from every base_url source."""
        exporter_endpoints = []

        class RecordingExporter(NoOpSpanExporter):
            def __init__(self, *, endpoint, **kwargs):
                exporter_endpoints.append(endpoint)

        monkeypatch.setattr(
            span_processor_module, "OTLPSpanExporter", RecordingExporter
        )

        kwargs = {}
        if source in ("base_url", "host"):
            kwargs[source] = "http://localhost:3000//"
        else:
            os.environ[source] = "http://localhost:3000//"

        client = Langfuse(
            public_key=f"test_pk_trailing_slash_{source}",
            secret_key="test_sk",
            **kwargs,
        )

        assert client._base_url == "http://localhost:3000"
        assert client.api._client_wrapper.get_base_url() == "http://localhost:3000"
        assert (
            client.async_api._client_wrapper.get_base_url() == "http://localhost:3000"
        )
        assert exporter_endpoints == ["http://localhost:3000/api/public/otel/v1/traces"]

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
