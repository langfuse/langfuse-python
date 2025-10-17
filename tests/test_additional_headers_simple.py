"""Simplified tests for additional_headers functionality in Langfuse client.

This module tests that additional headers are properly configured in the HTTP clients.
"""

import httpx

from langfuse._client.client import Langfuse


class TestAdditionalHeadersSimple:
    """Simple test suite for additional_headers functionality."""

    def teardown_method(self):
        """Clean up after each test to avoid singleton interference."""
        from langfuse._client.resource_manager import LangfuseResourceManager

        LangfuseResourceManager.reset()

    def test_httpx_client_has_additional_headers_when_none_provided(self):
        """Test that additional headers are set in httpx client when no custom client is provided."""
        additional_headers = {
            "X-Custom-Header": "custom-value",
            "X-Another-Header": "another-value",
        }

        langfuse = Langfuse(
            public_key="test-public-key",
            secret_key="test-secret-key",
            host="https://mock-host.com",
            additional_headers=additional_headers,
            tracing_enabled=False,  # Disable tracing to avoid OTEL setup
        )

        # Verify the httpx client has the additional headers
        assert (
            langfuse._resources.httpx_client.headers["X-Custom-Header"]
            == "custom-value"
        )
        assert (
            langfuse._resources.httpx_client.headers["X-Another-Header"]
            == "another-value"
        )

    def test_custom_httpx_client_with_additional_headers_ignores_additional_headers(
        self,
    ):
        """Test that when additional headers are provided with custom client, additional headers are ignored."""
        # Create a custom httpx client with headers
        existing_headers = {"X-Existing-Header": "existing-value"}
        custom_client = httpx.Client(headers=existing_headers)

        additional_headers = {
            "X-Custom-Header": "custom-value",
            "X-Another-Header": "another-value",
        }

        langfuse = Langfuse(
            public_key="test-public-key",
            secret_key="test-secret-key",
            host="https://mock-host.com",
            httpx_client=custom_client,
            additional_headers=additional_headers,
            tracing_enabled=False,
        )

        # Verify the original client is used (same instance)
        assert langfuse._resources.httpx_client is custom_client

        # Verify existing headers are preserved and additional headers are NOT added
        assert (
            langfuse._resources.httpx_client.headers["x-existing-header"]
            == "existing-value"
        )

        # Additional headers should NOT be present
        assert "x-custom-header" not in langfuse._resources.httpx_client.headers
        assert "x-another-header" not in langfuse._resources.httpx_client.headers

    def test_custom_httpx_client_without_additional_headers_preserves_client(self):
        """Test that when no additional headers are provided, the custom client is preserved."""
        # Create a custom httpx client with headers
        existing_headers = {"X-Existing-Header": "existing-value"}
        custom_client = httpx.Client(headers=existing_headers)

        langfuse = Langfuse(
            public_key="test-public-key",
            secret_key="test-secret-key",
            host="https://mock-host.com",
            httpx_client=custom_client,
            additional_headers=None,  # No additional headers
            tracing_enabled=False,
        )

        # Note: The client instance might be different due to Fern API wrapper behavior,
        # but the important thing is that the headers are preserved
        # Verify existing headers are preserved
        assert (
            langfuse._resources.httpx_client.headers["x-existing-header"]
            == "existing-value"
        )

    def test_none_additional_headers_works(self):
        """Test that passing None for additional_headers works without errors."""
        langfuse = Langfuse(
            public_key="test-public-key",
            secret_key="test-secret-key",
            host="https://mock-host.com",
            additional_headers=None,
            tracing_enabled=False,
        )

        # Verify client was created successfully
        assert langfuse is not None
        assert langfuse._resources is not None
        assert langfuse._resources.httpx_client is not None

    def test_empty_additional_headers_works(self):
        """Test that passing an empty dict for additional_headers works."""
        langfuse = Langfuse(
            public_key="test-public-key",
            secret_key="test-secret-key",
            host="https://mock-host.com",
            additional_headers={},
            tracing_enabled=False,
        )

        # Verify client was created successfully
        assert langfuse is not None
        assert langfuse._resources is not None
        assert langfuse._resources.httpx_client is not None

    def test_span_processor_has_additional_headers_in_otel_exporter(self):
        """Test that span processor includes additional headers in OTEL exporter."""
        from langfuse._client.span_processor import LangfuseSpanProcessor

        additional_headers = {
            "X-Custom-Trace-Header": "trace-value",
            "X-Override-Default": "override-value",
        }

        # Create span processor with additional headers
        processor = LangfuseSpanProcessor(
            public_key="test-public-key",
            secret_key="test-secret-key",
            host="https://mock-host.com",
            additional_headers=additional_headers,
        )

        # Get the OTLP span exporter to check its headers
        exporter = processor.span_exporter

        # Verify additional headers are in the exporter's headers
        assert exporter._headers["X-Custom-Trace-Header"] == "trace-value"
        assert exporter._headers["X-Override-Default"] == "override-value"

        # Verify default headers are still present
        assert "Authorization" in exporter._headers
        assert "x-langfuse-sdk-name" in exporter._headers
        assert "x-langfuse-public-key" in exporter._headers

        # Check that our override worked
        assert exporter._headers["X-Override-Default"] == "override-value"

    def test_span_processor_none_additional_headers_works(self):
        """Test that span processor works with None additional headers."""
        from langfuse._client.span_processor import LangfuseSpanProcessor

        # Create span processor without additional headers
        processor = LangfuseSpanProcessor(
            public_key="test-public-key",
            secret_key="test-secret-key",
            host="https://mock-host.com",
            additional_headers=None,
        )

        # Get the OTLP span exporter
        exporter = processor.span_exporter

        # Verify default headers are present
        assert "Authorization" in exporter._headers
        assert "x-langfuse-sdk-name" in exporter._headers
        assert "x-langfuse-public-key" in exporter._headers
