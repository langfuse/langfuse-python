"""Simplified tests for additional_headers functionality in Langfuse client.

This module tests that additional headers are properly configured in the HTTP clients.
"""

from typing import Sequence

import httpx
import pytest
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

from langfuse._client.client import Langfuse


class NoOpSpanExporter(SpanExporter):
    """Minimal exporter used to verify custom exporter injection."""

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        pass


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

    def test_media_manager_uses_custom_httpx_client(self):
        """Test that media manager reuses the configured custom httpx client."""
        custom_client = httpx.Client()

        langfuse = Langfuse(
            public_key="test-public-key",
            secret_key="test-secret-key",
            host="https://mock-host.com",
            httpx_client=custom_client,
            tracing_enabled=False,
        )

        assert langfuse._resources is not None
        assert langfuse._resources._media_manager._httpx_client is custom_client

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
            base_url="https://mock-host.com",
            additional_headers=additional_headers,
        )

        # Get the OTLP span exporter to check its headers
        exporter = processor.span_exporter

        # Verify additional headers are in the exporter's headers
        assert exporter._client._headers["x-custom-trace-header"] == "trace-value"
        assert exporter._client._headers["x-override-default"] == "override-value"

        # Verify default headers are still present
        assert "authorization" in exporter._client._headers
        assert "x-langfuse-sdk-name" in exporter._client._headers
        assert "x-langfuse-public-key" in exporter._client._headers

        # Check that our override worked
        assert exporter._client._headers["x-override-default"] == "override-value"

    def test_span_processor_none_additional_headers_works(self):
        """Test that span processor works with None additional headers."""
        from langfuse._client.span_processor import LangfuseSpanProcessor

        # Create span processor without additional headers
        processor = LangfuseSpanProcessor(
            public_key="test-public-key",
            secret_key="test-secret-key",
            base_url="https://mock-host.com",
            additional_headers=None,
        )

        # Get the OTLP span exporter
        exporter = processor.span_exporter

        # Verify default headers are present
        assert "authorization" in exporter._client._headers
        assert "x-langfuse-sdk-name" in exporter._client._headers
        assert "x-langfuse-public-key" in exporter._client._headers
        assert exporter._client._headers["x-langfuse-ingestion-version"] == "4"

    def test_span_processor_additional_headers_override_ingestion_version(self):
        """Test that additional headers can override the default ingestion version."""
        from langfuse._client.span_processor import LangfuseSpanProcessor

        processor = LangfuseSpanProcessor(
            public_key="test-public-key",
            secret_key="test-secret-key",
            base_url="https://mock-host.com",
            additional_headers={"x-langfuse-ingestion-version": "3"},
        )

        exporter = processor.span_exporter

        assert exporter._client._headers["x-langfuse-ingestion-version"] == "3"

    def test_span_processor_mixed_case_override_replaces_ingestion_version(self):
        """Test that a differently cased override replaces the default header."""
        from langfuse._client.span_processor import LangfuseSpanProcessor

        processor = LangfuseSpanProcessor(
            public_key="test-public-key",
            secret_key="test-secret-key",
            base_url="https://mock-host.com",
            additional_headers={"X-Langfuse-Ingestion-Version": "3"},
        )

        ingestion_headers = {
            key: value
            for key, value in processor.span_exporter._client._headers.items()
            if key.lower() == "x-langfuse-ingestion-version"
        }

        assert ingestion_headers == {"x-langfuse-ingestion-version": "3"}

    def test_span_processor_uses_custom_span_exporter_when_provided(self):
        """Test that a custom exporter bypasses the default OTLP exporter construction."""
        from langfuse._client.span_processor import LangfuseSpanProcessor

        custom_exporter = NoOpSpanExporter()

        processor = LangfuseSpanProcessor(
            public_key="test-public-key",
            secret_key="test-secret-key",
            base_url="https://mock-host.com",
            additional_headers={"X-Custom-Trace-Header": "trace-value"},
            span_exporter=custom_exporter,
        )

        assert processor.span_exporter is custom_exporter


def _langfuse_otlp_exporter(langfuse: Langfuse):
    """Return the OTLP exporter behind a client's Langfuse span processor."""
    from langfuse._client.span_processor import LangfuseSpanProcessor

    processor = next(
        p
        for p in langfuse._resources.tracer_provider._active_span_processor._span_processors
        if isinstance(p, LangfuseSpanProcessor)
    )
    exporter = processor.span_exporter

    # Langfuse wraps the OTLP exporter to handle media
    return getattr(exporter, "_exporter", exporter)


class TestProjectIdHeader:
    """The configured project ID is sent in the x-langfuse-project-id header."""

    @pytest.fixture(autouse=True)
    def _clean_state(self, monkeypatch):
        from langfuse._client.resource_manager import LangfuseResourceManager

        monkeypatch.delenv("LANGFUSE_PROJECT_ID", raising=False)
        LangfuseResourceManager.reset()
        yield
        LangfuseResourceManager.reset()

    def test_header_on_otlp_exporter_and_api_clients(self):
        langfuse = Langfuse(
            public_key="pk-lf-org",
            secret_key="sk-lf-org",
            base_url="https://mock-host.com",
            project_id="project-a",
            tracer_provider=TracerProvider(),
        )

        exporter = _langfuse_otlp_exporter(langfuse)
        assert exporter._client._headers["x-langfuse-project-id"] == "project-a"
        assert (
            langfuse.api._client_wrapper.get_headers()["x-langfuse-project-id"]
            == "project-a"
        )
        assert (
            langfuse.async_api._client_wrapper.get_headers()["x-langfuse-project-id"]
            == "project-a"
        )

    def test_no_header_without_project_id(self):
        langfuse = Langfuse(
            public_key="pk-lf-project",
            secret_key="sk-lf-project",
            base_url="https://mock-host.com",
            tracer_provider=TracerProvider(),
        )

        exporter = _langfuse_otlp_exporter(langfuse)
        assert "x-langfuse-project-id" not in exporter._client._headers
        assert "x-langfuse-project-id" not in (
            langfuse.api._client_wrapper.get_headers()
        )

    def test_env_var_sets_project_id(self, monkeypatch):
        monkeypatch.setenv("LANGFUSE_PROJECT_ID", "project-from-env")

        langfuse = Langfuse(
            public_key="pk-lf-org",
            secret_key="sk-lf-org",
            base_url="https://mock-host.com",
            tracing_enabled=False,
        )

        assert langfuse._resources.project_id == "project-from-env"
        assert (
            langfuse.api._client_wrapper.get_headers()["x-langfuse-project-id"]
            == "project-from-env"
        )

    def test_argument_wins_over_env_var(self, monkeypatch):
        monkeypatch.setenv("LANGFUSE_PROJECT_ID", "project-from-env")

        langfuse = Langfuse(
            public_key="pk-lf-org",
            secret_key="sk-lf-org",
            base_url="https://mock-host.com",
            project_id="project-from-arg",
            tracer_provider=TracerProvider(),
        )

        assert langfuse._resources.project_id == "project-from-arg"
        exporter = _langfuse_otlp_exporter(langfuse)
        assert exporter._client._headers["x-langfuse-project-id"] == "project-from-arg"
        assert (
            langfuse.api._client_wrapper.get_headers()["x-langfuse-project-id"]
            == "project-from-arg"
        )

    def test_custom_httpx_client_requests_carry_header(self):
        captured_requests = []

        def handler(request: httpx.Request) -> httpx.Response:
            captured_requests.append(request)
            return httpx.Response(200, json={"data": []})

        custom_client = httpx.Client(
            transport=httpx.MockTransport(handler),
            headers={"X-Existing-Header": "existing-value"},
        )

        langfuse = Langfuse(
            public_key="pk-lf-org",
            secret_key="sk-lf-org",
            base_url="https://mock-host.com",
            project_id="project-a",
            httpx_client=custom_client,
            tracing_enabled=False,
        )
        langfuse.api.projects.get()

        assert len(captured_requests) == 1
        assert captured_requests[0].headers["x-langfuse-project-id"] == "project-a"
        assert captured_requests[0].headers["x-existing-header"] == "existing-value"

    @pytest.mark.parametrize(
        "header_name",
        ["x-langfuse-project-id", "X-Langfuse-Project-Id", "langfuse-project-id"],
    )
    def test_conflicting_additional_header_raises(self, header_name):
        with pytest.raises(ValueError, match="Conflicting project IDs"):
            Langfuse(
                public_key="pk-lf-org",
                secret_key="sk-lf-org",
                base_url="https://mock-host.com",
                project_id="project-a",
                additional_headers={header_name: "project-b"},
                tracing_enabled=False,
            )

    @pytest.mark.parametrize(
        "header_name", ["X-LANGFUSE-PROJECT-ID", "Langfuse-Project-Id"]
    )
    def test_conflicting_httpx_client_header_raises(self, header_name):
        custom_client = httpx.Client(headers={header_name: "project-b"})

        with pytest.raises(ValueError, match="httpx_client.headers"):
            Langfuse(
                public_key="pk-lf-org",
                secret_key="sk-lf-org",
                base_url="https://mock-host.com",
                project_id="project-a",
                httpx_client=custom_client,
                tracing_enabled=False,
            )

    def test_conflicting_header_raises_with_project_id_from_env(self, monkeypatch):
        monkeypatch.setenv("LANGFUSE_PROJECT_ID", "project-a")

        with pytest.raises(ValueError, match="Conflicting project IDs"):
            Langfuse(
                public_key="pk-lf-org",
                secret_key="sk-lf-org",
                base_url="https://mock-host.com",
                additional_headers={"x-langfuse-project-id": "project-b"},
                tracing_enabled=False,
            )

    def test_matching_headers_are_accepted(self):
        langfuse = Langfuse(
            public_key="pk-lf-org",
            secret_key="sk-lf-org",
            base_url="https://mock-host.com",
            project_id="project-a",
            additional_headers={"X-Langfuse-Project-Id": "project-a"},
            httpx_client=httpx.Client(headers={"langfuse-project-id": "project-a"}),
            tracing_enabled=False,
        )

        assert langfuse._resources.project_id == "project-a"

    def test_header_without_project_id_is_left_alone(self):
        langfuse = Langfuse(
            public_key="pk-lf-org",
            secret_key="sk-lf-org",
            base_url="https://mock-host.com",
            additional_headers={"x-langfuse-project-id": "project-b"},
            tracer_provider=TracerProvider(),
        )

        assert langfuse._resources.project_id is None
        assert (
            langfuse._resources.httpx_client.headers["x-langfuse-project-id"]
            == "project-b"
        )
        exporter = _langfuse_otlp_exporter(langfuse)
        assert exporter._client._headers["x-langfuse-project-id"] == "project-b"

    def test_custom_span_exporter_warns_about_project_id(self, caplog):
        with caplog.at_level("WARNING", logger="langfuse"):
            Langfuse(
                public_key="pk-lf-org",
                secret_key="sk-lf-org",
                base_url="https://mock-host.com",
                project_id="project-a",
                tracer_provider=TracerProvider(),
                span_exporter=NoOpSpanExporter(),
            )

        assert (
            sum(
                "project_id is not sent by a custom span_exporter" in r.message
                for r in caplog.records
            )
            == 1
        )
