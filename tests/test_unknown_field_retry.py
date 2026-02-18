"""Tests for unknown field retry functionality.

This module tests the automatic retry behavior when the server rejects unknown fields,
enabling backward compatibility between newer SDK versions and older server versions.
"""

import httpx
import pytest
from datetime import datetime

from langfuse import Langfuse
from langfuse._client.http_retry_patch import _remove_fields
from langfuse._client.resource_manager import LangfuseResourceManager
from langfuse.api.core import ApiError


class TestUnknownFieldRetry:
    """Test that SDK handles unknown field errors gracefully."""

    @pytest.fixture(autouse=True)
    def reset_resource_manager(self):
        """Reset LangfuseResourceManager singleton before each test."""
        LangfuseResourceManager.reset()
        yield
        LangfuseResourceManager.reset()

    @pytest.fixture
    def complete_dataset_response(self):
        """Provide a complete dataset response with all required fields."""
        return {
            "id": "ds-1",
            "name": "test",
            "projectId": "proj-1",
            "metadata": {},
            "createdAt": datetime.now().isoformat(),
            "updatedAt": datetime.now().isoformat(),
        }

    def test_datasets_route_retries(self, monkeypatch, complete_dataset_response):
        """Test retry works for datasets route."""
        calls = []

        def mock_request(client_self, method, url, **kwargs):
            calls.append({"method": method, "url": str(url)})

            if "datasets" in str(url) and len(calls) == 1:
                # First call to datasets returns unrecognized_keys error
                return httpx.Response(
                    400,
                    json={
                        "message": "Invalid request data",
                        "error": [{"code": "unrecognized_keys", "keys": ["badField"]}],
                    },
                )
            elif "datasets" in str(url):
                # Retry succeeds
                return httpx.Response(200, json=complete_dataset_response)
            else:
                # Health check
                return httpx.Response(200, json={"status": "OK"})

        # Patch BEFORE creating the client
        monkeypatch.setattr("httpx.Client.request", mock_request)

        langfuse = Langfuse(
            public_key="pk-test", secret_key="sk-test", host="http://localhost:3000"
        )
        dataset = langfuse.create_dataset(name="test")

        assert dataset.id == "ds-1"
        dataset_calls = [c for c in calls if "datasets" in c["url"]]
        assert len(dataset_calls) == 2, "Should retry once on unrecognized_keys error"

    def test_prompts_route_retries(self, monkeypatch):
        """Test retry works for prompts route."""
        calls = []

        def mock_request(client_self, method, url, **kwargs):
            url_str = str(url)
            calls.append({"method": method, "url": url_str})

            # Count how many prompts calls we've had so far (before this one)
            prompts_count = len([c for c in calls[:-1] if "prompts" in c["url"]])

            print(f"Mock request: {url_str}, prompts_count: {prompts_count}")

            if "prompts" in url_str:
                if prompts_count == 0:
                    # First call to prompts returns error
                    print("Returning 400 error for prompts")
                    return httpx.Response(
                        400,
                        json={
                            "message": "Invalid request data",
                            "error": [
                                {"code": "unrecognized_keys", "keys": ["extraField"]}
                            ],
                        },
                    )
                else:
                    # Retry succeeds
                    print("Returning 200 success for prompts")
                    return httpx.Response(
                        200,
                        json={
                            "id": "prompt-1",
                            "name": "test",
                            "version": 1,
                            "type": "text",
                            "prompt": "hello",
                            "config": {},
                            "labels": [],
                            "tags": [],
                        },
                    )
            else:
                # Health check or other calls
                print(f"Returning health check for: {url_str}")
                return httpx.Response(200, json={"status": "OK"})

        monkeypatch.setattr("httpx.Client.request", mock_request)

        langfuse = Langfuse(
            public_key="pk-test", secret_key="sk-test", host="http://localhost:3000"
        )
        print("Created Langfuse client, now calling create_prompt")
        prompt = langfuse.create_prompt(name="test", prompt="hello", type="text")

        assert prompt.name == "test"
        assert prompt.version == 1
        prompt_calls = [c for c in calls if "prompts" in c["url"]]
        print(f"All calls: {calls}")
        assert len(prompt_calls) == 2, "Should retry once on unrecognized_keys error"

    def test_multiple_unrecognized_keys(self, monkeypatch, complete_dataset_response):
        """Test handling multiple keys in one error."""
        calls = []

        def mock_request(client_self, method, url, **kwargs):
            calls.append({"method": method, "url": str(url)})

            if "datasets" in str(url) and len(calls) == 1:
                return httpx.Response(
                    400,
                    json={
                        "message": "Invalid request data",
                        "error": [
                            {
                                "code": "unrecognized_keys",
                                "keys": ["field1", "field2", "field3"],
                            }
                        ],
                    },
                )
            elif "datasets" in str(url):
                return httpx.Response(200, json=complete_dataset_response)
            else:
                return httpx.Response(200, json={"status": "OK"})

        monkeypatch.setattr("httpx.Client.request", mock_request)

        langfuse = Langfuse(
            public_key="pk-test", secret_key="sk-test", host="http://localhost:3000"
        )
        dataset = langfuse.create_dataset(name="test")

        assert dataset.id == "ds-1"

    def test_422_status_code_also_retries(self, monkeypatch, complete_dataset_response):
        """Test 422 validation errors trigger retry."""
        calls = []

        def mock_request(client_self, method, url, **kwargs):
            calls.append({"method": method, "url": str(url)})

            if "datasets" in str(url) and len(calls) == 1:
                return httpx.Response(
                    422,
                    json={
                        "message": "Invalid request data",
                        "error": [{"code": "unrecognized_keys", "keys": ["badField"]}],
                    },
                )
            elif "datasets" in str(url):
                return httpx.Response(200, json=complete_dataset_response)
            else:
                return httpx.Response(200, json={"status": "OK"})

        monkeypatch.setattr("httpx.Client.request", mock_request)

        langfuse = Langfuse(
            public_key="pk-test", secret_key="sk-test", host="http://localhost:3000"
        )
        dataset = langfuse.create_dataset(name="test")

        assert dataset.id == "ds-1"

    def test_other_errors_not_retried(self, monkeypatch):
        """Test non-unrecognized_keys errors don't retry."""
        calls = []

        def mock_request(client_self, method, url, **kwargs):
            url_str = str(url)
            calls.append({"method": method, "url": url_str})
            print(f"Mock request for other_errors test: {url_str}")

            if "datasets" in url_str:
                print("Returning 401 for datasets")
                return httpx.Response(401, json={"message": "Invalid authentication"})
            else:
                print("Returning OK for health check")
                return httpx.Response(200, json={"status": "OK"})

        monkeypatch.setattr("httpx.Client.request", mock_request)

        langfuse = Langfuse(
            public_key="pk-test", secret_key="sk-test", host="http://localhost:3000"
        )
        print("About to call create_dataset expecting error")

        with pytest.raises(ApiError):
            result = langfuse.create_dataset(name="test")
            print(f"create_dataset returned: {result}")

        dataset_calls = [c for c in calls if "datasets" in c["url"]]
        print(f"Dataset calls: {dataset_calls}")
        assert len(dataset_calls) == 1, "Should not retry on auth errors"

    def test_non_json_response_handled_gracefully(self, monkeypatch):
        """Test plain text errors don't crash."""

        def mock_request(client_self, method, url, **kwargs):
            if "datasets" in str(url):
                return httpx.Response(400, text="Internal error")
            else:
                return httpx.Response(200, json={"status": "OK"})

        monkeypatch.setattr("httpx.Client.request", mock_request)

        langfuse = Langfuse(
            public_key="pk-test", secret_key="sk-test", host="http://localhost:3000"
        )

        with pytest.raises(ApiError):
            langfuse.create_dataset(name="test")


def test_remove_fields_helper():
    """Test the field removal utility function."""
    # Top-level removal
    assert _remove_fields({"keep": "yes", "remove": "no"}, {"remove"}) == {
        "keep": "yes"
    }

    # Nested removal
    assert _remove_fields({"a": {"b": 1, "c": 2}}, {"c"}) == {"a": {"b": 1}}

    # List of dicts
    assert _remove_fields({"items": [{"id": 1, "bad": "x"}]}, {"bad"}) == {
        "items": [{"id": 1}]
    }

    # Primitives unchanged
    assert _remove_fields("string", {"field"}) == "string"
    assert _remove_fields(123, {"field"}) == 123

    # Multiple fields
    assert _remove_fields({"a": 1, "b": 2, "c": 3}, {"a", "c"}) == {"b": 2}
