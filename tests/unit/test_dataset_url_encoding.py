import pytest
from unittest.mock import patch, MagicMock
from langfuse import Langfuse
import httpx

def test_dataset_url_encoding_in_requests():
    with patch("httpx.Client.send") as mock_send:
        # Mock response for get
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "dataset-id",
            "name": "my/dataset",
            "description": "test",
            "metadata": {},
            "projectId": "project-id",
            "createdAt": "2026-01-01T00:00:00Z",
            "updatedAt": "2026-01-01T00:00:00Z"
        }
        mock_response.headers = httpx.Headers()

        # Mock response for list
        mock_items_response = MagicMock(spec=httpx.Response)
        mock_items_response.status_code = 200
        mock_items_response.json.return_value = {
            "data": [],
            "meta": {"page": 1, "limit": 50, "totalItems": 0, "totalPages": 1}
        }

        # Mock response for run
        mock_run_response = MagicMock(spec=httpx.Response)
        mock_run_response.status_code = 200
        mock_run_response.json.return_value = {
            "id": "run-id",
            "name": "my/run",
            "datasetName": "my/dataset",
            "datasetId": "dataset-id",
            "createdAt": "2026-01-01T00:00:00Z",
            "updatedAt": "2026-01-01T00:00:00Z",
            "metadata": {},
            "datasetRunItems": []
        }

        # Mock response for runs
        mock_runs_response = MagicMock(spec=httpx.Response)
        mock_runs_response.status_code = 200
        mock_runs_response.json.return_value = {
            "data": [],
            "meta": {"page": 1, "limit": 50, "totalItems": 0, "totalPages": 1}
        }

        # Mock response for delete
        mock_delete_response = MagicMock(spec=httpx.Response)
        mock_delete_response.status_code = 200
        mock_delete_response.json.return_value = {
            "message": "Dataset run deleted successfully"
        }

        def side_effect(request, *args, **kwargs):
            url_str = str(request.url)
            if "dataset-items" in url_str:
                return mock_items_response
            elif "/runs/" in url_str:
                if request.method == "DELETE":
                    return mock_delete_response
                return mock_run_response
            elif "/runs" in url_str:
                return mock_runs_response
            return mock_response

        mock_send.side_effect = side_effect

        langfuse = Langfuse(public_key="pk-test", secret_key="sk-test", base_url="http://localhost:3000")

        # 1. get_dataset
        langfuse.get_dataset("my/dataset")
        langfuse.get_dataset("my dataset")

        # 2. get_dataset_run
        langfuse.get_dataset_run(dataset_name="my/dataset", run_name="my/run")
        langfuse.get_dataset_run(dataset_name="my dataset", run_name="my run")

        # 3. get_dataset_runs
        langfuse.get_dataset_runs(dataset_name="my/dataset")
        langfuse.get_dataset_runs(dataset_name="my dataset")

        # 4. delete_dataset_run
        langfuse.delete_dataset_run(dataset_name="my/dataset", run_name="my/run")
        langfuse.delete_dataset_run(dataset_name="my dataset", run_name="my run")

        # Collect all requested URLs
        requested_urls = [str(call[0][0].url) for call in mock_send.call_args_list]
        for url in requested_urls:
            print(url)
