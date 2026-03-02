from queue import Queue
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest

from langfuse._task_manager.media_manager import MediaManager
from langfuse.media import LangfuseMedia


def _upload_response(status_code: int, text: str = "") -> httpx.Response:
    request = httpx.Request("PUT", "https://example.com/upload")
    return httpx.Response(status_code=status_code, request=request, text=text)


def _upload_job() -> dict:
    return {
        "media_id": "media-id",
        "content_bytes": b"payload",
        "content_type": "image/jpeg",
        "content_length": 7,
        "content_sha256_hash": "sha256hash",
        "trace_id": "trace-id",
        "observation_id": None,
        "field": "input",
    }


def test_media_upload_retries_on_retryable_http_status():
    media_api = Mock()
    media_api.get_upload_url.return_value = SimpleNamespace(
        upload_url="https://example.com/upload",
        media_id="media-id",
    )
    media_api.patch.return_value = None

    httpx_client = Mock()
    httpx_client.put.side_effect = [
        _upload_response(503, "temporary failure"),
        _upload_response(200, "ok"),
    ]

    manager = MediaManager(
        api_client=SimpleNamespace(media=media_api),
        httpx_client=httpx_client,
        media_upload_queue=Queue(),
        max_retries=3,
    )

    manager._process_upload_media_job(data=_upload_job())

    assert httpx_client.put.call_count == 2
    media_api.patch.assert_called_once()
    assert media_api.patch.call_args.kwargs["upload_http_status"] == 200


def test_media_upload_gives_up_on_non_retryable_http_status():
    media_api = Mock()
    media_api.get_upload_url.return_value = SimpleNamespace(
        upload_url="https://example.com/upload",
        media_id="media-id",
    )
    media_api.patch.return_value = None

    httpx_client = Mock()
    httpx_client.put.return_value = _upload_response(403, "forbidden")

    manager = MediaManager(
        api_client=SimpleNamespace(media=media_api),
        httpx_client=httpx_client,
        media_upload_queue=Queue(),
        max_retries=3,
    )

    with pytest.raises(httpx.HTTPStatusError):
        manager._process_upload_media_job(data=_upload_job())

    assert httpx_client.put.call_count == 1
    media_api.patch.assert_called_once()
    assert media_api.patch.call_args.kwargs["upload_http_status"] == 403


def test_find_and_process_media_sse_done_passes_through():
    queue = Queue()
    manager = MediaManager(
        api_client=SimpleNamespace(media=Mock()),
        httpx_client=Mock(),
        media_upload_queue=queue,
    )

    data = "data: [DONE]"
    result = manager._find_and_process_media(
        data=data, trace_id="trace-id", observation_id=None, field="output"
    )

    assert result == data
    assert queue.empty()


def test_find_and_process_media_sse_json_payload_passes_through():
    queue = Queue()
    manager = MediaManager(
        api_client=SimpleNamespace(media=Mock()),
        httpx_client=Mock(),
        media_upload_queue=queue,
    )

    plain_sse = 'data: {"choices": [{"delta": {"content": "hello"}}]}'
    result = manager._find_and_process_media(
        data=plain_sse, trace_id="trace-id", observation_id=None, field="output"
    )
    assert result == plain_sse

    tricky_sse = 'data: {"text": "what is ;base64 encoding?", "count": 1}'
    result = manager._find_and_process_media(
        data=tricky_sse, trace_id="trace-id", observation_id=None, field="output"
    )
    assert result == tricky_sse
    assert queue.empty()


def test_find_and_process_media_valid_base64_uri_is_processed():
    queue = Queue()
    manager = MediaManager(
        api_client=SimpleNamespace(media=Mock()),
        httpx_client=Mock(),
        media_upload_queue=queue,
    )

    valid_base64_data_uri = (
        "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/4QBARXhpZgAA"
    )
    result = manager._find_and_process_media(
        data=valid_base64_data_uri,
        trace_id="trace-id",
        observation_id=None,
        field="input",
    )

    assert isinstance(result, LangfuseMedia)
    assert not queue.empty()


def test_find_and_process_media_data_uri_without_comma_passes_through():
    queue = Queue()
    manager = MediaManager(
        api_client=SimpleNamespace(media=Mock()),
        httpx_client=Mock(),
        media_upload_queue=queue,
    )

    data = "data:image/png;base64"
    result = manager._find_and_process_media(
        data=data, trace_id="trace-id", observation_id=None, field="input"
    )

    assert result == data
    assert queue.empty()


def test_find_and_process_media_non_base64_data_uri_passes_through():
    queue = Queue()
    manager = MediaManager(
        api_client=SimpleNamespace(media=Mock()),
        httpx_client=Mock(),
        media_upload_queue=queue,
    )

    data = "data:text/plain,hello world"
    result = manager._find_and_process_media(
        data=data, trace_id="trace-id", observation_id=None, field="input"
    )

    assert result == data
    assert queue.empty()


def test_find_and_process_media_sse_in_nested_structure_passes_through():
    queue = Queue()
    manager = MediaManager(
        api_client=SimpleNamespace(media=Mock()),
        httpx_client=Mock(),
        media_upload_queue=queue,
    )

    data = [
        {"role": "assistant", "content": "data: [DONE]"},
        {"role": "user", "content": "data: hello"},
    ]
    result = manager._find_and_process_media(
        data=data, trace_id="trace-id", observation_id=None, field="output"
    )

    assert result == data
    assert queue.empty()
