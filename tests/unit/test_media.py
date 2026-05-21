import base64
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from langfuse.media import LangfuseMedia

# Test data
SAMPLE_JPEG_BYTES = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00H\x00H\x00\x00"
SAMPLE_BASE64_DATA_URI = (
    "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/4QBARXhpZgAA"
)


def test_init_with_base64_data_uri():
    media = LangfuseMedia(base64_data_uri=SAMPLE_BASE64_DATA_URI)
    assert media._source == "base64_data_uri"
    assert media._content_type == "image/jpeg"
    assert media._content_bytes is not None


def test_init_with_content_bytes():
    media = LangfuseMedia(content_bytes=SAMPLE_JPEG_BYTES, content_type="image/jpeg")
    assert media._source == "bytes"
    assert media._content_type == "image/jpeg"
    assert media._content_bytes == SAMPLE_JPEG_BYTES


def test_init_with_invalid_input():
    # LangfuseMedia logs error but doesn't raise ValueError when initialized without required params
    media = LangfuseMedia()
    assert media._source is None
    assert media._content_type is None
    assert media._content_bytes is None

    media = LangfuseMedia(content_bytes=SAMPLE_JPEG_BYTES)  # Missing content_type
    assert media._source is None
    assert media._content_type is None
    assert media._content_bytes is None

    media = LangfuseMedia(content_type="image/jpeg")  # Missing content_bytes
    assert media._source is None
    assert media._content_type is None
    assert media._content_bytes is None


def test_content_length():
    media = LangfuseMedia(content_bytes=SAMPLE_JPEG_BYTES, content_type="image/jpeg")
    assert media._content_length == len(SAMPLE_JPEG_BYTES)


def test_content_sha256_hash():
    media = LangfuseMedia(content_bytes=SAMPLE_JPEG_BYTES, content_type="image/jpeg")
    assert media._content_sha256_hash is not None
    # Hash should be base64 encoded
    assert base64.b64decode(media._content_sha256_hash)


def test_reference_string():
    media = LangfuseMedia(content_bytes=SAMPLE_JPEG_BYTES, content_type="image/jpeg")

    media._media_id = "MwoGlsMS6lW8ijWeRyZKfD"
    reference = media._reference_string
    assert (
        reference
        == "@@@langfuseMedia:type=image/jpeg|id=MwoGlsMS6lW8ijWeRyZKfD|source=bytes@@@"
    )


def test_parse_reference_string():
    valid_ref = "@@@langfuseMedia:type=image/jpeg|id=test-id|source=base64_data_uri@@@"
    result = LangfuseMedia.parse_reference_string(valid_ref)

    assert result["media_id"] == "test-id"
    assert result["content_type"] == "image/jpeg"
    assert result["source"] == "base64_data_uri"


def test_parse_invalid_reference_string():
    with pytest.raises(ValueError):
        LangfuseMedia.parse_reference_string("")

    with pytest.raises(ValueError):
        LangfuseMedia.parse_reference_string("invalid")

    with pytest.raises(ValueError):
        LangfuseMedia.parse_reference_string(
            "@@@langfuseMedia:type=image/jpeg@@@"
        )  # Missing fields


def test_file_handling():
    file_path = "static/puton.jpg"

    media = LangfuseMedia(file_path=file_path, content_type="image/jpeg")
    assert media._source == "file"
    assert media._content_bytes is not None
    assert media._content_type == "image/jpeg"


def test_nonexistent_file():
    media = LangfuseMedia(file_path="nonexistent.jpg")

    assert media._source is None
    assert media._content_bytes is None
    assert media._content_type is None


def test_resolve_media_references_uses_configured_httpx_client():
    reference_string = "@@@langfuseMedia:type=image/jpeg|id=test-id|source=bytes@@@"
    fetch_timeout_seconds = 7

    media_api = Mock()
    media_api.get.return_value = SimpleNamespace(
        url="https://example.com/test.jpg", content_type="image/jpeg"
    )

    response = Mock()
    response.content = b"test-bytes"
    response.raise_for_status.return_value = None

    httpx_client = Mock()
    httpx_client.get.return_value = response

    mock_langfuse_client = SimpleNamespace(
        api=SimpleNamespace(media=media_api),
        _resources=SimpleNamespace(httpx_client=httpx_client),
    )

    resolved = LangfuseMedia.resolve_media_references(
        obj={"image": reference_string},
        langfuse_client=mock_langfuse_client,
        resolve_with="base64_data_uri",
        content_fetch_timeout_seconds=fetch_timeout_seconds,
    )

    assert resolved["image"] == "data:image/jpeg;base64,dGVzdC1ieXRlcw=="
    httpx_client.get.assert_called_once_with(
        "https://example.com/test.jpg", timeout=fetch_timeout_seconds
    )
