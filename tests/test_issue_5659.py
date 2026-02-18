"""Test for issue #5659: _parse_base64_data_uri misidentifies SSE data as base64 media."""

import base64
import logging

from langfuse.media import LangfuseMedia


def _make_media():
    """Create a LangfuseMedia instance for testing _parse_base64_data_uri."""
    return LangfuseMedia(
        content_bytes=b"dummy", content_type="application/octet-stream"
    )


def test_sse_data_is_not_parsed_as_base64():
    """Verify SSE data strings return (None, None) and are not decoded as media."""
    media = _make_media()
    result = media._parse_base64_data_uri("data: {'foo': 'bar'}")

    assert result == (None, None)


def test_sse_data_with_json():
    """Verify SSE data with JSON payload returns (None, None) and is not decoded as media."""
    media = _make_media()
    result = media._parse_base64_data_uri('data: {"event": "message", "data": "hello"}')

    assert result == (None, None)


def test_valid_base64_data_uri_still_works():
    """Verify a proper base64 data URI is parsed correctly."""
    original_bytes = b"hello world"
    encoded = base64.b64encode(original_bytes).decode("utf-8")
    data_uri = f"data:text/plain;base64,{encoded}"

    media = _make_media()
    content_bytes, content_type = media._parse_base64_data_uri(data_uri)

    assert content_bytes == original_bytes
    assert content_type == "text/plain"


def test_data_uri_without_base64_returns_none():
    """Verify a data URI without ;base64 encoding returns (None, None)."""
    media = _make_media()
    result = media._parse_base64_data_uri("data:text/plain,hello")

    assert result == (None, None)


def test_empty_string_returns_none(caplog):
    """Verify an empty string returns (None, None) without error logging."""
    media = _make_media()
    with caplog.at_level(logging.ERROR, logger="langfuse.media"):
        result = media._parse_base64_data_uri("")

    assert result == (None, None)
    assert caplog.records == []


def test_non_data_uri_returns_none():
    """Verify a regular string returns (None, None)."""
    media = _make_media()
    result = media._parse_base64_data_uri("just a regular string")

    assert result == (None, None)


def test_valid_image_data_uri():
    """Verify a valid image data URI parses correctly."""
    pixel_bytes = b"\x89PNG\r\n"
    encoded = base64.b64encode(pixel_bytes).decode("utf-8")
    data_uri = f"data:image/png;base64,{encoded}"

    media = _make_media()
    content_bytes, content_type = media._parse_base64_data_uri(data_uri)

    assert content_bytes == pixel_bytes
    assert content_type == "image/png"


def test_data_uri_with_mime_params():
    """Verify a data URI with extra MIME parameters (e.g. charset) parses correctly."""
    original_bytes = b"hello world"
    encoded = base64.b64encode(original_bytes).decode("utf-8")
    data_uri = f"data:text/plain;charset=utf-8;base64,{encoded}"

    media = _make_media()
    content_bytes, content_type = media._parse_base64_data_uri(data_uri)

    assert content_bytes == original_bytes
    assert content_type == "text/plain"


def test_data_uri_without_mime_type():
    """Verify a data URI without MIME type defaults to text/plain per RFC 2397."""
    original_bytes = b"hello world"
    encoded = base64.b64encode(original_bytes).decode("utf-8")
    data_uri = f"data:;base64,{encoded}"

    media = _make_media()
    content_bytes, content_type = media._parse_base64_data_uri(data_uri)

    assert content_bytes == original_bytes
    assert content_type == "text/plain"
