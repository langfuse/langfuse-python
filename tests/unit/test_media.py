import base64
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from langfuse._client.resource_manager import LangfuseResourceManager
from langfuse.media import LangfuseMedia, LangfuseMediaReference

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


def test_parse_reference_string_with_empty_field_values_raises_missing_fields_error():
    # All three required keys present, but with empty values, used to pass
    # the "key in parsed_data" check and return a reference with an empty
    # media_id -- which would then reach a real api.media.get("") network
    # call in LangfuseMedia.resolve_media_references.
    with pytest.raises(ValueError, match="Missing required fields"):
        LangfuseMedia.parse_reference_string("@@@langfuseMedia:type=|id=|source=@@@")


def test_parse_reference_string_with_malformed_pair_raises_missing_fields_error():
    # A pipe-separated segment with no "=" (here, a typo dropping the "="
    # from "id=") used to crash with a raw "not enough values to unpack"
    # ValueError from the key/value split, instead of the intended
    # "Missing required fields" validation error.
    with pytest.raises(ValueError, match="Missing required fields"):
        LangfuseMedia.parse_reference_string(
            "@@@langfuseMedia:type=image/jpeg|idtest-id|source=bytes@@@"
        )


def test_parse_reference_string_ignores_malformed_pair_when_fields_still_present():
    # A malformed segment alongside all three required fields elsewhere in
    # the string should be ignored, not crash the whole parse.
    result = LangfuseMedia.parse_reference_string(
        "@@@langfuseMedia:type=image/jpeg|badpair|id=test-id|source=bytes@@@"
    )

    assert result["media_id"] == "test-id"
    assert result["content_type"] == "image/jpeg"
    assert result["source"] == "bytes"


def test_parse_reference_string_ignores_trailing_empty_pair():
    # A trailing "|" produces an empty segment with no "=", which should be
    # ignored rather than crash, as long as the required fields are present.
    result = LangfuseMedia.parse_reference_string(
        "@@@langfuseMedia:type=image/jpeg|id=test-id|source=bytes|@@@"
    )

    assert result["media_id"] == "test-id"
    assert result["content_type"] == "image/jpeg"
    assert result["source"] == "bytes"


@pytest.mark.parametrize(
    ("url_expiry", "expected"),
    [
        (None, False),
        ("not-a-date", False),
        # Fixed past/future timestamps so the test ids stay stable across xdist
        # workers (a computed datetime.now() would differ per worker collection).
        ("2000-01-01T00:00:00+00:00", True),
        ("2999-01-01T00:00:00+00:00", False),
        ("2000-01-01T00:00:00Z", True),  # "Z" suffix, in the past
    ],
)
def test_media_reference_is_url_expired(url_expiry, expected):
    reference = LangfuseMediaReference(
        media_id="media-id",
        content_type="image/jpeg",
        url="https://example.com/test.jpg",
        url_expiry=url_expiry,
    )

    assert reference.is_url_expired() is expected


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


def test_media_reference_fetch_uses_configured_httpx_client(monkeypatch):
    response = Mock()
    response.content = b"test-bytes"
    response.raise_for_status.return_value = None
    configured_httpx_client = Mock()
    configured_httpx_client.get.return_value = response
    httpx_get = Mock()
    monkeypatch.setattr("langfuse.media.httpx.get", httpx_get)
    monkeypatch.setattr(
        LangfuseResourceManager,
        "_instances",
        {"pk-test": SimpleNamespace(httpx_client=configured_httpx_client)},
    )

    reference = LangfuseMediaReference(
        media_id="media-id",
        content_type="image/jpeg",
        url="https://example.com/test.jpg",
    )

    assert reference.fetch_bytes(timeout=12.5) == b"test-bytes"
    configured_httpx_client.get.assert_called_once_with(
        "https://example.com/test.jpg", timeout=12.5
    )
    httpx_get.assert_not_called()


def test_media_reference_fetch_uses_explicit_client(monkeypatch):
    response = Mock()
    response.content = b"explicit-bytes"
    response.raise_for_status.return_value = None
    explicit_client = Mock()
    explicit_client.get.return_value = response

    singleton_client = Mock()
    httpx_get = Mock()
    monkeypatch.setattr("langfuse.media.httpx.get", httpx_get)
    monkeypatch.setattr(
        LangfuseResourceManager,
        "_instances",
        {"pk-test": SimpleNamespace(httpx_client=singleton_client)},
    )

    reference = LangfuseMediaReference(
        media_id="media-id",
        content_type="image/jpeg",
        url="https://example.com/test.jpg",
    )

    assert (
        reference.fetch_bytes(timeout=5.0, client=explicit_client) == b"explicit-bytes"
    )
    explicit_client.get.assert_called_once_with(
        "https://example.com/test.jpg", timeout=5.0
    )
    # Explicit client wins over the configured singleton and the default httpx.
    singleton_client.get.assert_not_called()
    httpx_get.assert_not_called()


def test_media_reference_fetch_falls_back_to_default_with_multiple_clients(
    monkeypatch, caplog
):
    import logging

    response = Mock()
    response.content = b"default-bytes"
    response.raise_for_status.return_value = None
    httpx_get = Mock(return_value=response)
    monkeypatch.setattr("langfuse.media.httpx.get", httpx_get)

    client_a = Mock()
    client_b = Mock()
    monkeypatch.setattr(
        LangfuseResourceManager,
        "_instances",
        {
            "pk-a": SimpleNamespace(httpx_client=client_a),
            "pk-b": SimpleNamespace(httpx_client=client_b),
        },
    )

    reference = LangfuseMediaReference(
        media_id="media-id",
        content_type="image/jpeg",
        url="https://example.com/test.jpg",
    )

    with caplog.at_level(logging.WARNING, logger="langfuse"):
        assert reference.fetch_bytes(timeout=8.0) == b"default-bytes"

    # Ambiguous multi-client setup: warn and fall back to the default httpx
    # instead of silently using an arbitrary instance's transport config.
    assert "Multiple Langfuse clients" in caplog.text
    httpx_get.assert_called_once_with("https://example.com/test.jpg", timeout=8.0)
    client_a.get.assert_not_called()
    client_b.get.assert_not_called()


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
