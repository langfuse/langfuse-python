import base64
import re
from uuid import uuid4

import pytest

from langfuse._client.client import Langfuse
from langfuse.media import LangfuseMedia
from tests.utils import get_api

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
    # Reference string should be None initially as media_id is not set
    assert media._reference_string is None

    # Set media_id
    media._media_id = "test-id"
    reference = media._reference_string
    assert reference is not None
    assert "test-id" in reference
    assert "image/jpeg" in reference
    assert "bytes" in reference


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


def test_replace_media_reference_string_in_object():
    # Create test audio file
    audio_file = "static/joke_prompt.wav"
    with open(audio_file, "rb") as f:
        mock_audio_bytes = f.read()

    # Create Langfuse client and trace with media
    langfuse = Langfuse()

    mock_trace_name = f"test-trace-with-audio-{uuid4()}"
    base64_audio = base64.b64encode(mock_audio_bytes).decode()

    span = langfuse.start_span(
        name=mock_trace_name,
        metadata={
            "context": {
                "nested": LangfuseMedia(
                    base64_data_uri=f"data:audio/wav;base64,{base64_audio}"
                )
            }
        },
    ).end()

    langfuse.flush()

    # Verify media reference string format
    fetched_trace = get_api().trace.get(span.trace_id)
    media_ref = fetched_trace.observations[0].metadata["context"]["nested"]
    assert re.match(
        r"^@@@langfuseMedia:type=audio/wav\|id=.+\|source=base64_data_uri@@@$",
        media_ref,
    )

    # Resolve media references back to base64
    resolved_obs = langfuse.resolve_media_references(
        obj=fetched_trace.observations[0], resolve_with="base64_data_uri"
    )

    # Verify resolved base64 matches original
    expected_base64 = f"data:audio/wav;base64,{base64_audio}"
    assert resolved_obs["metadata"]["context"]["nested"] == expected_base64

    # Create second trace reusing the media reference
    span2 = langfuse.start_span(
        name=f"2-{mock_trace_name}",
        metadata={"context": {"nested": resolved_obs["metadata"]["context"]["nested"]}},
    ).end()

    langfuse.flush()

    # Verify second trace has same media reference
    fetched_trace2 = get_api().trace.get(span2.trace_id)
    assert (
        fetched_trace2.observations[0].metadata["context"]["nested"]
        == fetched_trace.observations[0].metadata["context"]["nested"]
    )
