import base64
import re
from uuid import uuid4

from langfuse._client.client import Langfuse
from langfuse.media import LangfuseMedia
from tests.support.utils import wait_for_trace


def test_replace_media_reference_string_in_object():
    audio_file = "static/joke_prompt.wav"
    with open(audio_file, "rb") as f:
        mock_audio_bytes = f.read()

    langfuse = Langfuse()

    mock_trace_name = f"test-trace-with-audio-{uuid4()}"
    base64_audio = base64.b64encode(mock_audio_bytes).decode()

    span = langfuse.start_observation(
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

    fetched_trace = wait_for_trace(
        span.trace_id,
        is_result_ready=lambda trace: (
            bool(trace.observations)
            and re.match(
                r"^@@@langfuseMedia:type=audio/wav\|id=.+\|source=base64_data_uri@@@$",
                trace.observations[0].metadata.get("context", {}).get("nested", ""),
            )
            is not None
        ),
    )
    media_ref = fetched_trace.observations[0].metadata["context"]["nested"]
    assert re.match(
        r"^@@@langfuseMedia:type=audio/wav\|id=.+\|source=base64_data_uri@@@$",
        media_ref,
    )

    resolved_obs = langfuse.resolve_media_references(
        obj=fetched_trace.observations[0], resolve_with="base64_data_uri"
    )

    expected_base64 = f"data:audio/wav;base64,{base64_audio}"
    assert resolved_obs["metadata"]["context"]["nested"] == expected_base64

    span2 = langfuse.start_observation(
        name=f"2-{mock_trace_name}",
        metadata={"context": {"nested": resolved_obs["metadata"]["context"]["nested"]}},
    ).end()

    langfuse.flush()

    fetched_trace2 = wait_for_trace(
        span2.trace_id,
        is_result_ready=lambda trace: (
            bool(trace.observations)
            and trace.observations[0].metadata.get("context", {}).get("nested")
            == fetched_trace.observations[0].metadata["context"]["nested"]
        ),
    )
    assert (
        fetched_trace2.observations[0].metadata["context"]["nested"]
        == fetched_trace.observations[0].metadata["context"]["nested"]
    )
