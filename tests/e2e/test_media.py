import base64
import re
from uuid import uuid4

from langfuse._client.client import Langfuse
from langfuse.media import LangfuseMedia
from tests.support.utils import wait_for_observations


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

    fetched_observations = wait_for_observations(
        span.trace_id,
        is_result_ready=lambda observations: (
            re.match(
                r"^@@@langfuseMedia:type=audio/wav\|id=.+\|source=base64_data_uri@@@$",
                observations[0].metadata.get("context", {}).get("nested", ""),
            )
            is not None
        ),
    )
    assert len(fetched_observations) == 1
    fetched_observation = fetched_observations[0]
    media_ref = fetched_observation.metadata["context"]["nested"]
    assert re.match(
        r"^@@@langfuseMedia:type=audio/wav\|id=.+\|source=base64_data_uri@@@$",
        media_ref,
    )

    resolved_obs = langfuse.resolve_media_references(
        obj=fetched_observation, resolve_with="base64_data_uri"
    )

    expected_base64 = f"data:audio/wav;base64,{base64_audio}"
    assert resolved_obs["metadata"]["context"]["nested"] == expected_base64

    span2 = langfuse.start_observation(
        name=f"2-{mock_trace_name}",
        metadata={"context": {"nested": resolved_obs["metadata"]["context"]["nested"]}},
    ).end()

    langfuse.flush()

    fetched_observations2 = wait_for_observations(
        span2.trace_id,
        is_result_ready=lambda observations: (
            observations[0].metadata.get("context", {}).get("nested") == media_ref
        ),
    )
    assert len(fetched_observations2) == 1
    assert fetched_observations2[0].metadata["context"]["nested"] == media_ref
