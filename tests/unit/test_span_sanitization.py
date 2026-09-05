from unittest.mock import MagicMock, patch
from opentelemetry import trace as otel_trace_api

from langfuse._client.span import LangfuseObservationWrapper

@patch("langfuse._client.span.LangfuseObservationWrapper._process_media_and_apply_mask")
def test_langfuse_observation_wrapper_metadata_sanitization(mock_process):
    mock_otel_span = MagicMock(spec=otel_trace_api.Span)
    mock_otel_span.is_recording.return_value = True
    
    mock_client = MagicMock()
    
    # Input with empty string key and None key
    bad_metadata = {"valid_key": "value", "": "bad_value", None: "another_bad"}
    
    wrapper = LangfuseObservationWrapper(
        otel_span=mock_otel_span,
        langfuse_client=mock_client,
        as_type="span",
        metadata=bad_metadata
    )
    
    # Check that _process_media_and_apply_mask was called for metadata with the sanitized dict
    mock_process.assert_any_call(data={"valid_key": "value"}, field="metadata", span=mock_otel_span)

