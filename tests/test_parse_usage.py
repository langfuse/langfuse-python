from langfuse.openai import _parse_usage


class TestParseUsageNone:
    def test_returns_none_for_none(self):
        assert _parse_usage(None) is None


class TestParseUsageEmbedding:
    def test_embedding_usage_returns_input_only(self):
        usage = {"prompt_tokens": 5, "total_tokens": 5}
        result = _parse_usage(usage)
        assert result == {"input": 5}


class TestParseUsageChatCompletions:
    def test_prompt_tokens_details_flattened(self):
        usage = {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
            "prompt_tokens_details": {"cached_tokens": 20, "audio_tokens": None},
            "completion_tokens_details": {"reasoning_tokens": 10},
        }
        result = _parse_usage(usage)
        assert result["prompt_tokens"] == 100
        assert result["completion_tokens"] == 50
        assert result["total_tokens"] == 150
        # None values are stripped, non-None kept
        assert result["prompt_tokens_details"] == {"cached_tokens": 20}
        assert result["completion_tokens_details"] == {"reasoning_tokens": 10}

    def test_details_as_object(self):
        """Token details may arrive as an object with __dict__ instead of a dict."""

        class Details:
            def __init__(self):
                self.cached_tokens = 30
                self.audio_tokens = None

        usage = {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
            "prompt_tokens_details": Details(),
            "completion_tokens_details": None,
        }
        result = _parse_usage(usage)
        assert result["prompt_tokens_details"] == {"cached_tokens": 30}
        assert result["completion_tokens_details"] is None


class TestParseUsageResponseApi:
    """Tests for OpenAI Response API usage format (input_tokens_details / output_tokens_details)."""

    def test_input_and_output_tokens_details_flattened(self):
        usage = {
            "input_tokens": 200,
            "output_tokens": 80,
            "total_tokens": 280,
            "input_tokens_details": {"cached_tokens": 50},
            "output_tokens_details": {"reasoning_tokens": 30},
        }
        result = _parse_usage(usage)
        assert result["input_tokens"] == 200
        assert result["output_tokens"] == 80
        assert result["input_tokens_details"] == {"cached_tokens": 50}
        assert result["output_tokens_details"] == {"reasoning_tokens": 30}

    def test_none_values_stripped_from_details(self):
        usage = {
            "input_tokens": 200,
            "output_tokens": 80,
            "total_tokens": 280,
            "input_tokens_details": {"cached_tokens": 50, "audio_tokens": None},
            "output_tokens_details": {"reasoning_tokens": None},
        }
        result = _parse_usage(usage)
        assert result["input_tokens_details"] == {"cached_tokens": 50}
        assert result["output_tokens_details"] == {}

    def test_details_as_object(self):
        class InputDetails:
            def __init__(self):
                self.cached_tokens = 40

        class OutputDetails:
            def __init__(self):
                self.reasoning_tokens = 15

        usage = {
            "input_tokens": 100,
            "output_tokens": 50,
            "total_tokens": 150,
            "input_tokens_details": InputDetails(),
            "output_tokens_details": OutputDetails(),
        }
        result = _parse_usage(usage)
        assert result["input_tokens_details"] == {"cached_tokens": 40}
        assert result["output_tokens_details"] == {"reasoning_tokens": 15}
