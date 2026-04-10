from langfuse.langchain.CallbackHandler import _parse_usage_model


def test_standard_tier_input_token_details():
    """Standard tier: audio and cache_read are subtracted from input."""
    usage = {
        "input_tokens": 13,
        "output_tokens": 1,
        "total_tokens": 14,
        "input_token_details": {"audio": 0, "cache_read": 3},
        "output_token_details": {"audio": 0},
    }
    result = _parse_usage_model(usage)
    assert result["input"] == 10  # 13 - 0 (audio) - 3 (cache_read)
    assert result["output"] == 1  # 1 - 0 (audio)
    assert result["total"] == 14


def test_prompt_tokens_details_dict_cached_tokens():
    """OpenAI/LiteLLM: prompt_tokens_details as dict with cached_tokens."""
    usage = {
        "prompt_tokens": 15000,
        "completion_tokens": 500,
        "total_tokens": 15500,
        "prompt_tokens_details": {"cached_tokens": 12000},
    }
    result = _parse_usage_model(usage)
    assert result["input"] == 3000  # 15000 - 12000
    assert result["output"] == 500
    assert result["total"] == 15500
    assert result["input_cached_tokens"] == 12000


def test_prompt_tokens_details_dict_with_cache_creation():
    """OpenAI/LiteLLM: prompt_tokens_details dict + top-level cache_creation."""
    usage = {
        "prompt_tokens": 15000,
        "completion_tokens": 500,
        "total_tokens": 15500,
        "prompt_tokens_details": {"cached_tokens": 12000},
        "cache_creation_input_tokens": 3000,
    }
    result = _parse_usage_model(usage)
    assert result["input"] == 3000  # 15000 - 12000 (cached_tokens only subtracted here)
    assert result["input_cached_tokens"] == 12000
    assert result["cache_creation_input_tokens"] == 3000


def test_prompt_tokens_details_list_vertex_ai():
    """Vertex AI: prompt_tokens_details as list — existing behavior preserved."""
    usage = {
        "prompt_token_count": 1000,
        "candidates_token_count": 200,
        "total_token_count": 1200,
        "prompt_tokens_details": [
            {"modality": "text", "token_count": 800},
            {"modality": "image", "token_count": 200},
        ],
    }
    result = _parse_usage_model(usage)
    assert result["input"] == 0  # 1000 - 800 - 200
    assert result["output"] == 200
    assert result["total"] == 1200
    assert result["input_modality_text"] == 800
    assert result["input_modality_image"] == 200


def test_prompt_tokens_details_dict_empty():
    """Empty dict prompt_tokens_details — no crash, input unchanged."""
    usage = {
        "prompt_tokens": 5000,
        "completion_tokens": 100,
        "total_tokens": 5100,
        "prompt_tokens_details": {},
    }
    result = _parse_usage_model(usage)
    assert result["input"] == 5000
    assert result["output"] == 100


def test_priority_tier_not_subtracted():
    """Priority tier: 'priority' and 'priority_*' keys must NOT be subtracted."""
    usage = {
        "input_tokens": 13,
        "output_tokens": 1,
        "total_tokens": 14,
        "input_token_details": {"audio": 0, "priority_cache_read": 0, "priority": 13},
        "output_token_details": {"audio": 0, "priority_reasoning": 0, "priority": 1},
    }
    result = _parse_usage_model(usage)
    assert result["input"] == 13  # priority keys not subtracted
    assert result["output"] == 1
    assert result["total"] == 14
    # Priority keys are still stored with prefixed names
    assert result["input_priority"] == 13
    assert result["output_priority"] == 1
