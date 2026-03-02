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
