from langfuse.langchain.CallbackHandler import _parse_usage_model


def test_anthropic_cache_creation_dict_flattened():
    """Anthropic extended caching: cache_creation dict is flattened into per-tier keys
    and an aggregated cache_creation_input_tokens total is added."""
    usage = {
        "input_tokens": 9454,
        "output_tokens": 380,
        "cache_read_input_tokens": 0,
        "cache_creation": {
            "ephemeral_1h_input_tokens": 500,
            "ephemeral_5m_input_tokens": 200,
        },
    }
    result = _parse_usage_model(usage)

    # Core fields survive
    assert result["input"] == 9454
    assert result["output"] == 380
    assert result["cache_read_input_tokens"] == 0

    # Per-tier keys are present and individually correct
    assert result["cache_creation_ephemeral_1h_input_tokens"] == 500
    assert result["cache_creation_ephemeral_5m_input_tokens"] == 200

    # Aggregated total equals sum of all tiers
    assert result["cache_creation_input_tokens"] == 700

    # The original nested dict must not be present
    assert "cache_creation" not in result


def test_anthropic_cache_creation_all_zeros_no_aggregate():
    """When all cache_creation tier values are zero no aggregate key is added
    (avoids noise in traces where caching did not fire)."""
    usage = {
        "input_tokens": 100,
        "output_tokens": 50,
        "cache_creation": {
            "ephemeral_1h_input_tokens": 0,
            "ephemeral_5m_input_tokens": 0,
        },
    }
    result = _parse_usage_model(usage)

    assert result["input"] == 100
    assert result["output"] == 50
    # Per-tier zero keys are still stored
    assert result["cache_creation_ephemeral_1h_input_tokens"] == 0
    assert result["cache_creation_ephemeral_5m_input_tokens"] == 0
    # No aggregate added when total is zero
    assert "cache_creation_input_tokens" not in result
    assert "cache_creation" not in result


def test_anthropic_cache_creation_legacy_field_not_overwritten():
    """If both the legacy cache_creation_input_tokens (int) and the new cache_creation
    (dict) are present, the legacy value is preserved and the dict total is not added."""
    usage = {
        "input_tokens": 100,
        "output_tokens": 50,
        "cache_creation_input_tokens": 300,  # legacy field already present
        "cache_creation": {
            "ephemeral_1h_input_tokens": 200,
            "ephemeral_5m_input_tokens": 100,
        },
    }
    result = _parse_usage_model(usage)

    # setdefault must not overwrite the existing legacy value
    assert result["cache_creation_input_tokens"] == 300
    assert "cache_creation" not in result


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
