from langfuse.langchain.CallbackHandler import _parse_usage_model


def test_parse_usage_model_skips_priority_subtraction():
    usage = {
        "input": 13,
        "output": 1,
        "total": 14,
        "input_token_details": {
            "audio": 0,
            "priority_cache_read": 0,
            "priority": 13,
        },
        "output_token_details": {
            "audio": 0,
            "priority_reasoning": 0,
            "priority": 1,
        },
    }

    parsed = _parse_usage_model(usage)

    assert parsed["input"] == 13
    assert parsed["output"] == 1
    assert parsed["total"] == 14


def test_parse_usage_model_subtracts_known_details():
    usage = {
        "input": 100,
        "output": 50,
        "total": 150,
        "input_token_details": {
            "cache_read": 20,
            "audio": 5,
        },
        "output_token_details": {
            "reasoning": 10,
        },
    }

    parsed = _parse_usage_model(usage)

    assert parsed["input"] == 75
    assert parsed["output"] == 40
    assert parsed["input_cache_read"] == 20
    assert parsed["input_audio"] == 5
    assert parsed["output_reasoning"] == 10
    assert parsed["total"] == 150
