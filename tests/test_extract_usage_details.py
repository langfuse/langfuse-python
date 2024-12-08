from langfuse.client import _extract_usage_details
from tests.utils import CompletionUsage, LlmUsage, LlmUsageWithCost


def test_extract_usage_details_openai_style():
    usage = {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}

    result = _extract_usage_details(usage)

    assert result == {"input": 10, "output": 20, "total": 30}


def test_extract_usage_details_openai_with_token_details():
    usage = {
        "prompt_tokens": 10,
        "completion_tokens": 20,
        "total_tokens": 30,
        "completion_token_details": {"audio_tokens": 5},
        "prompt_token_details": {"audio_tokens": 3, "cached_tokens": 2},
    }

    result = _extract_usage_details(usage)

    assert result == {
        "input": 5,
        "input_audio": 3,
        "input_cached": 2,
        "output": 15,
        "output_audio": 5,
        "total": 30,
    }


def test_extract_usage_details_openai_with_completion_token_details_only():
    usage = {
        "prompt_tokens": 10,
        "completion_tokens": 20,
        "total_tokens": 30,
        "completion_token_details": {"audio_tokens": 5},
    }

    result = _extract_usage_details(usage)

    assert result == {
        "input": 10,
        "output": 15,
        "output_audio": 5,
        "total": 30,
    }


def test_extract_usage_details_openai_with_prompt_token_details_only():
    usage = {
        "prompt_tokens": 10,
        "completion_tokens": 20,
        "total_tokens": 30,
        "prompt_token_details": {"cached_tokens": 3, "audio_tokens": 7},
    }

    result = _extract_usage_details(usage)

    assert result == {
        "input": 0,
        "input_cached": 3,
        "input_audio": 7,
        "output": 20,
        "total": 30,
    }


def test_extract_usage_details_pydantic_openai():
    usage = CompletionUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30)

    result = _extract_usage_details(usage.dict())

    assert result == {"input": 10, "output": 20, "total": 30}


def test_extract_usage_details_llm_usage():
    usage = LlmUsage(promptTokens=10, completionTokens=20, totalTokens=30)

    result = _extract_usage_details(usage.dict())

    assert result == {"input": 10, "output": 20, "total": 30}


def test_extract_usage_details_llm_usage_with_cost():
    usage = LlmUsageWithCost(
        promptTokens=10,
        completionTokens=20,
        totalTokens=30,
        inputCost=0.0001,
        outputCost=0.0002,
        totalCost=0.0003,
    )

    result = _extract_usage_details(usage.dict())

    assert result == {"input": 10, "output": 20, "total": 30}


def test_extract_usage_details_raw():
    usage = {"input": 100, "output": 200, "total": 300}

    result = _extract_usage_details(usage)

    assert result == usage


def test_extract_usage_details_raw_with_cached():
    usage = {"input": 100, "input_cached": 50, "output": 200, "total": 300}

    result = _extract_usage_details(usage)

    assert result == usage


def test_extract_usage_details_empty():
    result = _extract_usage_details({})

    assert result == {}


def test_extract_usage_details_invalid_keys():
    usage = {"foo": 10, "bar": 20}

    result = _extract_usage_details(usage)

    assert result == usage
