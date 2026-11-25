"""Unit tests for langfuse.langchain.utils model-name extraction."""

import pytest

from langfuse.langchain.utils import _extract_model_name, _extract_model_with_regex

_MODEL_ID = "Qwen/Qwen2.5-Coder-32B-Instruct"


def _chat_huggingface_serialized(inner_repr: str) -> dict:
    """Mirror ``langchain_core.load.dumpd(ChatHuggingFace(...))``.

    ChatHuggingFace is not LangChain-serializable, so it serializes to a
    ``not_implemented`` stub with no ``kwargs``; the model id is only available
    inside the ``repr`` string as ``model_id='...'``.
    """
    return {
        "lc": 1,
        "type": "not_implemented",
        "id": [
            "langchain_huggingface",
            "chat_models",
            "huggingface",
            "ChatHuggingFace",
        ],
        "repr": f"ChatHuggingFace(llm={inner_repr}, model_id='{_MODEL_ID}')",
        "name": "ChatHuggingFace",
    }


@pytest.mark.parametrize(
    "inner_repr",
    [
        f"HuggingFaceEndpoint(repo_id='{_MODEL_ID}', model='{_MODEL_ID}', task='text-generation')",
        f"HuggingFaceHub(repo_id='{_MODEL_ID}')",
        # HuggingFacePipeline exposes its own model_id, so the repr carries two
        # model_id='...' occurrences; re.search picks the first (inner) one.
        f"HuggingFacePipeline(model_id='{_MODEL_ID}')",
    ],
)
def test_extract_model_name_chat_huggingface(inner_repr: str):
    serialized = _chat_huggingface_serialized(inner_repr)

    assert _extract_model_name(serialized) == _MODEL_ID


def test_extract_model_with_regex_does_not_match_substring_field_names():
    assert (
        _extract_model_with_regex(
            "model", "ChatAnthropic(embedding_model='text-embedding-3-small')"
        )
        is None
    )
    assert (
        _extract_model_with_regex(
            "model_id", "ChatHuggingFace(embedding_model_id='text-embedding-3-small')"
        )
        is None
    )


def test_extract_model_with_regex_extracts_exact_field_when_substring_exists():
    repr_str = (
        "ChatAnthropic(embedding_model='text-embedding-3-small', "
        "model='claude-3-5-sonnet')"
    )
    assert _extract_model_with_regex("model", repr_str) == "claude-3-5-sonnet"


def test_extract_model_with_regex_valid_and_missing_fields():
    assert (
        _extract_model_with_regex("model", "ChatAnthropic(model='claude-3-5-sonnet')")
        == "claude-3-5-sonnet"
    )
    assert _extract_model_with_regex("model", "ChatAnthropic(temperature=0.0)") is None


def test_extract_model_name_ignores_embedding_model_in_repr():
    serialized = {
        "lc": 1,
        "type": "not_implemented",
        "id": ["langchain_anthropic", "chat_models", "ChatAnthropic"],
        "repr": (
            "ChatAnthropic(embedding_model='text-embedding-3-small', "
            "model='claude-3-5-sonnet')"
        ),
    }

    assert _extract_model_name(serialized) == "claude-3-5-sonnet"
