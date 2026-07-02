"""Unit tests for langfuse.langchain.utils model-name extraction."""

import pytest

from langfuse.langchain.utils import _extract_model_name

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
