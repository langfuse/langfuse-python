"""Unit tests for langfuse.langchain.utils model-name extraction."""

from langfuse.langchain.utils import _extract_model_name


def _chat_huggingface_serialized(model_id: str) -> dict:
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
        "repr": (
            f"ChatHuggingFace(llm=HuggingFaceEndpoint(repo_id='{model_id}', "
            f"model='{model_id}', task='text-generation'), model_id='{model_id}')"
        ),
        "name": "ChatHuggingFace",
    }


def test_extract_model_name_chat_huggingface():
    serialized = _chat_huggingface_serialized("Qwen/Qwen2.5-Coder-32B-Instruct")

    assert _extract_model_name(serialized) == "Qwen/Qwen2.5-Coder-32B-Instruct"
