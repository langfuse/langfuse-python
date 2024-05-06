"""@private"""

import re
from typing import Any, Dict, List, Literal, Optional

# NOTE ON DEPENDENCIES:
# - since Jan 2024, there is https://pypi.org/project/langchain-openai/ which is a separate package and imports openai models.
#   Decided to not make this a dependency of langfuse as few people will have this. Need to match these models manually
# - langchain_community is loaded as a dependency of langchain, so we can use it here


def _extract_model_name(
    serialized: Dict[str, Any],
    **kwargs: Any,
):
    """Extracts the model name from the serialized or kwargs object. This is used to get the model names for Langfuse."""
    # we have to deal with ChatGoogleGenerativeAI and ChatMistralAI first, as
    # if we run loads(dumps(serialized)) on it, it will throw in case of missing api keys
    print(serialized, kwargs)
    models = [
        ("ChatGoogleGenerativeAI", ["kwargs", "model"], "serialized"),
        ("ChatMistralAI", ["kwargs", "model"], "serialized"),
        ("ChatVertexAi", ["kwargs", "model_name"], "serialized"),
        ("ChatVertexAI", ["kwargs", "model_name"], "serialized"),
        ("OpenAI", ["invocation_params", "model_name"], "kwargs"),
        ("ChatOpenAI", ["invocation_params", "model_name"], "kwargs"),
        ("AzureChatOpenAI", ["invocation_params", "model"], "kwargs"),
        ("AzureChatOpenAI", ["invocation_params", "model_name"], "kwargs"),
        ("HuggingFacePipeline", ["invocation_params", "model_id"], "kwargs"),
        ("BedrockChat", ["kwargs", "model_id"], "serialized"),
        ("Bedrock", ["kwargs", "model_id"], "serialized"),
    ]

    for model_name, keys, select_from in models:
        model = _extract_model_by_path_for_id(
            model_name, serialized, kwargs, keys, select_from
        )
        if model:
            return model

    if serialized.get("id")[-1] == "AzureOpenAI":
        if kwargs.get("invocation_params").get("model_name"):
            return kwargs.get("invocation_params").get("model_name")

        deployment_name = None
        if serialized.get("kwargs").get("openai_api_version"):
            deployment_name = serialized.get("kwargs").get("deployment_version")
        deployment_version = None
        if serialized.get("kwargs").get("deployment_name"):
            deployment_name = serialized.get("kwargs").get("deployment_name")
        return deployment_name + "-" + deployment_version

    configs = [
        ("Anthropic", "model", "anthropic"),
        ("ChatAnthropic", "model", None),
        ("ChatTongyi", "model_name", None),
        ("ChatCohere", "model", None),
        ("Cohere", "model", None),
        ("HuggingFaceHub", "model", None),
        ("ChatAnyscale", "model_name", None),
        ("TextGen", "model", "text-gen"),
    ]

    for model_name, pattern, default in configs:
        model = _extract_model_from_repr_by_pattern(
            model_name, serialized, pattern, default
        )
        if model:
            return model

    # try most likely paths
    random_paths = [
        ["kwargs", "model_name"],
        ["kwargs", "model"],
        ["invocation_params", "model_name"],
        ["invocation_params", "model"],
    ]
    for select in ["kwargs", "serialized"]:
        for path in random_paths:
            model = _extract_model_by_path(serialized, kwargs, path, select)
            if model:
                return model

    return None


def _extract_model_from_repr_by_pattern(
    id: str, serialized: dict, pattern: str, default: Optional[str] = None
):
    if serialized.get("id")[-1] == id:
        extracted = _extract_model_with_regex(pattern, serialized["repr"])
        return extracted if extracted else default if default else None


def _extract_model_with_regex(pattern: str, text: str):
    match = re.search(rf"{pattern}='(.*?)'", text)
    if match:
        return match.group(1)
    return None


def _extract_model_by_path_for_id(
    id: str,
    serialized: dict,
    kwargs: dict,
    keys: List[str],
    select_from: str = Literal["serialized", "kwargs"],
):
    print(id)
    if serialized.get("id")[-1] == id:
        return _extract_model_by_path(serialized, kwargs, keys, select_from)


def _extract_model_by_path(
    serialized: dict,
    kwargs: dict,
    keys: List[str],
    select_from: str = Literal["serialized", "kwargs"],
):
    current_obj = kwargs if select_from == "kwargs" else serialized

    for key in keys:
        print(current_obj)
        current_obj = current_obj.get(key)
        if not current_obj:
            return None

    return current_obj if current_obj else None
