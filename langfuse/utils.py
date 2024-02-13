import logging
import typing
from datetime import datetime, timezone

try:
    import pydantic.v1 as pydantic  # type: ignore
except ImportError:
    import pydantic  # type: ignore

from langfuse.model import ModelUsage, PromptClient

log = logging.getLogger("langfuse")


def _get_timestamp():
    return datetime.now(timezone.utc)


def _create_prompt_context(
    prompt: typing.Optional[PromptClient] = None,
):
    if prompt is not None:
        return {"prompt_version": prompt.version, "prompt_name": prompt.name}

    return {"prompt_version": None, "prompt_name": None}


T = typing.TypeVar("T")


def extract_by_priority(usage: dict, keys: typing.List[str]) -> typing.Optional[T]:
    """Extracts the first key that exists in usage and converts its value to target_type"""

    for key in keys:
        if key in usage:
            value = usage[key]
            try:
                if value is None:
                    return None
                return value  # Remove the type argument from the return statement
            except Exception:
                continue
    return None


def _convert_usage_input(usage: typing.Union[pydantic.BaseModel, ModelUsage]):
    """Converts any usage input to a usage object"""

    # converts usage to dict if it is a pydantic model
    usage_dict = (
        usage.dict() if isinstance(usage, pydantic.BaseModel) else usage.__dict__
    )

    # validate that usage object has input, output, total, usage
    is_langfuse_usage = any(
        k in usage_dict for k in ("input", "output", "total", "unit")
    )

    if is_langfuse_usage:
        return usage

    is_openai_usage = any(
        k in usage
        for k in (
            "promptTokens",
            "prompt_tokens",
            "completionTokens",
            "completion_tokens",
            "totalTokens",
            "total_tokens",
            "inputCost",
            "input_cost",
            "outputCost",
            "output_cost",
            "totalCost",
            "total_cost",
        )
    )

    if is_openai_usage:
        # convert to langfuse usage
        return ModelUsage(
            unit="TOKENS",
            input=extract_by_priority(usage_dict, ["promptTokens", "prompt_tokens"]),
            output=extract_by_priority(
                usage_dict, ["completionTokens", "completion_tokens"]
            ),
            total=extract_by_priority(usage_dict, ["totalTokens", "total_tokens"]),
            input_cost=extract_by_priority(usage_dict, ["inputCost", "input_cost"]),
            output_cost=extract_by_priority(usage_dict, ["outputCost", "output_cost"]),
            total_cost=extract_by_priority(usage_dict, ["totalCost", "total_cost"]),
        )

    if not is_langfuse_usage and not is_openai_usage:
        raise ValueError(
            "Usage object must have either {input, output, total, unit} or {promptTokens, completionTokens, totalTokens}"
        )
