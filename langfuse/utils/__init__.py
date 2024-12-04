"""@private"""

import logging
import typing
from datetime import datetime, timezone

import pydantic


from langfuse.model import ModelUsage, PromptClient

IS_PYDANTIC_V2 = pydantic.VERSION.startswith("2.")

if IS_PYDANTIC_V2:
    import pydantic.v1 as pydantic_v1  # noqa
    import pydantic as pydantic_v2  # noqa
else:
    import pydantic as pydantic_v1  # noqa

    pydantic_v2 = None  # type: ignore

log = logging.getLogger("langfuse")


def _get_timestamp():
    return datetime.now(timezone.utc)


def _create_prompt_context(
    prompt: typing.Optional[PromptClient] = None,
):
    if prompt is not None and not prompt.is_fallback:
        return {"prompt_version": prompt.version, "prompt_name": prompt.name}

    return {"prompt_version": None, "prompt_name": None}


T = typing.TypeVar("T")


def extract_by_priority(
    usage: dict, keys: typing.List[str], target_type: typing.Type[T]
) -> typing.Optional[T]:
    """Extracts the first key that exists in usage and converts its value to target_type"""
    for key in keys:
        if key in usage:
            value = usage[key]
            try:
                if value is None:
                    return None
                return target_type(value)
            except Exception:
                continue
    return None


def _convert_usage_input(usage: typing.Union[pydantic_v1.BaseModel, ModelUsage]):
    """Convert any usage input to a usage object.

    Deprecated, only used for backwards compatibility with legacy 'usage' objects in generation create / update
    """
    if isinstance(usage, pydantic_v1.BaseModel):
        usage = usage.dict()

    # sometimes we do not match the pydantic usage object
    # in these cases, we convert to dict manually
    if hasattr(usage, "__dict__"):
        usage = usage.__dict__

    # validate that usage object has input, output, total, usage
    is_langfuse_usage = any(k in usage for k in ("input", "output", "total", "unit"))

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
        usage = {
            "input": extract_by_priority(usage, ["promptTokens", "prompt_tokens"], int),
            "output": extract_by_priority(
                usage,
                ["completionTokens", "completion_tokens"],
                int,
            ),
            "total": extract_by_priority(usage, ["totalTokens", "total_tokens"], int),
            "unit": "TOKENS",
            "inputCost": extract_by_priority(usage, ["inputCost", "input_cost"], float),
            "outputCost": extract_by_priority(
                usage, ["outputCost", "output_cost"], float
            ),
            "totalCost": extract_by_priority(usage, ["totalCost", "total_cost"], float),
        }
        return usage

    if not is_langfuse_usage and not is_openai_usage:
        raise ValueError(
            "Usage object must have either {input, output, total, unit} or {promptTokens, completionTokens, totalTokens}"
        )


def _extract_usage_details(usage_details: typing.Dict[str, typing.Any]):
    if isinstance(usage_details, pydantic_v1.BaseModel):
        usage_details = usage_details.dict()

    if pydantic_v2 is not None and isinstance(usage_details, pydantic_v2.BaseModel):
        usage_details = usage_details.model_dump()

    if hasattr(usage_details, "__dict__"):
        usage_details = usage_details.__dict__

    # Handle openai usage details
    if all(
        k in usage_details
        for k in ("prompt_tokens", "completion_tokens", "total_tokens")
    ) or all(
        k in usage_details for k in ("promptTokens", "completionTokens", "totalTokens")
    ):
        openai_usage_details = {
            "input": usage_details.get("prompt_tokens", None)
            or usage_details.get("promptTokens", None),
            "output": usage_details.get("completion_tokens", None)
            or usage_details.get("completionTokens", None),
            "total": usage_details.get("total_tokens", None)
            or usage_details.get("totalTokens", None),
        }

        # Handle input token details
        prompt_tokens_details = usage_details.get("prompt_tokens_details", {})
        if pydantic_v2 is not None and isinstance(
            prompt_tokens_details, pydantic_v2.BaseModel
        ):
            prompt_tokens_details = prompt_tokens_details.model_dump()
        elif hasattr(prompt_tokens_details, "__dict__"):
            prompt_tokens_details = prompt_tokens_details.__dict__

        if isinstance(prompt_tokens_details, dict):
            for key in prompt_tokens_details:
                openai_usage_details[f"input_{key}"] = prompt_tokens_details[key]
                openai_usage_details["input"] = max(
                    openai_usage_details.get("input", 0)
                    - openai_usage_details[f"input_{key}"],
                    0,
                )

        # Handle output token details
        completion_tokens_details = usage_details.get("completion_tokens_details", {})
        if pydantic_v2 is not None and isinstance(
            completion_tokens_details, pydantic_v2.BaseModel
        ):
            completion_tokens_details = completion_tokens_details.model_dump()
        elif hasattr(completion_tokens_details, "__dict__"):
            completion_tokens_details = completion_tokens_details.__dict__

        if isinstance(completion_tokens_details, dict):
            for key in completion_tokens_details:
                openai_usage_details[f"output_{key}"] = completion_tokens_details[key]
                openai_usage_details["output"] = max(
                    openai_usage_details.get("output", 0)
                    - openai_usage_details[f"output_{key}"],
                    0,
                )

        # Remove input and output if they are 0, i.e. all details add up to the total provided by OpenAI
        if openai_usage_details["input"] == 0:
            openai_usage_details.pop("input")
        if openai_usage_details["output"] == 0:
            openai_usage_details.pop("output")

        return openai_usage_details

    return usage_details
