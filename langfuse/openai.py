"""If you use the OpenAI Python SDK, you can use the Langfuse drop-in replacement to get full logging by changing only the import.

```diff
- import openai
+ from langfuse.openai import openai
```

Langfuse automatically tracks:

- All prompts/completions with support for streaming, async and functions
- Latencies
- API Errors
- Model usage (tokens) and cost (USD)

The integration is fully interoperable with the `observe()` decorator and the low-level tracing SDK.

Calls made via the OpenAI SDK's `.with_raw_response` API are traced as well, except for
raw streaming calls which are passed through untraced. Set the environment variable
`LANGFUSE_OPENAI_SKIP_RAW_RESPONSES=True` to exclude all raw-response calls from tracing,
e.g. when another instrumented library (such as LiteLLM) calls the OpenAI SDK internally
through the raw-response API and would otherwise produce duplicate observations.

See docs for more details: https://langfuse.com/docs/integrations/openai
"""

import json
import os
import types
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from inspect import isawaitable, isclass
from typing import Any, Optional, cast

from openai._types import NotGiven, Omit
from packaging.version import Version
from pydantic import BaseModel
from pydantic_core import to_jsonable_python
from wrapt import wrap_function_wrapper

from langfuse._client.environment_variables import (
    LANGFUSE_OPENAI_SKIP_RAW_RESPONSES,
)
from langfuse._client.get_client import get_client
from langfuse._client.span import LangfuseGeneration
from langfuse._utils import _get_timestamp
from langfuse.logger import langfuse_logger as logger
from langfuse.media import LangfuseMedia

try:
    import openai
    from openai import AsyncAzureOpenAI, AsyncOpenAI, AzureOpenAI, OpenAI  # noqa: F401
except ImportError:
    raise ModuleNotFoundError(
        "Please install OpenAI to use this feature: 'pip install openai'"
    )

try:
    from openai._constants import RAW_RESPONSE_HEADER
except ImportError:
    RAW_RESPONSE_HEADER = "X-Stainless-Raw-Response"


@dataclass
class OpenAiDefinition:
    module: str
    object: str
    method: str
    type: str
    sync: bool
    min_version: Optional[str] = None
    max_version: Optional[str] = None


OPENAI_METHODS_V0 = [
    OpenAiDefinition(
        module="openai",
        object="ChatCompletion",
        method="create",
        type="chat",
        sync=True,
    ),
    OpenAiDefinition(
        module="openai",
        object="Completion",
        method="create",
        type="completion",
        sync=True,
    ),
]


OPENAI_METHODS_V1 = [
    OpenAiDefinition(
        module="openai.resources.chat.completions",
        object="Completions",
        method="create",
        type="chat",
        sync=True,
    ),
    OpenAiDefinition(
        module="openai.resources.completions",
        object="Completions",
        method="create",
        type="completion",
        sync=True,
    ),
    OpenAiDefinition(
        module="openai.resources.chat.completions",
        object="AsyncCompletions",
        method="create",
        type="chat",
        sync=False,
    ),
    OpenAiDefinition(
        module="openai.resources.completions",
        object="AsyncCompletions",
        method="create",
        type="completion",
        sync=False,
    ),
    OpenAiDefinition(
        module="openai.resources.beta.chat.completions",
        object="Completions",
        method="parse",
        type="chat",
        sync=True,
        min_version="1.50.0",
        max_version="1.92.0",
    ),
    OpenAiDefinition(
        module="openai.resources.beta.chat.completions",
        object="AsyncCompletions",
        method="parse",
        type="chat",
        sync=False,
        min_version="1.50.0",
        max_version="1.92.0",
    ),
    OpenAiDefinition(
        module="openai.resources.chat.completions",
        object="Completions",
        method="parse",
        type="chat",
        sync=True,
        min_version="1.92.0",
    ),
    OpenAiDefinition(
        module="openai.resources.chat.completions",
        object="AsyncCompletions",
        method="parse",
        type="chat",
        sync=False,
        min_version="1.92.0",
    ),
    OpenAiDefinition(
        module="openai.resources.responses",
        object="Responses",
        method="create",
        type="chat",
        sync=True,
        min_version="1.66.0",
    ),
    OpenAiDefinition(
        module="openai.resources.responses",
        object="AsyncResponses",
        method="create",
        type="chat",
        sync=False,
        min_version="1.66.0",
    ),
    OpenAiDefinition(
        module="openai.resources.responses",
        object="Responses",
        method="parse",
        type="chat",
        sync=True,
        min_version="1.66.0",
    ),
    OpenAiDefinition(
        module="openai.resources.responses",
        object="AsyncResponses",
        method="parse",
        type="chat",
        sync=False,
        min_version="1.66.0",
    ),
    OpenAiDefinition(
        module="openai.resources.embeddings",
        object="Embeddings",
        method="create",
        type="embedding",
        sync=True,
    ),
    OpenAiDefinition(
        module="openai.resources.embeddings",
        object="AsyncEmbeddings",
        method="create",
        type="embedding",
        sync=False,
    ),
]


_RESPONSES_PROMPT_FIELDS = ("tools", "tool_choice", "parallel_tool_calls")
_STRUCTURED_OUTPUT_METADATA_FIELDS = ("response_format", "text_format")


def _is_not_given(value: Any) -> bool:
    return isinstance(value, NotGiven)


def _get_attr_or_item(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(key, default)

    return getattr(value, key, default)


def _serialize_openai_value(value: Any) -> Any:
    """Convert OpenAI SDK request/response wrapper values into plain data."""

    if _is_not_given(value):
        return None

    if isclass(value) and issubclass(value, BaseModel):
        return value.model_json_schema()

    if isinstance(value, BaseModel):
        value.model_rebuild()

        try:
            return _serialize_openai_value(
                value.model_dump(mode="json", warnings=False)
            )
        except Exception:
            try:
                return _serialize_openai_value(
                    json.loads(value.model_dump_json(warnings=False))
                )
            except Exception:
                return _serialize_openai_value(value.model_dump(warnings=False))

    if isinstance(value, dict):
        return {
            key: _serialize_openai_value(val)
            for key, val in value.items()
            if not _is_not_given(val)
        }

    if isinstance(value, (list, tuple)):
        return [_serialize_openai_value(item) for item in value]

    try:
        return to_jsonable_python(value)
    except Exception:
        return str(value)


def _get_structured_output_metadata(metadata: Optional[Any], kwargs: Any) -> Any:
    structured_output_metadata = {}

    for key in _STRUCTURED_OUTPUT_METADATA_FIELDS:
        value = kwargs.get(key, None)

        if value is not None and not _is_not_given(value):
            structured_output_metadata[key] = _serialize_openai_value(value)

    if not structured_output_metadata:
        return _serialize_openai_value(metadata)

    metadata_dict = (
        _serialize_openai_value(metadata)
        if isinstance(metadata, BaseModel)
        else metadata
    )

    if metadata_dict is None:
        metadata_dict = {}

    if not isinstance(metadata_dict, dict):
        metadata_dict = {}

    return {**metadata_dict, **structured_output_metadata}


def _extract_response_api_completion(output: Any) -> Any:
    output = _serialize_openai_value(output)

    if not isinstance(output, list):
        return output

    if len(output) > 1:
        return output

    if len(output) == 1:
        return output[0]

    return None


class OpenAiArgsExtractor:
    def __init__(
        self,
        metadata: Optional[Any] = None,
        name: Optional[str] = None,
        langfuse_prompt: Optional[
            Any
        ] = None,  # we cannot use prompt because it's an argument of the old OpenAI completions API
        langfuse_public_key: Optional[str] = None,
        trace_id: Optional[str] = None,
        parent_observation_id: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        self.args = {}
        self.metadata = metadata
        self.args["metadata"] = _get_structured_output_metadata(metadata, kwargs)
        self.args["name"] = name
        self.args["langfuse_public_key"] = langfuse_public_key
        self.args["langfuse_prompt"] = langfuse_prompt
        self.args["trace_id"] = trace_id
        self.args["parent_observation_id"] = parent_observation_id

        self.kwargs = kwargs

    def get_langfuse_args(self) -> Any:
        return {**self.args, **self.kwargs}

    def get_openai_args(self) -> Any:
        openai_args = self.kwargs.copy()

        # If OpenAI model distillation is enabled, we need to add the metadata to the kwargs
        # https://platform.openai.com/docs/guides/distillation
        if openai_args.get("store", False):
            metadata = _serialize_openai_value(self.metadata)
            openai_args["metadata"] = metadata if isinstance(metadata, dict) else {}

        return openai_args


def _langfuse_wrapper(func: Any) -> Any:
    def _with_langfuse(open_ai_definitions: Any) -> Any:
        def wrapper(wrapped: Any, instance: Any, args: Any, kwargs: Any) -> Any:
            return func(open_ai_definitions, wrapped, args, kwargs)

        return wrapper

    return _with_langfuse


def _extract_responses_prompt(kwargs: Any) -> Any:
    input_value = kwargs.get("input", None)
    instructions = kwargs.get("instructions", None)
    prompt_fields = {}

    for key in _RESPONSES_PROMPT_FIELDS:
        value = kwargs.get(key, None)

        if value is not None and not isinstance(value, NotGiven):
            prompt_fields[key] = _serialize_openai_value(value)

    if isinstance(input_value, NotGiven):
        input_value = None

    if isinstance(instructions, NotGiven):
        instructions = None

    if instructions is None:
        prompt = input_value
    elif input_value is None:
        prompt = {"instructions": instructions}
    elif isinstance(input_value, str):
        prompt = [
            {"role": "system", "content": instructions},
            {"role": "user", "content": input_value},
        ]
    elif isinstance(input_value, list):
        prompt = [{"role": "system", "content": instructions}, *input_value]
    else:
        prompt = {"instructions": instructions, "input": input_value}

    if not prompt_fields:
        return prompt

    if isinstance(prompt, dict) and set(prompt.keys()) <= {"instructions", "input"}:
        return {**prompt, **prompt_fields}

    if prompt is not None:
        return {"input": prompt, **prompt_fields}

    return prompt_fields


def _extract_chat_prompt(kwargs: Any) -> Any:
    """Extracts the user input from prompts. Returns an array of messages or dict with messages and functions"""
    prompt = {}

    if kwargs.get("functions") is not None:
        prompt.update({"functions": kwargs["functions"]})

    if kwargs.get("function_call") is not None:
        prompt.update({"function_call": kwargs["function_call"]})

    if kwargs.get("tools") is not None:
        prompt.update({"tools": kwargs["tools"]})

    if prompt:
        # uf user provided functions, we need to send these together with messages to langfuse
        prompt.update(
            {
                "messages": [
                    _process_message(message) for message in kwargs.get("messages", [])
                ],
            }
        )
        return prompt
    else:
        # vanilla case, only send messages in openai format to langfuse
        return [_process_message(message) for message in kwargs.get("messages", [])]


def _process_message(message: Any) -> Any:
    if not isinstance(message, dict):
        return message

    processed_message = {**message}

    content = processed_message.get("content", None)
    if not isinstance(content, list):
        return processed_message

    processed_content = []

    for content_part in content:
        if content_part.get("type") == "input_audio":
            audio_base64 = content_part.get("input_audio", {}).get("data", None)
            format = content_part.get("input_audio", {}).get("format", "wav")

            if audio_base64 is not None:
                base64_data_uri = f"data:audio/{format};base64,{audio_base64}"

                processed_content.append(
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": LangfuseMedia(base64_data_uri=base64_data_uri),
                            "format": format,
                        },
                    }
                )
        else:
            processed_content.append(content_part)

    processed_message["content"] = processed_content

    return processed_message


def _extract_chat_response(kwargs: Any) -> Any:
    """Extracts the llm output from the response."""
    response = {
        "role": kwargs.get("role", None),
    }

    audio = None

    if kwargs.get("function_call") is not None:
        response.update({"function_call": kwargs["function_call"]})

    if kwargs.get("tool_calls") is not None:
        response.update({"tool_calls": kwargs["tool_calls"]})

    if kwargs.get("audio") is not None:
        audio = kwargs["audio"].__dict__

        if "data" in audio and audio["data"] is not None:
            base64_data_uri = f"data:audio/{audio.get('format', 'wav')};base64,{audio.get('data', None)}"
            audio["data"] = LangfuseMedia(base64_data_uri=base64_data_uri)

    response.update(
        {
            "content": kwargs.get("content", None),
        }
    )

    if audio is not None:
        response.update({"audio": audio})

    return response


def _get_langfuse_data_from_kwargs(resource: OpenAiDefinition, kwargs: Any) -> Any:
    default_name = (
        "OpenAI-embedding" if resource.type == "embedding" else "OpenAI-generation"
    )
    name = kwargs.get("name", default_name)

    if name is None:
        name = default_name

    if name is not None and not isinstance(name, str):
        raise TypeError("name must be a string")

    langfuse_public_key = kwargs.get("langfuse_public_key", None)
    if langfuse_public_key is not None and not isinstance(langfuse_public_key, str):
        raise TypeError("langfuse_public_key must be a string")

    trace_id = kwargs.get("trace_id", None)
    if trace_id is not None and not isinstance(trace_id, str):
        raise TypeError("trace_id must be a string")

    session_id = kwargs.get("session_id", None)
    if session_id is not None and not isinstance(session_id, str):
        raise TypeError("session_id must be a string")

    user_id = kwargs.get("user_id", None)
    if user_id is not None and not isinstance(user_id, str):
        raise TypeError("user_id must be a string")

    tags = kwargs.get("tags", None)
    if tags is not None and (
        not isinstance(tags, list) or not all(isinstance(tag, str) for tag in tags)
    ):
        raise TypeError("tags must be a list of strings")

    parent_observation_id = kwargs.get("parent_observation_id", None)
    if parent_observation_id is not None and not isinstance(parent_observation_id, str):
        raise TypeError("parent_observation_id must be a string")

    if parent_observation_id is not None and trace_id is None:
        raise ValueError("parent_observation_id requires trace_id to be set")

    metadata = kwargs.get("metadata", {})
    if (
        metadata is not None
        and not isinstance(metadata, NotGiven)
        and not isinstance(metadata, Omit)
        and not isinstance(metadata, dict)
    ):
        if isinstance(metadata, BaseModel):
            metadata = _serialize_openai_value(metadata)
        else:
            metadata = {}

    model = kwargs.get("model", None) or None

    prompt = None

    if resource.type == "completion":
        prompt = kwargs.get("prompt", None)
    elif resource.object == "Responses" or resource.object == "AsyncResponses":
        prompt = _extract_responses_prompt(kwargs)
    elif resource.type == "chat":
        prompt = _extract_chat_prompt(kwargs)
    elif resource.type == "embedding":
        prompt = kwargs.get("input", None)

    parsed_temperature = (
        kwargs.get("temperature", 1)
        if not isinstance(kwargs.get("temperature", 1), NotGiven)
        else 1
    )

    parsed_max_tokens = (
        kwargs.get("max_tokens", float("inf"))
        if not isinstance(kwargs.get("max_tokens", float("inf")), NotGiven)
        else float("inf")
    )

    parsed_max_completion_tokens = (
        kwargs.get("max_completion_tokens", None)
        if not isinstance(kwargs.get("max_completion_tokens", float("inf")), NotGiven)
        else None
    )

    parsed_top_p = (
        kwargs.get("top_p", 1)
        if not isinstance(kwargs.get("top_p", 1), NotGiven)
        else 1
    )

    parsed_frequency_penalty = (
        kwargs.get("frequency_penalty", 0)
        if not isinstance(kwargs.get("frequency_penalty", 0), NotGiven)
        else 0
    )

    parsed_presence_penalty = (
        kwargs.get("presence_penalty", 0)
        if not isinstance(kwargs.get("presence_penalty", 0), NotGiven)
        else 0
    )

    parsed_seed = (
        kwargs.get("seed", None)
        if not isinstance(kwargs.get("seed", None), NotGiven)
        else None
    )

    parsed_n = kwargs.get("n", 1) if not isinstance(kwargs.get("n", 1), NotGiven) else 1

    parsed_service_tier = (
        kwargs.get("service_tier", None)
        if not isinstance(kwargs.get("service_tier", None), NotGiven)
        else None
    )

    if resource.type == "embedding":
        parsed_dimensions = (
            kwargs.get("dimensions", None)
            if not isinstance(kwargs.get("dimensions", None), NotGiven)
            else None
        )
        parsed_encoding_format = (
            kwargs.get("encoding_format", "float")
            if not isinstance(kwargs.get("encoding_format", "float"), NotGiven)
            else "float"
        )

        modelParameters = {}
        if parsed_dimensions is not None:
            modelParameters["dimensions"] = parsed_dimensions
        if parsed_encoding_format != "float":
            modelParameters["encoding_format"] = parsed_encoding_format
    else:
        modelParameters = {
            "temperature": parsed_temperature,
            "max_tokens": parsed_max_tokens,
            "top_p": parsed_top_p,
            "frequency_penalty": parsed_frequency_penalty,
            "presence_penalty": parsed_presence_penalty,
        }

        if parsed_max_completion_tokens is not None:
            modelParameters.pop("max_tokens", None)
            modelParameters["max_completion_tokens"] = parsed_max_completion_tokens

        if parsed_n is not None and isinstance(parsed_n, int) and parsed_n > 1:
            modelParameters["n"] = parsed_n

        if parsed_seed is not None:
            modelParameters["seed"] = parsed_seed

        if parsed_service_tier is not None:
            modelParameters["service_tier"] = parsed_service_tier

    langfuse_prompt = kwargs.get("langfuse_prompt", None)

    return {
        "name": name,
        "metadata": metadata,
        "langfuse_public_key": langfuse_public_key,
        "trace_id": trace_id,
        "parent_observation_id": parent_observation_id,
        "user_id": user_id,
        "input": prompt,
        "model_parameters": modelParameters,
        "model": model or None,
        "prompt": langfuse_prompt,
    }


def _create_langfuse_update(
    completion: Any,
    generation: LangfuseGeneration,
    completion_start_time: Any,
    model: Optional[str] = None,
    usage: Optional[Any] = None,
    metadata: Optional[Any] = None,
    model_parameters: Optional[Any] = None,
) -> Any:
    update = {
        "output": completion,
        "completion_start_time": completion_start_time,
    }
    if model is not None:
        update["model"] = model

    if metadata is not None:
        update["metadata"] = metadata

    if model_parameters is not None:
        update["model_parameters"] = model_parameters

    if usage is not None:
        update["usage_details"] = _parse_usage(usage)
        update["cost_details"] = _parse_cost(usage)

    generation.update(**update)


def _parse_usage(usage: Optional[Any] = None) -> Any:
    if usage is None:
        return

    usage_dict = usage.copy() if isinstance(usage, dict) else usage.__dict__.copy()

    for tokens_details in [
        "prompt_tokens_details",
        "completion_tokens_details",
        "input_tokens_details",
        "output_tokens_details",
    ]:
        if tokens_details in usage_dict and usage_dict[tokens_details] is not None:
            tokens_details_dict = (
                usage_dict[tokens_details]
                if isinstance(usage_dict[tokens_details], dict)
                else usage_dict[tokens_details].__dict__
            )
            usage_dict[tokens_details] = {
                k: v for k, v in tokens_details_dict.items() if v is not None
            }

    if (
        len(usage_dict) == 2
        and "prompt_tokens" in usage_dict
        and "total_tokens" in usage_dict
    ):
        # handle embedding usage
        return {"input": usage_dict["prompt_tokens"]}

    return usage_dict


def _parse_cost(usage: Optional[Any] = None) -> Any:
    if usage is None:
        return

    # OpenRouter is returning total cost of the invocation
    # https://openrouter.ai/docs/use-cases/usage-accounting#cost-breakdown
    if hasattr(usage, "cost") and isinstance(getattr(usage, "cost"), float):
        return {"total": getattr(usage, "cost")}

    return None


def _extract_streamed_response_api_response(chunks: Any) -> Any:
    completion, model, usage, service_tier = None, None, None, None
    metadata = {}

    for raw_chunk in chunks:
        chunk = raw_chunk.__dict__
        if raw_response := chunk.get("response", None):
            usage = chunk.get("usage", None) or getattr(raw_response, "usage", None)

            response = raw_response.__dict__
            model = response.get("model")
            service_tier = response.get("service_tier", None) or service_tier

            for key, val in response.items():
                if key not in ["created_at", "model", "output", "usage", "text"]:
                    metadata[key] = val

                if key == "output":
                    completion = _extract_response_api_completion(val)

    return (model, completion, usage, metadata, service_tier)


def _extract_streamed_openai_response(resource: Any, chunks: Any) -> Any:
    completion: Any = defaultdict(lambda: None) if resource.type == "chat" else ""
    model, usage, finish_reason, service_tier = None, None, None, None

    for chunk in chunks:
        if _is_openai_v1():
            chunk = chunk.__dict__

        model = model or chunk.get("model", None) or None
        service_tier = service_tier or chunk.get("service_tier", None) or None
        chunk_usage = chunk.get("usage", None)
        if chunk_usage is not None:
            usage = chunk_usage

        choices = chunk.get("choices") or []

        for choice in choices:
            if _is_openai_v1():
                choice = choice.__dict__
            if resource.type == "chat":
                delta = choice.get("delta", None)
                choice_finish_reason = choice.get("finish_reason", None)
                if choice_finish_reason is not None:
                    finish_reason = choice_finish_reason

                if _is_openai_v1() and delta is not None:
                    delta = delta.__dict__

                if delta is None:
                    delta = {}

                if delta.get("role", None) is not None:
                    completion["role"] = delta["role"]

                if delta.get("content", None) is not None:
                    completion["content"] = (
                        delta.get("content", None)
                        if completion["content"] is None
                        else completion["content"] + delta.get("content", None)
                    )

                if delta.get("function_call", None) is not None:
                    curr = completion["function_call"]
                    tool_call_chunk = delta.get("function_call", None)

                    if not curr:
                        completion["function_call"] = {
                            "name": getattr(tool_call_chunk, "name", ""),
                            "arguments": getattr(tool_call_chunk, "arguments", ""),
                        }

                    else:
                        curr["name"] = curr["name"] or getattr(
                            tool_call_chunk, "name", None
                        )
                        curr["arguments"] += getattr(tool_call_chunk, "arguments", "")

                if (
                    delta.get("tool_calls", None) is not None
                    and len(delta.get("tool_calls")) > 0
                ):
                    curr = completion["tool_calls"]

                    if not curr:
                        completion["tool_calls"] = []
                        curr = completion["tool_calls"]

                    for raw_tool_call in delta.get("tool_calls", []):
                        index = _get_attr_or_item(raw_tool_call, "index", None)

                        if not isinstance(index, int):
                            index = len(curr) - 1 if curr else 0

                        while len(curr) <= index:
                            curr.append({"function": {"name": "", "arguments": ""}})

                        current_tool_call = curr[index]
                        tool_call_id = _get_attr_or_item(raw_tool_call, "id", None)
                        tool_call_type = _get_attr_or_item(raw_tool_call, "type", None)
                        tool_call_chunk = _get_attr_or_item(
                            raw_tool_call, "function", None
                        )

                        if tool_call_id is not None:
                            current_tool_call["id"] = tool_call_id

                        if tool_call_type is not None:
                            current_tool_call["type"] = tool_call_type

                        if tool_call_chunk is None:
                            continue

                        function_call = current_tool_call.setdefault("function", {})
                        tool_name = _get_attr_or_item(tool_call_chunk, "name", None)
                        tool_arguments = _get_attr_or_item(
                            tool_call_chunk, "arguments", None
                        )

                        if tool_name is not None:
                            function_call["name"] = (
                                function_call.get("name") or tool_name
                            )

                        if tool_arguments is not None:
                            function_call["arguments"] = (
                                function_call.get("arguments") or ""
                            ) + tool_arguments

            if resource.type == "completion":
                completion += choice.get("text", "")

    def get_response_for_chat() -> Any:
        content = completion["content"]

        if completion["tool_calls"]:
            response = {
                "role": "assistant",
                "tool_calls": completion["tool_calls"],
            }

            if content is not None:
                response["content"] = content

            return response

        if completion["function_call"]:
            response = {
                "role": "assistant",
                "function_call": completion["function_call"],
            }

            if content is not None:
                response["content"] = content

            return response

        return content or None

    return (
        model,
        get_response_for_chat() if resource.type == "chat" else completion,
        usage,
        {"finish_reason": finish_reason} if finish_reason is not None else None,
        service_tier,
    )


def _get_langfuse_data_from_default_response(
    resource: OpenAiDefinition, response: Any
) -> Any:
    if response is None:
        return None, "<NoneType response returned from OpenAI>", None, None

    model = response.get("model", None) or None
    service_tier = response.get("service_tier", None) or None

    completion = None

    if resource.type == "completion":
        choices = response.get("choices") or []
        if len(choices) > 0:
            choice = choices[-1]

            completion = choice.text if _is_openai_v1() else choice.get("text", None)

    elif resource.object == "Responses" or resource.object == "AsyncResponses":
        completion = _extract_response_api_completion(response.get("output", {}))

    elif resource.type == "chat":
        choices = response.get("choices") or []
        if len(choices) > 0:
            # If multiple choices were generated, we'll show all of them in the UI as a list.
            if len(choices) > 1:
                completion = [
                    _extract_chat_response(choice.message.__dict__)
                    if _is_openai_v1()
                    else choice.get("message", None)
                    for choice in choices
                ]
            else:
                choice = choices[0]
                completion = (
                    _extract_chat_response(choice.message.__dict__)
                    if _is_openai_v1()
                    else choice.get("message", None)
                )

    elif resource.type == "embedding":
        data = response.get("data") or []
        if len(data) > 0:
            first_embedding = data[0]
            embedding_vector = (
                first_embedding.embedding
                if hasattr(first_embedding, "embedding")
                else first_embedding.get("embedding", [])
            )
            completion = {
                "dimensions": len(embedding_vector) if embedding_vector else 0,
                "count": len(data),
            }

    usage = _parse_usage(response.get("usage", None))

    return (model, completion, usage, service_tier)


def _merge_service_tier_into_model_parameters(
    model_parameters: Optional[Any], service_tier: Optional[Any]
) -> Optional[Any]:
    """Merge the response-side service tier into the request-side model parameters.

    The response value is authoritative because OpenAI returns the tier that
    actually processed the request (e.g. when the request specified "auto").
    Returns None when there is nothing to update so callers can skip the
    update and keep the request-side model parameters untouched.
    """
    if service_tier is None:
        return None

    return {**(model_parameters or {}), "service_tier": service_tier}


def _is_openai_v1() -> bool:
    return Version(openai.__version__) >= Version("1.0.0")


def _is_streaming_response(response: Any) -> bool:
    return (
        isinstance(response, types.GeneratorType)
        or isinstance(response, types.AsyncGeneratorType)
        or (_is_openai_v1() and isinstance(response, openai.Stream))
        or (_is_openai_v1() and isinstance(response, openai.AsyncStream))
    )


_openai_stream_iter_hook_installed = False


def _install_openai_stream_iteration_hooks() -> None:
    global _openai_stream_iter_hook_installed

    if not _is_openai_v1():
        return

    if not _openai_stream_iter_hook_installed:
        original_iter = openai.Stream.__iter__
        original_aiter = openai.AsyncStream.__aiter__

        def traced_iter(self: Any) -> Any:
            try:
                yield from original_iter(self)
            finally:
                finalize_once = getattr(self, "_langfuse_finalize_once", None)
                if finalize_once is not None:
                    finalize_once()

        async def traced_aiter(self: Any) -> Any:
            try:
                async for item in original_aiter(self):
                    yield item
            finally:
                finalize_once = getattr(self, "_langfuse_finalize_once", None)
                if finalize_once is not None:
                    await finalize_once()

        setattr(openai.Stream, "__iter__", traced_iter)
        setattr(openai.AsyncStream, "__aiter__", traced_aiter)
        _openai_stream_iter_hook_installed = True


def _finalize_stream_response(
    *,
    resource: OpenAiDefinition,
    items: list[Any],
    generation: LangfuseGeneration,
    completion_start_time: Optional[datetime],
    model_parameters: Optional[Any] = None,
) -> None:
    try:
        model, completion, usage, metadata, service_tier = (
            _extract_streamed_response_api_response(items)
            if resource.object == "Responses" or resource.object == "AsyncResponses"
            else _extract_streamed_openai_response(resource, items)
        )

        _create_langfuse_update(
            completion,
            generation,
            completion_start_time,
            model=model,
            usage=usage,
            metadata=metadata,
            model_parameters=_merge_service_tier_into_model_parameters(
                model_parameters, service_tier
            ),
        )
    except Exception:
        pass
    finally:
        generation.end()


def _instrument_openai_stream(
    *,
    resource: OpenAiDefinition,
    response: Any,
    generation: LangfuseGeneration,
    model_parameters: Optional[Any] = None,
) -> Any:
    if not hasattr(response, "_iterator"):
        return LangfuseResponseGeneratorSync(
            resource=resource,
            response=response,
            generation=generation,
            model_parameters=model_parameters,
        )

    items: list[Any] = []
    raw_iterator = response._iterator
    completion_start_time: Optional[datetime] = None
    is_finalized = False
    close = response.close

    def finalize_once() -> None:
        nonlocal is_finalized
        if is_finalized:
            return

        is_finalized = True
        _finalize_stream_response(
            resource=resource,
            items=items,
            generation=generation,
            completion_start_time=completion_start_time,
            model_parameters=model_parameters,
        )

    response._langfuse_finalize_once = finalize_once  # type: ignore[attr-defined]

    def traced_iterator() -> Any:
        nonlocal completion_start_time
        try:
            for item in raw_iterator:
                items.append(item)

                if completion_start_time is None:
                    completion_start_time = _get_timestamp()

                yield item
        finally:
            finalize_once()

    def traced_close() -> Any:
        try:
            return close()
        finally:
            finalize_once()

    response._iterator = traced_iterator()
    response.close = traced_close

    return response


def _instrument_openai_async_stream(
    *,
    resource: OpenAiDefinition,
    response: Any,
    generation: LangfuseGeneration,
    model_parameters: Optional[Any] = None,
) -> Any:
    if not hasattr(response, "_iterator"):
        return LangfuseResponseGeneratorAsync(
            resource=resource,
            response=response,
            generation=generation,
            model_parameters=model_parameters,
        )

    items: list[Any] = []
    raw_iterator = response._iterator
    completion_start_time: Optional[datetime] = None
    is_finalized = False
    close = response.close

    async def finalize_once() -> None:
        nonlocal is_finalized
        if is_finalized:
            return

        is_finalized = True
        _finalize_stream_response(
            resource=resource,
            items=items,
            generation=generation,
            completion_start_time=completion_start_time,
            model_parameters=model_parameters,
        )

    response._langfuse_finalize_once = finalize_once  # type: ignore[attr-defined]

    async def traced_iterator() -> Any:
        nonlocal completion_start_time
        try:
            async for item in raw_iterator:
                items.append(item)

                if completion_start_time is None:
                    completion_start_time = _get_timestamp()

                yield item
        finally:
            await finalize_once()

    async def traced_close() -> Any:
        try:
            return await close()
        finally:
            await finalize_once()

    async def traced_aclose() -> Any:
        return await traced_close()

    response._iterator = traced_iterator()
    response.close = traced_close
    response.aclose = traced_aclose

    return response


def _get_raw_response_mode(kwargs: Any) -> Optional[str]:
    """Return the value of the OpenAI SDK's internal raw-response sentinel header.

    The SDK's `.with_raw_response` wrapper sets it to "true" and
    `.with_streaming_response` sets it to "stream" before invoking the same
    resource method that Langfuse instruments. Returns None for regular calls.
    """
    extra_headers = kwargs.get("extra_headers", None)

    if extra_headers is None or isinstance(extra_headers, NotGiven):
        return None

    try:
        return cast(Optional[str], extra_headers.get(RAW_RESPONSE_HEADER, None))
    except AttributeError:
        return None


def _should_skip_raw_response_instrumentation(kwargs: Any) -> bool:
    raw_response_mode = _get_raw_response_mode(kwargs)

    if raw_response_mode is None:
        return False

    if os.environ.get(LANGFUSE_OPENAI_SKIP_RAW_RESPONSES, "False").lower() in (
        "true",
        "1",
    ):
        return True

    # Raw streaming responses cannot be instrumented without consuming the
    # caller's stream or raw body, so they are always passed through untraced.
    return raw_response_mode == "stream" or kwargs.get("stream", False) is True


def _unwrap_raw_response(openai_response: Any) -> Any:
    """Return the parsed model for raw API responses so data extraction works.

    Libraries wrapping the OpenAI SDK (e.g. LiteLLM) call it via
    `.with_raw_response`, in which case the instrumented method returns a raw
    response object instead of the parsed model. `.parse()` caches its result
    on the response, so callers parsing later are unaffected.
    """
    if openai_response is None:
        return openai_response

    try:
        from openai._legacy_response import LegacyAPIResponse
        from openai._response import APIResponse

        if isinstance(openai_response, (LegacyAPIResponse, APIResponse)):
            return openai_response.parse()
    except Exception as e:
        logger.debug(f"Failed to parse raw OpenAI response for tracing: {e}")

    return openai_response


@_langfuse_wrapper
def _wrap(
    open_ai_resource: OpenAiDefinition, wrapped: Any, args: Any, kwargs: Any
) -> Any:
    arg_extractor = OpenAiArgsExtractor(*args, **kwargs)

    if _should_skip_raw_response_instrumentation(kwargs):
        return wrapped(**arg_extractor.get_openai_args())

    langfuse_args = arg_extractor.get_langfuse_args()

    langfuse_data = _get_langfuse_data_from_kwargs(open_ai_resource, langfuse_args)
    langfuse_client = get_client(public_key=langfuse_args["langfuse_public_key"])

    observation_type = (
        "embedding" if open_ai_resource.type == "embedding" else "generation"
    )

    generation = langfuse_client.start_observation(
        as_type=observation_type,  # type: ignore
        name=langfuse_data["name"],
        input=langfuse_data.get("input", None),
        metadata=langfuse_data.get("metadata", None),
        model_parameters=langfuse_data.get("model_parameters", None),
        trace_context={
            "trace_id": cast(str, langfuse_data.get("trace_id", None)),
            "parent_span_id": cast(
                str, langfuse_data.get("parent_observation_id", None)
            ),
        },
        model=langfuse_data.get("model", None),
        prompt=langfuse_data.get("prompt", None),
    )

    try:
        openai_response = wrapped(**arg_extractor.get_openai_args())

        if _is_openai_v1() and isinstance(openai_response, openai.Stream):
            return _instrument_openai_stream(
                resource=open_ai_resource,
                response=openai_response,
                generation=generation,
                model_parameters=langfuse_data.get("model_parameters", None),
            )
        elif _is_streaming_response(openai_response):
            return LangfuseResponseGeneratorSync(
                resource=open_ai_resource,
                response=openai_response,
                generation=generation,
                model_parameters=langfuse_data.get("model_parameters", None),
            )

        else:
            parsed_response = _unwrap_raw_response(openai_response)
            model, completion, usage, service_tier = (
                _get_langfuse_data_from_default_response(
                    open_ai_resource,
                    (parsed_response and parsed_response.__dict__)
                    if _is_openai_v1()
                    else parsed_response,
                )
            )

            generation.update(
                model=model,
                output=completion,
                usage_details=usage,
                cost_details=_parse_cost(parsed_response.usage)
                if hasattr(parsed_response, "usage")
                else None,
                model_parameters=_merge_service_tier_into_model_parameters(
                    langfuse_data.get("model_parameters", None), service_tier
                ),
            ).end()

        return openai_response
    except Exception as ex:
        logger.warning(ex)
        model = kwargs.get("model", None) or None
        generation.update(
            status_message=str(ex),
            level="ERROR",
            model=model,
            cost_details={"input": 0, "output": 0, "total": 0},
        ).end()

        raise ex


@_langfuse_wrapper
async def _wrap_async(
    open_ai_resource: OpenAiDefinition, wrapped: Any, args: Any, kwargs: Any
) -> Any:
    arg_extractor = OpenAiArgsExtractor(*args, **kwargs)

    if _should_skip_raw_response_instrumentation(kwargs):
        return await wrapped(**arg_extractor.get_openai_args())

    langfuse_args = arg_extractor.get_langfuse_args()

    langfuse_data = _get_langfuse_data_from_kwargs(open_ai_resource, langfuse_args)
    langfuse_client = get_client(public_key=langfuse_args["langfuse_public_key"])

    observation_type = (
        "embedding" if open_ai_resource.type == "embedding" else "generation"
    )

    generation = langfuse_client.start_observation(
        as_type=observation_type,  # type: ignore
        name=langfuse_data["name"],
        input=langfuse_data.get("input", None),
        metadata=langfuse_data.get("metadata", None),
        trace_context={
            "trace_id": cast(str, langfuse_data.get("trace_id", None)),
            "parent_span_id": cast(
                str, langfuse_data.get("parent_observation_id", None)
            ),
        },
        model_parameters=langfuse_data.get("model_parameters", None),
        model=langfuse_data.get("model", None),
        prompt=langfuse_data.get("prompt", None),
    )

    try:
        openai_response = await wrapped(**arg_extractor.get_openai_args())

        if _is_openai_v1() and isinstance(openai_response, openai.AsyncStream):
            return _instrument_openai_async_stream(
                resource=open_ai_resource,
                response=openai_response,
                generation=generation,
                model_parameters=langfuse_data.get("model_parameters", None),
            )
        elif _is_streaming_response(openai_response):
            return LangfuseResponseGeneratorAsync(
                resource=open_ai_resource,
                response=openai_response,
                generation=generation,
                model_parameters=langfuse_data.get("model_parameters", None),
            )

        else:
            parsed_response = _unwrap_raw_response(openai_response)
            model, completion, usage, service_tier = (
                _get_langfuse_data_from_default_response(
                    open_ai_resource,
                    (parsed_response and parsed_response.__dict__)
                    if _is_openai_v1()
                    else parsed_response,
                )
            )
            generation.update(
                model=model,
                output=completion,
                usage=usage,  # backward compat for all V2 self hosters
                usage_details=usage,
                cost_details=_parse_cost(parsed_response.usage)
                if hasattr(parsed_response, "usage")
                else None,
                model_parameters=_merge_service_tier_into_model_parameters(
                    langfuse_data.get("model_parameters", None), service_tier
                ),
            ).end()

        return openai_response
    except Exception as ex:
        logger.warning(ex)
        model = kwargs.get("model", None) or None
        generation.update(
            status_message=str(ex),
            level="ERROR",
            model=model,
            cost_details={"input": 0, "output": 0, "total": 0},
        ).end()

        raise ex


def register_tracing() -> None:
    resources = OPENAI_METHODS_V1 if _is_openai_v1() else OPENAI_METHODS_V0

    for resource in resources:
        if resource.min_version is not None and Version(openai.__version__) < Version(
            resource.min_version
        ):
            continue

        if resource.max_version is not None and Version(openai.__version__) >= Version(
            resource.max_version
        ):
            continue

        wrap_function_wrapper(
            resource.module,
            f"{resource.object}.{resource.method}",
            _wrap(resource) if resource.sync else _wrap_async(resource),
        )


register_tracing()
_install_openai_stream_iteration_hooks()


class LangfuseResponseGeneratorSync:
    def __init__(
        self,
        *,
        resource: Any,
        response: Any,
        generation: Any,
        model_parameters: Optional[Any] = None,
    ) -> None:
        self.items: list[Any] = []

        self.resource = resource
        self.response = response
        self.generation = generation
        self.model_parameters = model_parameters
        self.completion_start_time: Optional[datetime] = None
        self._is_finalized = False

    def __iter__(self) -> Any:
        try:
            for i in self.response:
                self.items.append(i)

                if self.completion_start_time is None:
                    self.completion_start_time = _get_timestamp()

                yield i
        finally:
            self._finalize()

    def __next__(self) -> Any:
        try:
            item = self.response.__next__()
            self.items.append(item)

            if self.completion_start_time is None:
                self.completion_start_time = _get_timestamp()

            return item

        except StopIteration:
            self._finalize()

            raise

    def __enter__(self) -> Any:
        return self.__iter__()

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()

    def close(self) -> None:
        close = getattr(self.response, "close", None)

        try:
            if callable(close):
                close()
        finally:
            self._finalize()

    def _finalize(self) -> None:
        if self._is_finalized:
            return

        self._is_finalized = True
        _finalize_stream_response(
            resource=self.resource,
            items=self.items,
            generation=self.generation,
            completion_start_time=self.completion_start_time,
            model_parameters=self.model_parameters,
        )


class LangfuseResponseGeneratorAsync:
    def __init__(
        self,
        *,
        resource: Any,
        response: Any,
        generation: Any,
        model_parameters: Optional[Any] = None,
    ) -> None:
        self.items: list[Any] = []

        self.resource = resource
        self.response = response
        self.generation = generation
        self.model_parameters = model_parameters
        self.completion_start_time: Optional[datetime] = None
        self._is_finalized = False

    async def __aiter__(self) -> Any:
        try:
            async for i in self.response:
                self.items.append(i)

                if self.completion_start_time is None:
                    self.completion_start_time = _get_timestamp()

                yield i
        finally:
            await self._finalize()

    async def __anext__(self) -> Any:
        try:
            item = await self.response.__anext__()
            self.items.append(item)

            if self.completion_start_time is None:
                self.completion_start_time = _get_timestamp()

            return item

        except StopAsyncIteration:
            await self._finalize()

            raise

    async def __aenter__(self) -> Any:
        return self.__aiter__()

    async def __aexit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        await self.aclose()

    async def _finalize(self) -> None:
        if self._is_finalized:
            return

        self._is_finalized = True
        _finalize_stream_response(
            resource=self.resource,
            items=self.items,
            generation=self.generation,
            completion_start_time=self.completion_start_time,
            model_parameters=self.model_parameters,
        )

    async def close(self) -> None:
        """Close the response and release the connection.

        Automatically called if the response body is read to completion.
        """
        close = getattr(self.response, "close", None)
        aclose = getattr(self.response, "aclose", None)

        try:
            if callable(close):
                result = close()
                if isawaitable(result):
                    await result
            elif callable(aclose):
                result = aclose()
                if isawaitable(result):
                    await result
        finally:
            await self._finalize()

    async def aclose(self) -> None:
        """Close the response and release the connection.

        Automatically called if the response body is read to completion.
        """
        aclose = getattr(self.response, "aclose", None)
        close = getattr(self.response, "close", None)

        try:
            if callable(aclose):
                result = aclose()
                if isawaitable(result):
                    await result
            elif callable(close):
                result = close()
                if isawaitable(result):
                    await result
        finally:
            await self._finalize()
