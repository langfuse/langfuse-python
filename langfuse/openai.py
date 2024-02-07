import copy
import logging
import threading
import types
from typing import List, Optional

import openai
from openai import AsyncAzureOpenAI, AsyncOpenAI, AzureOpenAI, OpenAI  # noqa: F401
from packaging.version import Version
from wrapt import wrap_function_wrapper

from langfuse import Langfuse
from langfuse.client import StatefulGenerationClient
from langfuse.utils import _get_timestamp

log = logging.getLogger("langfuse")


class OpenAiDefinition:
    module: str
    object: str
    method: str
    type: str
    sync: bool

    def __init__(self, module: str, object: str, method: str, type: str, sync: bool):
        self.module = module
        self.object = object
        self.method = method
        self.type = type
        self.sync = sync


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
]


class OpenAiArgsExtractor:
    def __init__(
        self, name=None, metadata=None, trace_id=None, session_id=None, **kwargs
    ):
        self.args = {}
        self.args["name"] = name
        self.args["metadata"] = metadata
        self.args["trace_id"] = trace_id
        self.args["session_id"] = session_id
        self.kwargs = kwargs

    def get_langfuse_args(self):
        return {**self.args, **self.kwargs}

    def get_openai_args(self):
        return self.kwargs


def _langfuse_wrapper(func):
    def _with_langfuse(open_ai_definitions, initialize):
        def wrapper(wrapped, instance, args, kwargs):
            return func(open_ai_definitions, initialize, wrapped, args, kwargs)

        return wrapper

    return _with_langfuse


def _get_langfuse_data_from_kwargs(
    resource: OpenAiDefinition, langfuse: Langfuse, start_time, kwargs
):
    name = kwargs.get("name", "OpenAI-generation")

    if name is not None and not isinstance(name, str):
        raise TypeError("name must be a string")

    trace_id = kwargs.get("trace_id", None)
    if trace_id is not None and not isinstance(trace_id, str):
        raise TypeError("trace_id must be a string")

    session_id = kwargs.get("session_id", None)
    if session_id is not None and not isinstance(session_id, str):
        raise TypeError("session_id must be a string")

    if trace_id:
        langfuse.trace(id=trace_id, session_id=session_id)
    elif session_id:
        # If a session_id is provided but no trace_id, we should create a trace using the SDK and then use its trace_id
        trace_id = langfuse.trace(session_id=session_id).id

    metadata = kwargs.get("metadata", {})

    if metadata is not None and not isinstance(metadata, dict):
        raise TypeError("metadata must be a dictionary")

    model = kwargs.get("model", None)

    prompt = None

    if resource.type == "completion":
        prompt = kwargs.get("prompt", None)
    elif resource.type == "chat":
        prompt = (
            {
                "messages": filter_image_data(kwargs.get("messages", [])),
                "functions": kwargs.get("functions", []),
                "function_call": kwargs.get("function_call", {}),
            }
            if kwargs.get("functions", None) is not None
            else filter_image_data(kwargs.get("messages", []))
        )

    modelParameters = {
        "temperature": kwargs.get("temperature", 1),
        "max_tokens": kwargs.get("max_tokens", float("inf")),  # casing?
        "top_p": kwargs.get("top_p", 1),
        "frequency_penalty": kwargs.get("frequency_penalty", 0),
        "presence_penalty": kwargs.get("presence_penalty", 0),
    }

    return {
        "name": name,
        "metadata": metadata,
        "trace_id": trace_id,
        "start_time": start_time,
        "input": prompt,
        "model_parameters": modelParameters,
        "model": model,
    }


def _get_langfuse_data_from_sync_streaming_response(
    resource: OpenAiDefinition,
    response,
    generation: StatefulGenerationClient,
    langfuse: Langfuse,
):
    responses = []
    for i in response:
        responses.append(i)
        yield i

    model, completion_start_time, completion = _extract_data(resource, responses)

    _create_langfuse_update(completion, generation, completion_start_time, model=model)


async def _get_langfuse_data_from_async_streaming_response(
    resource: OpenAiDefinition,
    response,
    generation: StatefulGenerationClient,
    langfuse: Langfuse,
):
    responses = []
    async for i in response:
        responses.append(i)
        yield i

    model, completion_start_time, completion = _extract_data(resource, responses)

    _create_langfuse_update(completion, generation, completion_start_time, model=model)


def _create_langfuse_update(
    completion, generation: StatefulGenerationClient, completion_start_time, model=None
):
    update = {
        "end_time": _get_timestamp(),
        "output": completion,
        "completion_start_time": completion_start_time,
    }
    if model is not None:
        update["model"] = model

    generation.update(**update)


def _extract_data(resource, responses):
    completion = [] if resource.type == "chat" else ""
    model = None
    completion_start_time = None

    for index, i in enumerate(responses):
        if index == 0:
            completion_start_time = _get_timestamp()

        if _is_openai_v1():
            i = i.__dict__

        model = i.get("model", None) if model is None else model

        choices = i.get("choices", [])

        for choice in choices:
            if _is_openai_v1():
                choice = choice.__dict__
            if resource.type == "chat":
                delta = choice.get("delta", None)

                if _is_openai_v1():
                    delta = delta.__dict__

                if delta.get("role", None) is not None:
                    completion.append(
                        {
                            "role": delta.get("role", None),
                            "function_call": None,
                            "tool_calls": None,
                            "content": None,
                        }
                    )

                elif delta.get("content", None) is not None:
                    completion[-1]["content"] = (
                        delta.get("content", None)
                        if completion[-1]["content"] is None
                        else completion[-1]["content"] + delta.get("content", None)
                    )

                elif delta.get("function_call", None) is not None:
                    completion[-1]["function_call"] = (
                        delta.get("function_call", None)
                        if completion[-1]["function_call"] is None
                        else completion[-1]["function_call"]
                        + delta.get("function_call", None)
                    )
                elif delta.get("tools_call", None) is not None:
                    completion[-1]["tool_calls"] = (
                        delta.get("tools_call", None)
                        if completion[-1]["tool_calls"] is None
                        else completion[-1]["tool_calls"]
                        + delta.get("tools_call", None)
                    )
            if resource.type == "completion":
                completion += choice.get("text", None)

    def get_response_for_chat():
        if len(completion) > 0:
            if completion[-1].get("content", None) is not None:
                return completion[-1]["content"]
            elif completion[-1].get("function_call", None) is not None:
                return completion[-1]["function_call"]
            elif completion[-1].get("tool_calls", None) is not None:
                return completion[-1]["tool_calls"]
        return None

    return (
        model,
        completion_start_time,
        get_response_for_chat() if resource.type == "chat" else completion,
    )


def _get_langfuse_data_from_default_response(resource: OpenAiDefinition, response):
    model = response.get("model", None)

    completion = None
    if resource.type == "completion":
        choices = response.get("choices", [])
        if len(choices) > 0:
            choice = choices[-1]

            completion = choice.text if _is_openai_v1() else choice.get("text", None)
    elif resource.type == "chat":
        choices = response.get("choices", [])
        if len(choices) > 0:
            choice = choices[-1]
            completion = (
                choice.message.json()
                if _is_openai_v1()
                else choice.get("message", None)
            )

    usage = response.get("usage", None)

    return model, completion, usage.__dict__ if _is_openai_v1() else usage


def _is_openai_v1():
    return Version(openai.__version__) >= Version("1.0.0")


def _is_streaming_response(response):
    return (
        isinstance(response, types.GeneratorType)
        or isinstance(response, types.AsyncGeneratorType)
        or (_is_openai_v1() and isinstance(response, openai.Stream))
        or (_is_openai_v1() and isinstance(response, openai.AsyncStream))
    )


@_langfuse_wrapper
def _wrap(open_ai_resource: OpenAiDefinition, initialize, wrapped, args, kwargs):
    new_langfuse: Langfuse = initialize()

    start_time = _get_timestamp()
    arg_extractor = OpenAiArgsExtractor(*args, **kwargs)

    generation = _get_langfuse_data_from_kwargs(
        open_ai_resource, new_langfuse, start_time, arg_extractor.get_langfuse_args()
    )
    generation = new_langfuse.generation(**generation)
    try:
        openai_response = wrapped(**arg_extractor.get_openai_args())

        if _is_streaming_response(openai_response):
            return _get_langfuse_data_from_sync_streaming_response(
                open_ai_resource, openai_response, generation, new_langfuse
            )

        else:
            model, completion, usage = _get_langfuse_data_from_default_response(
                open_ai_resource,
                openai_response.__dict__ if _is_openai_v1() else openai_response,
            )
            generation.update(
                model=model, output=completion, end_time=_get_timestamp(), usage=usage
            )

        return openai_response
    except Exception as ex:
        log.warning(ex)
        model = kwargs.get("model", None)
        generation.update(
            end_time=_get_timestamp(),
            status_message=str(ex),
            level="ERROR",
            model=model,
        )
        raise ex


@_langfuse_wrapper
async def _wrap_async(
    open_ai_resource: OpenAiDefinition, initialize, wrapped, args, kwargs
):
    new_langfuse = initialize()
    start_time = _get_timestamp()
    arg_extractor = OpenAiArgsExtractor(*args, **kwargs)

    generation = _get_langfuse_data_from_kwargs(
        open_ai_resource, new_langfuse, start_time, arg_extractor.get_langfuse_args()
    )
    generation = new_langfuse.generation(**generation)
    try:
        openai_response = await wrapped(**arg_extractor.get_openai_args())

        if _is_streaming_response(openai_response):
            return _get_langfuse_data_from_async_streaming_response(
                open_ai_resource, openai_response, generation, new_langfuse
            )

        else:
            model, completion, usage = _get_langfuse_data_from_default_response(
                open_ai_resource,
                openai_response.__dict__ if _is_openai_v1() else openai_response,
            )
            generation.update(
                model=model,
                output=completion,
                end_time=_get_timestamp(),
                usage=usage,
            )
        return openai_response
    except Exception as ex:
        model = kwargs.get("model", None)
        generation.update(
            end_time=_get_timestamp(),
            status_message=str(ex),
            level="ERROR",
            model=model,
        )
        raise ex


class OpenAILangfuse:
    _instance = None
    _lock = threading.Lock()
    _langfuse: Optional[Langfuse] = None

    def __new__(cls):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super(OpenAILangfuse, cls).__new__(cls)
        return cls._instance

    def initialize(self):
        if not self._langfuse:
            with self._lock:
                if not self._langfuse:
                    self._langfuse = Langfuse(
                        public_key=openai.langfuse_public_key,
                        secret_key=openai.langfuse_secret_key,
                        host=openai.langfuse_host,
                        debug=openai.langfuse_debug,
                        sdk_integration="openai",
                    )
        return self._langfuse

    def flush(cls):
        cls._langfuse.flush()

    def register_tracing(self):
        resources = OPENAI_METHODS_V1 if _is_openai_v1() else OPENAI_METHODS_V0

        for resource in resources:
            wrap_function_wrapper(
                resource.module,
                f"{resource.object}.{resource.method}",
                _wrap(resource, self.initialize)
                if resource.sync
                else _wrap_async(resource, self.initialize),
            )

        setattr(openai, "langfuse_public_key", None)
        setattr(openai, "langfuse_secret_key", None)
        setattr(openai, "langfuse_host", None)
        setattr(openai, "langfuse_debug", None)
        setattr(openai, "flush_langfuse", self.flush)


modifier = OpenAILangfuse()
modifier.register_tracing()


def auth_check():
    if modifier._langfuse is None:
        modifier.initialize()

    return modifier._langfuse.auth_check()


def filter_image_data(messages: List[dict]):
    """
    https://platform.openai.com/docs/guides/vision?lang=python

    The messages array remains the same, but the 'image_url' is removed from the 'content' array.
    It should only be removed if the value starts with 'data:image/jpeg;base64,'

    """

    output_messages = copy.deepcopy(messages)

    for message in output_messages:
        if message.get("content", None) is not None:
            content = message["content"]
            for index, item in enumerate(content):
                if isinstance(item, dict) and item.get("image_url", None) is not None:
                    url = item["image_url"]["url"]
                    if url.startswith("data:image/"):
                        del content[index]["image_url"]

    return output_messages
