import threading
from datetime import datetime
import types
from typing import Optional


from langfuse import Langfuse
from langfuse.client import InitialGeneration, CreateTrace, StatefulGenerationClient

from distutils.version import StrictVersion
import openai
from wrapt import wrap_function_wrapper

from langfuse.model import UpdateGeneration


class OpenAiDefinition:
    module: str
    object: str
    method: str
    type: str

    def __init__(self, module: str, object: str, method: str, type: str):
        self.module = module
        self.object = object
        self.method = method
        self.type = type


OPENAI_METHODS_V0 = [
    OpenAiDefinition(
        module="openai",
        object="ChatCompletion",
        method="create",
        type="chat",
    ),
    OpenAiDefinition(
        module="openai",
        object="Completion",
        method="create",
        type="completion",
    ),
]


OPENAI_METHODS_V1 = [
    OpenAiDefinition(
        module="openai.resources.chat.completions",
        object="Completions",
        method="create",
        type="chat",
    ),
    OpenAiDefinition(
        module="openai.resources.completions",
        object="Completions",
        method="create",
        type="completion",
    ),
]


class OpenAiArgsExtractor:
    def __init__(self, name=None, metadata=None, trace_id=None, **kwargs):
        self.args = {}
        self.args["name"] = name
        self.args["metadata"] = metadata
        self.args["trace_id"] = trace_id
        self.kwargs = kwargs

    def get_langfuse_args(self):
        return {**self.args, **self.kwargs}

    def get_openai_args(self):
        return self.kwargs


def _langfuse_wrapper(func):
    def _with_langfuse(open_ai_definitions, langfuse, initialize):
        def wrapper(wrapped, instance, args, kwargs):
            return func(open_ai_definitions, langfuse, initialize, wrapped, instance, args, kwargs)

        return wrapper

    return _with_langfuse


def _get_langfuse_data_from_kwargs(resource: OpenAiDefinition, langfuse: Langfuse, start_time, kwargs):
    name = kwargs.get("name", "OpenAI-generation")

    if name is not None and not isinstance(name, str):
        raise TypeError("name must be a string")

    trace_id = kwargs.get("trace_id", "OpenAI-generation")
    if trace_id is not None and not isinstance(trace_id, str):
        raise TypeError("trace_id must be a string")

    if trace_id:
        langfuse.trace(CreateTrace(id=trace_id))

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
                "messages": kwargs.get("messages", [{}]),
                "functions": kwargs.get("functions", [{}]),
                "function_call": kwargs.get("function_call", {}),
            }
            if kwargs.get("functions", None) is not None
            else kwargs.get("messages", [{}])
        )

    modelParameters = {
        "temperature": kwargs.get("temperature", 1),
        "maxTokens": kwargs.get("max_tokens", float("inf")),
        "top_p": kwargs.get("top_p", 1),
        "frequency_penalty": kwargs.get("frequency_penalty", 0),
        "presence_penalty": kwargs.get("presence_penalty", 0),
    }

    return InitialGeneration(name=name, metadata=metadata, trace_id=trace_id, start_time=start_time, prompt=prompt, modelParameters=modelParameters, model=model)


def _get_lagnfuse_data_from_streaming_response(resource: OpenAiDefinition, response, generation: StatefulGenerationClient, langfuse: Langfuse):
    final_response = [] if resource.type == "chat" else ""
    model = None
    completion_start_time = None
    for index, i in enumerate(response):
        print(index)
        if index == 0:
            completion_start_time = datetime.now()

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
                    final_response.append({"role": delta.get("role", None), "function_call": None, "tool_calls": None, "content": None})

                elif delta.get("content", None) is not None:
                    final_response[-1]["content"] = delta.get("content", None) if final_response[-1]["content"] is None else final_response[-1]["content"] + delta.get("content", None)

                elif delta.get("function_call", None) is not None:
                    final_response[-1]["function_call"] = (
                        delta.get("function_call", None) if final_response[-1]["function_call"] is None else final_response[-1]["function_call"] + delta.get("function_call", None)
                    )
                elif delta.get("tools_call", None) is not None:
                    final_response[-1]["tool_calls"] = delta.get("tools_call", None) if final_response[-1]["tool_calls"] is None else final_response[-1]["tool_calls"] + delta.get("tools_call", None)
            if resource.type == "completion":
                final_response += choice.get("text", None)

        yield i

    def get_response_for_chat():
        if len(final_response) > 0:
            if final_response[-1].get("content", None) is not None:
                return final_response[-1]["content"]
            elif final_response[-1].get("function_call", None) is not None:
                return final_response[-1]["function_call"]
            elif final_response[-1].get("tool_calls", None) is not None:
                return final_response[-1]["tool_calls"]
        return None

    update = UpdateGeneration(end_time=datetime.now(), completion=get_response_for_chat() if resource.type == "chat" else final_response, completion_start_time=completion_start_time)
    if model is not None:
        update = update.copy(update={"model": model})
    generation.update(update)


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
            completion = choice.message.content if _is_openai_v1() else choice.get("message", None).get("content", None)

    usage = response.get("usage", None)

    return model, completion, usage


def _is_openai_v1():
    return StrictVersion(openai.__version__) >= StrictVersion("1.0.0")


def _is_streaming_response(response):
    return isinstance(response, types.GeneratorType) or (_is_openai_v1() and isinstance(response, openai.Stream))


@_langfuse_wrapper
def _wrap(open_ai_resource: OpenAiDefinition, langfuse: Langfuse, initialize, wrapped, instance, args, kwargs):
    new_langfuse = initialize()

    start_time = datetime.now()
    arg_extractor = OpenAiArgsExtractor(*args, **kwargs)

    generation = _get_langfuse_data_from_kwargs(open_ai_resource, new_langfuse, start_time, arg_extractor.get_langfuse_args())
    generation = new_langfuse.generation(generation)
    try:
        openai_response = wrapped(**arg_extractor.get_openai_args())

        if _is_streaming_response(openai_response):
            return _get_lagnfuse_data_from_streaming_response(open_ai_resource, openai_response, generation, new_langfuse)

        else:
            model, completion, usage = _get_langfuse_data_from_default_response(open_ai_resource, openai_response.__dict__ if _is_openai_v1() else openai_response)
            generation.update(UpdateGeneration(model=model, completion=completion, end_time=datetime.now(), usage=usage))
        return openai_response
    except Exception as ex:
        model = kwargs.get("model", None)
        generation.update(UpdateGeneration(endTime=datetime.now(), statusMessage=str(ex), level="ERROR", model=model))
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
                    self._langfuse = Langfuse(public_key=openai.langfuse_public_key, secret_key=openai.langfuse_secret_key, host=openai.langfuse_host)
        return self._langfuse

    def flush(cls):
        cls._langfuse.flush()

    def register_tracing(self):
        resources = OPENAI_METHODS_V1 if _is_openai_v1() else OPENAI_METHODS_V0

        for resource in resources:
            wrap_function_wrapper(
                resource.module,
                f"{resource.object}.{resource.method}",
                _wrap(resource, self._langfuse, self.initialize),
            )

        setattr(openai, "langfuse_public_key", None)
        setattr(openai, "langfuse_secret_key", None)
        setattr(openai, "langfuse_host", None)

        setattr(openai, "flush_langfuse", self.flush)


modifier = OpenAILangfuse()
modifier.register_tracing()
