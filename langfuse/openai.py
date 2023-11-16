import threading
from datetime import datetime


from langfuse import Langfuse
from langfuse.client import InitialGeneration, CreateTrace

from distutils.version import StrictVersion
import openai
from wrapt import wrap_function_wrapper


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


def _with_tracer_wrapper(func):
    """Helper for providing tracer for wrapper functions."""

    def _with_tracer(open_ai_definitions, langfuse):
        def wrapper(wrapped, instance, args, kwargs):
            return func(open_ai_definitions, langfuse, wrapped, instance, args, kwargs)

        return wrapper

    return _with_tracer


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

    return InitialGeneration(name=name, metadata=metadata, trace_id=trace_id, start_time=start_time, prompt=prompt, modelParameters=modelParameters)


def _get_langfuse_data_from_response(resource: OpenAiDefinition, response):
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


@_with_tracer_wrapper
def _wrap(open_ai_resource: OpenAiDefinition, langfuse: Langfuse, wrapped, instance, args, kwargs):
    start_time = datetime.now()
    arg_extractor = OpenAiArgsExtractor(*args, **kwargs)

    generation = _get_langfuse_data_from_kwargs(open_ai_resource, langfuse, start_time, arg_extractor.get_langfuse_args())
    updated_generation = generation
    try:
        result = wrapped(**arg_extractor.get_openai_args())
        model, completion, usage = _get_langfuse_data_from_response(open_ai_resource, result.__dict__ if _is_openai_v1() else result)
        updated_generation = generation.copy(update={"model": model, "completion": completion, "end_time": datetime.now(), "usage": usage})
        langfuse.generation(updated_generation)
        return result
    except Exception as ex:
        model = kwargs.get("model", None)
        langfuse.generation(updated_generation.copy(update={"end_time": datetime.now(), "status_message": str(ex), "level": "ERROR", "model": model}))
        raise ex


class OpenAILangfuse:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super(OpenAILangfuse, cls).__new__(cls)
                    cls._instance.initialize()
        return cls._instance

    def initialize(self):
        self.langfuse = Langfuse()

    @classmethod
    def flush(cls):
        cls._instance.langfuse.flush()

    def register_tracing(self):
        resources = OPENAI_METHODS_V1 if _is_openai_v1() else OPENAI_METHODS_V0

        for resource in resources:
            wrap_function_wrapper(
                resource.module,
                f"{resource.object}.{resource.method}",
                _wrap(resource, self.langfuse),
            )

        setattr(openai, "flush_langfuse", self.flush)


modifier = OpenAILangfuse()
modifier.register_tracing()
