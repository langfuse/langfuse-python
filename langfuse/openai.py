import threading
import functools
from datetime import datetime

import openai
from openai.api_resources import ChatCompletion, Completion

from langfuse import Langfuse
from langfuse.client import InitialGeneration, CreateTrace
from langfuse.api.resources.commons.types.llm_usage import LlmUsage


class CreateArgsExtractor:
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

    def _get_call_details(self, result, api_resource_class, **kwargs):
        name = kwargs.get("name", "OpenAI-generation")

        if name is not None and not isinstance(name, str):
            raise TypeError("name must be a string")

        trace_id = kwargs.get("trace_id", "OpenAI-generation")
        if trace_id is not None and not isinstance(trace_id, str):
            raise TypeError("trace_id must be a string")

        metadata = kwargs.get("metadata", {})

        if metadata is not None and not isinstance(metadata, dict):
            raise TypeError("metadata must be a dictionary")

        completion = None

        if api_resource_class == ChatCompletion:
            prompt = (
                {
                    "messages": kwargs.get("messages", [{}]),
                    "functions": kwargs.get("functions", [{}]),
                    "function_call": kwargs.get("function_call", {}),
                }
                if kwargs.get("functions", None) is not None
                else kwargs.get("messages", [{}])
            )
            if not isinstance(result, Exception):
                completion = result.choices[-1].message.content
                if completion is None:
                    completion = result.choices[-1].message.function_call

        elif api_resource_class == Completion:
            prompt = kwargs.get("prompt", "")
            if not isinstance(result, Exception):
                completion = result.choices[-1].text
        else:
            completion = None

        model = kwargs.get("model", None) if isinstance(result, Exception) else result.model

        usage = None if isinstance(result, Exception) or result.usage is None else LlmUsage(**result.usage)
        endTime = datetime.now()
        modelParameters = {
            "temperature": kwargs.get("temperature", 1),
            "maxTokens": kwargs.get("max_tokens", float("inf")),
            "top_p": kwargs.get("top_p", 1),
            "frequency_penalty": kwargs.get("frequency_penalty", 0),
            "presence_penalty": kwargs.get("presence_penalty", 0),
        }
        all_details = {
            "status_message": str(result) if isinstance(result, Exception) else None,
            "name": name,
            "prompt": prompt,
            "completion": completion,
            "endTime": endTime,
            "model": model,
            "modelParameters": modelParameters,
            "usage": usage,
            "metadata": metadata,
            "level": "ERROR" if isinstance(result, Exception) else "DEFAULT",
            "trace_id": trace_id,
        }
        return all_details

    def _log_result(self, call_details):
        generation = InitialGeneration(**call_details)
        if call_details["trace_id"] is not None:
            self.langfuse.trace(CreateTrace(id=call_details["trace_id"]))
        self.langfuse.generation(generation)

    def langfuse_modified(self, func, api_resource_class):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                startTime = datetime.now()
                arg_extractor = CreateArgsExtractor(*args, **kwargs)
                result = func(**arg_extractor.get_openai_args())
                call_details = self._get_call_details(result, api_resource_class, **arg_extractor.get_langfuse_args())
                call_details["startTime"] = startTime
                self._log_result(call_details)
            except Exception as ex:
                call_details = self._get_call_details(ex, api_resource_class, **arg_extractor.get_langfuse_args())
                call_details["startTime"] = startTime
                self._log_result(call_details)
                raise ex

            return result

        return wrapper

    def replace_openai_funcs(self):
        api_resources_classes = [
            (ChatCompletion, "create"),
            (Completion, "create"),
        ]

        for api_resource_class, method in api_resources_classes:
            create_method = getattr(api_resource_class, method)
            setattr(api_resource_class, method, self.langfuse_modified(create_method, api_resource_class))

        setattr(openai, "flush_langfuse", self.flush)


modifier = OpenAILangfuse()
modifier.replace_openai_funcs()
