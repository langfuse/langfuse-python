import threading
import functools
from datetime import datetime
from dotenv import load_dotenv

import openai
from openai.api_resources import ChatCompletion, Completion

from langfuse import Langfuse
from langfuse.client import InitialGeneration
from langfuse.api.resources.commons.types.llm_usage import LlmUsage


load_dotenv()


class CreateArgsExtractor:
    def __init__(self, name=None, metadata=None, **kwargs):
        self.args = {}
        self.args["name"] = name
        self.args["metadata"] = metadata
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

    def _get_call_details(self, result, **kwargs):
        name = kwargs.get("name", "OpenAI-generation")

        if name is not None and not isinstance(name, str):
            raise TypeError("name must be a string")

        metadata = kwargs.get("metadata", {})

        if metadata is not None and not isinstance(metadata, dict):
            raise TypeError("metadata must be a dictionary")

        if result.object == "chat.completion":
            prompt = kwargs.get("messages", [{}])[-1].get("content", "")
            completion = result.choices[-1].message.content
        elif result.object == "text_completion":
            prompt = kwargs.get("prompt", "")
            completion = result.choices[-1].text
        else:
            completion = None
        model = result.model
        usage = None if result.usage is None else LlmUsage(**result.usage)
        endTime = datetime.now()
        modelParameters = {
            "temperature": kwargs.get("temperature", 1),
            "maxTokens": kwargs.get("max_tokens", float("inf")),
            "top_p": kwargs.get("top_p", 1),
            "frequency_penalty": kwargs.get("frequency_penalty", 0),
            "presence_penalty": kwargs.get("presence_penalty", 0),
        }
        all_details = {
            "name": name,
            "prompt": prompt,
            "completion": completion,
            "endTime": endTime,
            "model": model,
            "modelParameters": modelParameters,
            "usage": usage,
            "metadata": metadata,
        }
        return all_details

    def _log_result(self, result, call_details):
        generation = InitialGeneration(**call_details)
        self.langfuse.generation(generation)
        return result

    def langfuse_modified(self, func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                startTime = datetime.now()
                arg_extractor = CreateArgsExtractor(*args, **kwargs)
                result = func(**arg_extractor.get_openai_args())
                call_details = self._get_call_details(result, **arg_extractor.get_langfuse_args())
                call_details["startTime"] = startTime
            except Exception as ex:
                raise ex

            return self._log_result(result, call_details)

        return wrapper

    def replace_openai_funcs(self):
        api_resources_classes = [
            (ChatCompletion, "create"),
            (Completion, "create"),
        ]

        for api_resource_class, method in api_resources_classes:
            create_method = getattr(api_resource_class, method)
            setattr(api_resource_class, method, self.langfuse_modified(create_method))

        setattr(openai, "flush_langfuse", self.flush)


modifier = OpenAILangfuse()
modifier.replace_openai_funcs()
