import os
import functools
from datetime import datetime
from dotenv import load_dotenv

import openai
from openai.api_resources import ChatCompletion, Completion

from langfuse import Langfuse
from langfuse.client import InitialGeneration
from langfuse.api.resources.commons.types.llm_usage import LlmUsage


load_dotenv()


class OpenAILangfuse:
    def __init__(self):
        self.langfuse = Langfuse(os.environ["LF_PK"], os.environ["LF_SK"], os.environ["HOST"])

    def _get_call_details(self, result, **kwargs):
        if result.object == "chat.completion":
            prompt = kwargs.get("messages", [{}])[-1].get("content", "")
            completion = result.choices[-1].message.content
        elif result.object == "text_completion":
            prompt = kwargs.get("prompt", "")
            completion = result.choices[-1].text
        else:
            completion = ""
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
            "prompt": prompt,
            "completion": completion,
            "endTime": endTime,
            "model": model,
            "modelParameters": modelParameters,
            "usage": usage,
        }
        return all_details

    def _log_result(self, result, call_details):
        generation = InitialGeneration(name="OpenAI-generation", **call_details)
        self.langfuse.generation(generation)
        self.langfuse.flush()
        return result

    def langfuse_modified(self, func):
        @functools.wraps(func)
        def wrapper(**kwargs):
            try:
                startTime = datetime.now()
                result = func(**kwargs)
                call_details = self._get_call_details(result, **kwargs)
                call_details["startTime"] = startTime
            except Exception as ex:
                raise ex

            return self._log_result(result, call_details)

        return wrapper

    def set_openai_funcs(self):
        api_resources_classes = [
            (ChatCompletion, "create"),
            (Completion, "create"),
        ]

        for api_resource_class, method in api_resources_classes:
            create_method = getattr(api_resource_class, method)
            setattr(api_resource_class, method, self.langfuse_modified(create_method))


modifier = OpenAILangfuse()
modifier.set_openai_funcs()
