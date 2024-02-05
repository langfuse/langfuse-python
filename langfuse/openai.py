import copy
import logging
import threading
import types
from typing import List, Optional

import openai
import asyncio
from openai import AsyncAzureOpenAI, AsyncOpenAI, AzureOpenAI, OpenAI  # noqa: F401
from packaging.version import Version
from wrapt import wrap_function_wrapper

from langfuse import Langfuse
from langfuse.client import StatefulGenerationClient
from langfuse.utils import _get_timestamp, get_api

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
    OpenAiDefinition(
        module="openai.resources.beta.assistants",
        object="Assistants",
        method="create",
        type="assistant",
        sync=True,
    ),
    OpenAiDefinition(
        module="openai.resources.beta.threads",
        object="Threads",
        method="create",
        type="thread",
        sync=True,
    ),
    OpenAiDefinition(
        module="openai.resources.beta.threads.messages",
        object="Messages",
        method="create",
        type="message",
        sync=True,
    ),
    OpenAiDefinition(
        module="openai.resources.beta.threads.runs",
        object="Runs",
        method="create",
        type="run",
        sync=True,
    ),
    OpenAiDefinition(
        module="openai.resources.beta.threads.runs",
        object="Runs",
        method="retrieve",
        type="run",
        sync=True,
    ),
]


class OpenAiArgsExtractor:
    def __init__(
        self, name=None, metadata=None, trace_id=None, session_id=None, **kwargs
    ):
        self.args = {}
        self.args["name"] = name  # TODO: assistent name!!
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
    default_names = lambda resource: f"OpenAI-{resource.type}-{resource.method}"

    name = kwargs.get("name") or default_names(resource)

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

    prompt = None  # TODO: rename, not always prompt

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
    elif resource.type == "assistant":
        # based on https://platform.openai.com/docs/api-reference/assistants/createAssistant
        prompt = {
            "description": kwargs.get("description"),
            "instructions": kwargs.get("instructions", ""),
            "tools": kwargs.get("tools", []),
            "file_ids": kwargs.get("file_ids", []),
            # "metadata": kwargs.get("metadata"), # TODO: extract here?
        }
    elif resource.type == "message":
        # based on https://platform.openai.com/docs/api-reference/messages/createMessage
        prompt = {
            "thread_id": kwargs.get("thread_id"),
            "role": kwargs.get("role"),
            "content": kwargs.get("content"),
            "file_ids": kwargs.get("file_ids", []),
            # "metadata": kwargs.get("metadata"), # TODO: extract here?
        }
    elif resource.type == "run":
        prompt = {
            "thread_id": kwargs.get("thread_id"),
            "assistant_id": kwargs.get("assistant_id"),
            "model": kwargs.get("model"),
            "instructions": kwargs.get("instructions"),
            "additional_instructions": kwargs.get("additional_instructions"),
            "tools": kwargs.get("tools", []),
            "metadata": kwargs.get("metadata", {}),
        }

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
    elif resource.type == "assistant":
        # based on https://platform.openai.com/docs/api-reference/assistants/object
        completion = {
            "id": response.get("id"),
            "object": response.get("object"),
            "created_at": response.get("created_at"),
            "name": response.get("name"),
            "description": response.get("description"),
            "instructions": response.get("instructions"),
            "tools": response.get("tools", []),
            "file_ids": response.get("file_ids", []),
            "metadata": response.get("metadata", {}),
        }
    elif resource.type == "thread":
        # based on https://platform.openai.com/docs/api-reference/threads/object
        completion = {
            "id": response.get("id"),
            "object": response.get("object"),  # always "thread"
            "created_at": response.get("created_at"),
            "metadata": response.get("metadata", {}),
        }
    elif resource.type == "message":
        # based on https://platform.openai.com/docs/api-reference/messages/object
        completion = {
            "id": response.get("id"),
            "object": response.get("object"),
            "created_at": response.get("created_at"),
            "thread_id": response.get("thread_id"),
            "role": response.get("role"),
            "content": response.get("content", []),
            "assistant_id": response.get("assistant_id"),
            "run_id": response.get("run_id"),
            "file_ids": response.get("file_ids", []),
        }

    elif resource.type == "run":
        # based on https://platform.openai.com/docs/api-reference/messages/object
        completion = {**response}

    usage = response.get("usage", None)

    return (
        model,
        completion,
        usage.__dict__ if _is_openai_v1() and usage else usage,
    )


def _is_openai_v1():
    return Version(openai.__version__) >= Version("1.0.0")


def _is_streaming_response(response):
    return (
        isinstance(response, types.GeneratorType)
        or (_is_openai_v1() and isinstance(response, openai.Stream))
        or (_is_openai_v1() and isinstance(response, openai.AsyncStream))
    )


def _setup_run_create(langfuse: Langfuse, parsed_kwargs):
    generation = langfuse.generation(**parsed_kwargs)
    sub_span = langfuse.span(
        **{**parsed_kwargs, "name": "Run: queued"}, parent_observation_id=generation.id
    )  # overwrite name

    return generation, sub_span


def _setup_run_retrieve(langfuse: Langfuse, parsed_kwargs):
    # TODO: what in case no trace_id is given?
    api = get_api()
    trace_id = parsed_kwargs.get("trace_id")

    generations = api.observations.get_many(
        trace_id=trace_id, name="OpenAI-run-create", type="GENERATION"
    )

    # assume last generation is the one corresponding to "run create"
    if len(generations.data) > 0:
        generation = generations.data[-1]
        parent_observation_id = generation.id
    else:
        log.warning(
            f"No Generation found for trace_id {trace_id}. There should be one corresponding the Run create"
        )
        parent_observation_id = None

    event = langfuse.event(**parsed_kwargs, parent_observation_id=parent_observation_id)
    event.parent_observation_id = (
        parent_observation_id  # TODO: is this information somewhere else?
    )
    return event


def _post_run_retrieve(langfuse, openai_response, event, trace_id, output):
    # get spans of the parent observation
    api = get_api()

    spans = api.observations.get_many(
        trace_id=trace_id,
        type="SPAN",
        parent_observation_id=event.parent_observation_id,
    )

    run_status_spans = [
        span for span in spans.data if span.name.startswith("Run: ")
    ]  # TODO: use other fields
    run_status_spans.sort(key=lambda x: x.start_time)

    # again assume last one is the one corresponding to the run
    if run_status_spans:
        last_run_status_span = run_status_spans[-1]
        last_status = last_run_status_span.name.split(": ")[1]

        if last_status != openai_response.status:
            # TODO: end old status
            # last_run_status_span.update(
            #     end_time = _get_timestamp()
            # )
            status_span = langfuse.span(
                **{
                    "name": f"Run: {openai_response.status}",
                    "parent_observation_id": event.parent_observation_id,
                    "trace_id": trace_id,
                }
            )
        else:
            status_span = last_run_status_span

        event.update(parent_observation_id=status_span.id, output=output)


def _post_run_create(langfuse: Langfuse):
    pass


@_langfuse_wrapper
def _wrap(open_ai_resource: OpenAiDefinition, initialize, wrapped, args, kwargs):
    new_langfuse: Langfuse = initialize()

    start_time = _get_timestamp()
    arg_extractor = OpenAiArgsExtractor(*args, **kwargs)

    parsed_kwargs = _get_langfuse_data_from_kwargs(
        open_ai_resource, new_langfuse, start_time, arg_extractor.get_langfuse_args()
    )
    if open_ai_resource.type == "assistant":
        observation = new_langfuse.event(**parsed_kwargs)
    elif open_ai_resource.type == "thread":
        observation = new_langfuse.event(**parsed_kwargs)

    elif open_ai_resource.type == "run" and open_ai_resource.method == "retrieve":
        # get parent observation id
        observation = _setup_run_retrieve(new_langfuse, parsed_kwargs)
    elif open_ai_resource.type == "run" and open_ai_resource.method == "create":
        observation, sub_span = _setup_run_create(new_langfuse, parsed_kwargs)

    elif open_ai_resource.type == "message":
        # messages can only be created in the context of a thread, i.e. an existing trace
        if parsed_kwargs.get("trace_id", None) is None:
            # look if there exists a trace with the thread_id
            thread_id = parsed_kwargs.get("input", {}).get("thread_id", None)
            if thread_id is None:
                # TODO: Decide, if we want to capture the exception that will be raised by the openai.messages.create call in an observation in a new trace or not
                msg = "A message can only be created in the context of a thread, i.e. an existing trace. Please provide a thread_id."
                raise NotImplementedError(
                    "# TODO: Decide, if we want to capture the exception that will be raised by the openai.messages.create call in an observation in a new trace or not "
                )
            else:
                # get the trace with event of thread creation
                api = get_api()
                observations = api.observations.get_many(name=thread_id, type="EVENT")
                observation = observations.data[
                    0
                ]  # TODO: what in case observation does not exist?
                parsed_kwargs["trace_id"] = observation.trace_id

        observation = new_langfuse.event(**parsed_kwargs)
    else:
        observation = new_langfuse.generation(**parsed_kwargs)

    try:
        openai_response = wrapped(**arg_extractor.get_openai_args())

        if _is_streaming_response(openai_response):
            return _get_langfuse_data_from_sync_streaming_response(
                open_ai_resource, openai_response, observation, new_langfuse
            )
        else:
            model, completion, usage = _get_langfuse_data_from_default_response(
                open_ai_resource,
                openai_response.__dict__ if _is_openai_v1() else openai_response,
            )
            if open_ai_resource.type == "thread":
                observation.update(name=openai_response.id, output=completion)

            elif (
                open_ai_resource.type == "run" and open_ai_resource.method == "retrieve"
            ):
                _post_run_retrieve(
                    new_langfuse,
                    openai_response,
                    observation,
                    parsed_kwargs.get("trace_id"),
                    output=completion,
                )

            else:
                observation.update(
                    model=model,
                    output=completion,
                    end_time=_get_timestamp(),
                    usage=usage,
                )

        return openai_response
    except Exception as ex:
        log.warning(ex)
        model = kwargs.get("model", None)
        observation.update(
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
        if open_ai_resource.type == "run":
            pass

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
                    if url.startswith("data:image/jpeg;base64,"):
                        del content[index]["image_url"]

    return output_messages
