import os
import pytest
import time
import inspect
import io

from openai import BadRequestError

from langfuse.client import FernLangfuse
from langfuse.openai import openai
from tests.utils import create_uuid


# TODO: There is a lot of code duplication between the tests, refactor


def _convert_to_dict(obj):
    if isinstance(obj, dict):
        for key, value in obj.items():
            obj[key] = _convert_to_dict(value)
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            obj[i] = _convert_to_dict(item)
    elif hasattr(obj, "__dict__"):
        obj = _convert_to_dict(obj.__dict__)
    return obj


@pytest.fixture
def trace_id():
    return create_uuid()


@pytest.fixture(scope="module")
def openai_assistant_scope_module():
    assistant = openai.beta.assistants.create(model="gpt-3.5-turbo")
    yield assistant
    try:
        openai.beta.assistants.delete(assistant_id=assistant.id)
    except:
        pass


@pytest.fixture(scope="function")
def openai_assistant():
    assistant = openai.beta.assistants.create(
        model="gpt-3.5-turbo",
    )
    yield assistant
    try:
        openai.beta.assistants.delete(assistant_id=assistant.id)
    except:
        pass


@pytest.fixture(scope="module")
def openai_assistant_with_tools():
    assistant = openai.beta.assistants.create(
        model="gpt-4-turbo-preview", tools=[{"type": "retrieval"}]
    )
    yield assistant
    try:
        openai.beta.assistants.delete(assistant_id=assistant.id)
    except:
        pass


@pytest.fixture(scope="function")
def openai_thread():
    # TODO: Ideally this call should not be wrapped
    thread = openai.beta.threads.create()
    yield thread
    try:
        openai.beta.threads.delete(thread_id=thread.id)
    except:
        pass


@pytest.fixture(scope="function")
def openai_message(openai_thread):
    message = openai.beta.threads.messages.create(
        thread_id=openai_thread.id, content="I am a message", role="user"
    )
    yield message


@pytest.fixture(scope="function")
def openai_message_with_file(openai_thread, openai_file):
    message = openai.beta.threads.messages.create(
        thread_id=openai_thread.id,
        content="I am a message",
        role="user",
        file_ids=[openai_file.id],
    )
    yield message


@pytest.fixture(scope="function")
def openai_run(openai_thread, openai_assistant_scope_module):
    run = openai.beta.threads.runs.create(
        thread_id=openai_thread.id, assistant_id=openai_assistant_scope_module.id
    )
    yield run


@pytest.fixture(scope="function")
def openai_run_step(openai_run):
    # TODO: wait for openai_run to really change it's status
    time.sleep(2)
    run_step = openai.beta.threads.runs.steps.list(
        thread_id=openai_run.thread_id, run_id=openai_run.id
    ).data[0]
    yield run_step


@pytest.fixture(scope="module")
def openai_file():
    file = openai.files.create(
        file=io.BytesIO(b'{"message": "Hello world!"}'),  # needs to be .jsonl format
        purpose="assistants",
    )
    yield file
    try:
        openai.files.delete(file_id=file.id)
    except:
        pass


@pytest.fixture(scope="function")
def openai_assistant_file(openai_file, openai_assistant_with_tools):
    assistant_file = openai.beta.assistants.files.create(
        assistant_id=openai_assistant_with_tools.id, file_id=openai_file.id
    )
    yield assistant_file
    try:
        openai.beta.assistants.files.delete(
            assistant_id=assistant_file.assistant_id, file_id=assistant_file.id
        )
    except:
        pass


@pytest.fixture(scope="function")
def openai_message_file(openai_message_with_file):
    openai_message_file = openai.beta.threads.messages.files.retrieve(
        thread_id=openai_message_with_file.thread_id,
        message_id=openai_message_with_file.id,
        file_id=openai_message_with_file.file_ids[0],
    )
    yield openai_message_file


@pytest.fixture
def api():
    return FernLangfuse(
        username=os.environ.get("LANGFUSE_PUBLIC_KEY"),
        password=os.environ.get("LANGFUSE_SECRET_KEY"),
        base_url=os.environ.get("LANGFUSE_HOST"),
    )


def test_openai_assistant_create(api: FernLangfuse):
    openai_kwargs = {"model": "gpt-3.5-turbo"}

    langfuse_kwargs = {
        # "trace_id": trace_id,
    }
    fn = openai.beta.assistants.create

    openai_response = fn(**openai_kwargs, **langfuse_kwargs)
    openai.flush_langfuse()

    trace_id = openai_response.id
    observations = api.observations.get_many(
        trace_id=trace_id,
    )

    assert observations.data
    observation = observations.data[0]
    for key, value in openai_kwargs.items():
        assert observation.input[key] == value

    assert observation.output == dict(openai_response)


def test_openai_assistant_list(api: FernLangfuse, trace_id):
    # TODO: include openai_assistant
    openai_kwargs = {}

    langfuse_kwargs = {
        "trace_id": trace_id,
    }
    fn = openai.beta.assistants.list

    openai_response = fn(**openai_kwargs, **langfuse_kwargs)
    openai.flush_langfuse()

    observations = api.observations.get_many(
        trace_id=trace_id,
    )

    assert observations.data
    observation = observations.data[0]
    for key, value in openai_kwargs.items():
        assert observation.input[key] == value

    assert observation.output == {"data": _convert_to_dict(openai_response.data)}


@pytest.mark.parametrize(
    "openai_call,openai_kwargs",
    [
        (openai.beta.assistants.retrieve, {}),
        (openai.beta.assistants.update, {"description": "I am an updated description"}),
        (openai.beta.assistants.delete, {}),
    ],
)
def test_openai_assistant(
    openai_call, openai_kwargs, api: FernLangfuse, trace_id, openai_assistant
):
    openai_kwargs["assistant_id"] = openai_assistant.id

    langfuse_kwargs = {
        "trace_id": trace_id,
    }

    openai_response = openai_call(**openai_kwargs)
    openai.flush_langfuse()

    trace_id = openai_kwargs["assistant_id"]
    observations = api.observations.get_many(
        trace_id=trace_id,
    )

    assert observations.data
    observation = observations.data[0]
    for key, value in openai_kwargs.items():
        assert observation.input[key] == value

    assert observation.output == _convert_to_dict(openai_response)


def test_openai_assistant_file_create(
    api: FernLangfuse, openai_assistant_with_tools, openai_file
):
    openai_kwargs = {
        "assistant_id": openai_assistant_with_tools.id,
        "file_id": openai_file.id,
    }

    langfuse_kwargs = {
        # "trace_id": trace_id,
    }
    fn = openai.beta.assistants.files.create

    openai_response = fn(**openai_kwargs, **langfuse_kwargs)
    openai.flush_langfuse()

    trace_id = openai_assistant_with_tools.id
    observations = api.observations.get_many(
        trace_id=trace_id,
    )

    assert observations.data
    observation = observations.data[0]
    for key, value in openai_kwargs.items():
        assert observation.input[key] == value

    assert observation.output == dict(openai_response)


@pytest.mark.parametrize(
    "openai_call,openai_kwargs",
    [
        (openai.beta.assistants.files.delete, {"assistant_id": None, "file_id": None}),
        (
            openai.beta.assistants.files.retrieve,
            {"assistant_id": None, "file_id": None},
        ),
        (openai.beta.assistants.files.list, {"assistant_id": None}),
    ],
)
def test_openai_assistant_file(
    api: FernLangfuse, openai_call, openai_kwargs, trace_id, openai_assistant_file
):
    openai_kwargs["assistant_id"] = openai_assistant_file.assistant_id
    if "file_id" in openai_kwargs:
        openai_kwargs["file_id"] = openai_assistant_file.id

    langfuse_kwargs = {
        # "trace_id": trace_id,
    }

    openai_response = openai_call(**openai_kwargs, **langfuse_kwargs)
    openai.flush_langfuse()

    trace_id = openai_assistant_file.assistant_id
    observations = api.observations.get_many(
        trace_id=trace_id,
    )

    assert observations.data
    observation = observations.data[0]
    for key, value in openai_kwargs.items():
        assert observation.input[key] == value

    if openai_call.__name__ == "list":
        assert observation.output == {"data": _convert_to_dict(openai_response.data)}
    else:
        assert observation.output == _convert_to_dict(openai_response)


def test_openai_thread_create(api: FernLangfuse):
    openai_kwargs = {}

    langfuse_kwargs = {
        # "trace_id": trace_id,
    }
    fn = openai.beta.threads.create

    openai_response = fn(**openai_kwargs, **langfuse_kwargs)
    openai.flush_langfuse()

    trace_id = openai_response.id
    observations = api.observations.get_many(
        trace_id=trace_id,
    )

    assert observations.data
    observation = observations.data[0]
    for key, value in openai_kwargs.items():
        assert observation.input[key] == value

    assert observation.output == dict(openai_response)


@pytest.mark.parametrize(
    "openai_call,openai_kwargs",
    [
        (openai.beta.threads.delete, {}),
        (openai.beta.threads.retrieve, {}),
        (openai.beta.threads.update, {"metadata": {"info": "I am now updated!"}}),
    ],
)
def test_openai_thread(
    openai_call, openai_kwargs, api: FernLangfuse, trace_id, openai_thread
):
    openai_kwargs.update({"thread_id": openai_thread.id})
    langfuse_kwargs = {
        "trace_id": trace_id,
    }

    openai_response = openai_call(**openai_kwargs)
    openai.flush_langfuse()

    trace_id = openai_kwargs["thread_id"]
    observations = api.observations.get_many(
        trace_id=trace_id,
    )

    assert observations.data
    observation = observations.data[0]
    for key, value in openai_kwargs.items():
        assert observation.input[key] == value

    assert observation.output == dict(openai_response)


def test_openai_thread_create_and_run(api: FernLangfuse, openai_assistant):
    openai_kwargs = {"assistant_id": openai_assistant.id}

    langfuse_kwargs = {
        # "trace_id": trace_id,
    }
    fn = openai.beta.threads.create_and_run

    openai_response = fn(**openai_kwargs, **langfuse_kwargs)
    openai.flush_langfuse()

    trace_id = openai_response.thread_id
    observations = api.observations.get_many(
        trace_id=trace_id,
    )

    assert observations.data
    observation = observations.data[0]
    for key, value in openai_kwargs.items():
        assert observation.input[key] == value

    assert observation.output == dict(openai_response)


def test_openai_message_create(api: FernLangfuse, openai_thread):
    openai_kwargs = {
        "thread_id": openai_thread.id,
        "role": "user",
        "content": "Alles ist gut :)",
    }

    langfuse_kwargs = {
        # "trace_id": trace_id,
    }
    fn = openai.beta.threads.messages.create

    openai_response = fn(**openai_kwargs, **langfuse_kwargs)
    openai.flush_langfuse()

    trace_id = openai_thread.id
    observations = api.observations.get_many(
        trace_id=trace_id,
    )

    assert observations.data
    observation = observations.data[0]
    for key, value in openai_kwargs.items():
        assert observation.input[key] == value

    assert observation.output == _convert_to_dict(openai_response)


@pytest.mark.parametrize(
    "openai_call,openai_kwargs",
    [
        (
            openai.beta.threads.messages.retrieve,
            {"thread_id": None, "message_id": None},
        ),
        (
            openai.beta.threads.messages.update,
            {
                "thread_id": None,
                "message_id": None,
                "metadata": {"info": "I am now updated!", "modified": True},
            },
        ),
        (openai.beta.threads.messages.list, {"thread_id": None}),
    ],
)
def test_openai_message(
    openai_call, openai_kwargs, api: FernLangfuse, trace_id, openai_message
):
    openai_kwargs["thread_id"] = openai_message.thread_id
    if "message_id" in openai_kwargs:
        openai_kwargs["message_id"] = openai_message.id
    langfuse_kwargs = {
        "trace_id": trace_id,
    }

    openai_response = openai_call(**openai_kwargs)
    openai.flush_langfuse()

    trace_id = openai_kwargs["thread_id"]
    observations = api.observations.get_many(
        trace_id=trace_id,
    )

    assert observations.data
    observation = observations.data[0]
    for key, value in openai_kwargs.items():
        assert observation.input[key] == value

    if openai_call.__name__ == "list":
        assert observation.output == {"data": _convert_to_dict(openai_response.data)}
    else:
        assert observation.output == _convert_to_dict(openai_response)


@pytest.mark.parametrize(
    "openai_call,openai_kwargs",
    [
        (
            openai.beta.threads.messages.files.retrieve,
            {"thread_id": None, "message_id": None, "file_id": None},
        ),
        (
            openai.beta.threads.messages.files.list,
            {"thread_id": None, "message_id": None},
        ),
    ],
)
def test_openai_message_files(
    openai_call, openai_kwargs, api: FernLangfuse, trace_id, openai_message_with_file
):
    openai_kwargs["thread_id"] = openai_message_with_file.thread_id
    openai_kwargs["message_id"] = openai_message_with_file.id
    if "file_id" in openai_kwargs:
        openai_kwargs["file_id"] = openai_message_with_file.file_ids[0]

    langfuse_kwargs = {
        "trace_id": trace_id,
    }

    openai_response = openai_call(**openai_kwargs)
    openai.flush_langfuse()

    trace_id = openai_kwargs["thread_id"]
    observations = api.observations.get_many(
        trace_id=trace_id,
    )

    assert observations.data
    observation = observations.data[0]
    for key, value in openai_kwargs.items():
        assert observation.input[key] == value

    if openai_call.__name__ == "list":
        assert observation.output == {"data": _convert_to_dict(openai_response.data)}
    else:
        assert observation.output == _convert_to_dict(openai_response)


def test_openai_run_create(api: FernLangfuse, openai_thread, openai_assistant):
    openai_kwargs = {"thread_id": openai_thread.id, "assistant_id": openai_assistant.id}

    langfuse_kwargs = {
        # "trace_id": trace_id,
    }
    fn = openai.beta.threads.runs.create

    openai_response = fn(**openai_kwargs, **langfuse_kwargs)
    openai.flush_langfuse()

    trace_id = openai_thread.id
    observations = api.observations.get_many(
        trace_id=trace_id,
    )

    assert observations.data
    observation = observations.data[0]
    for key, value in openai_kwargs.items():
        assert observation.input[key] == value

    assert observation.output == _convert_to_dict(openai_response)


@pytest.mark.parametrize(
    "openai_call,openai_kwargs",
    [
        (openai.beta.threads.runs.retrieve, {"thread_id": None, "run_id": None}),
        (
            openai.beta.threads.runs.update,
            {
                "thread_id": None,
                "run_id": None,
                "metadata": {"info": "I am now updated!"},
            },
        ),
        (openai.beta.threads.runs.cancel, {"thread_id": None, "run_id": None}),
        (openai.beta.threads.runs.list, {"thread_id": None}),
    ],
)
def test_openai_run(
    openai_call, openai_kwargs, api: FernLangfuse, trace_id, openai_run
):
    openai_kwargs["thread_id"] = openai_run.thread_id
    if "run_id" in openai_kwargs:
        openai_kwargs["run_id"] = openai_run.id

    langfuse_kwargs = {
        "trace_id": trace_id,
    }

    if openai_call.__name__ in ["cancel", "update"]:
        # we have no control over if we actually can cancel or update the run, since depended on the run status the openai_client throws an error
        try:
            openai_response = openai_call(**openai_kwargs)
        except BadRequestError:
            # has the error been logged?
            openai.flush_langfuse()
            trace_id = openai_kwargs["thread_id"]
            observations = api.observations.get_many(
                trace_id=trace_id,
            )
            assert observations.data
            observation = observations.data[0]
            assert (
                observation.status_message
            )  # TODO: How to test observation.level instead
            return
    else:
        openai_response = openai_call(**openai_kwargs)

    openai.flush_langfuse()

    trace_id = openai_kwargs["thread_id"]
    observations = api.observations.get_many(
        trace_id=trace_id,
    )

    assert observations.data
    observation = observations.data[0]
    for key, value in openai_kwargs.items():
        assert observation.input[key] == value

    if openai_call.__name__ == "list":
        # TODO: Investigate why "usage" key exists in observation output but NOT in opeani_response.data !?!??!!
        # current findings: call dict(openai_response) removes the usage key :O
        # see https://github.com/openai/openai-python/pull/1090/commits/0c4de9b38fdc3464f249fc16d324b87b6a6953e8
        for x in observation.output["data"]:
            del x["usage"]

        assert observation.output == {"data": _convert_to_dict(openai_response.data)}
    else:
        assert observation.output == _convert_to_dict(openai_response)


def test_openai_run_steps_list(api: FernLangfuse, trace_id, openai_run):
    openai_kwargs = {"thread_id": openai_run.thread_id, "run_id": openai_run.id}

    langfuse_kwargs = {
        "trace_id": trace_id,
    }

    time.sleep(1)  # TODO: this is only so that the openai_run is already in progress
    openai_response = openai.beta.threads.runs.steps.list(**openai_kwargs)

    openai.flush_langfuse()

    trace_id = openai_kwargs["thread_id"]

    observations = api.observations.get_many(
        trace_id=trace_id,
    )

    assert observations.data
    observation = observations.data[0]
    for key, value in openai_kwargs.items():
        assert observation.input[key] == value

    # TODO: Investigate why "usage" key exists in observation output but NOT in opeani_response.data !?!??!!
    # current findings: call dict(openai_response) removes the usage key :O
    # see https://github.com/openai/openai-python/pull/1090/commits/0c4de9b38fdc3464f249fc16d324b87b6a6953e8
    # TODO: No idea what happens to "expires_at"
    for x in observation.output["data"]:
        del x["usage"]
        del x["expires_at"]

    assert observation.output == {"data": _convert_to_dict(openai_response.data)}


def test_openai_run_steps_retrieve(api: FernLangfuse, trace_id, openai_run_step):
    openai_kwargs = {
        "thread_id": openai_run_step.thread_id,
        "run_id": openai_run_step.run_id,
        "step_id": openai_run_step.id,
    }

    langfuse_kwargs = {
        "trace_id": trace_id,
    }

    openai_response = openai.beta.threads.runs.steps.retrieve(**openai_kwargs)

    openai.flush_langfuse()

    trace_id = openai_kwargs["thread_id"]

    observations = api.observations.get_many(
        trace_id=trace_id,
    )

    assert observations.data
    observation = observations.data[0]
    for key, value in openai_kwargs.items():
        assert observation.input[key] == value

    # TODO: Investigate why "usage" key exists in observation output but NOT in opeani_response.data !?!??!!
    # current findings: call dict(openai_response) removes the usage key :O
    # see https://github.com/openai/openai-python/pull/1090/commits/0c4de9b38fdc3464f249fc16d324b87b6a6953e8
    assert observation.output == _convert_to_dict(openai_response)


def test_full_example(api, openai_assistant_with_tools, openai_file):
    # this is a full end2end test to simulate possible user behavior

    openai_thread = openai.beta.threads.create()
    openai_message = openai.beta.threads.messages.create(
        thread_id=openai_thread.id,
        content="You are a test thread",
        role="user",
        file_ids=[openai_file.id],
    )
    openai_run = openai.beta.threads.runs.create(
        thread_id=openai_thread.id,
        assistant_id=openai_assistant_with_tools.id,
        instructions="Say hello",
    )

    while openai_run.status in ["queued", "in_progress"]:
        openai_run = openai.beta.threads.runs.retrieve(
            run_id=openai_run.id, thread_id=openai_thread.id
        )
        time.sleep(1)

    openai_run_steps = openai.beta.threads.runs.steps.list(
        run_id=openai_run.id, thread_id=openai_thread.id
    )
    openai_message = openai.beta.threads.messages.list(thread_id=openai_thread.id)

    openai_thread = openai.beta.threads.delete(thread_id=openai_thread.id)
    openai.flush_langfuse()

    trace_id = openai_thread.id
    observations = api.observations.get_many(
        trace_id=trace_id,
    )

    assert (
        len(observations.data) >= 7
    )  # we do not know in advance how many time run.retrieve is called
    # TODO: better check
