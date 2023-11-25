import importlib
import os
import httpx

import pytest
from pytest_httpserver import HTTPServer
from werkzeug import Response
import langfuse
from langfuse.api.resources.commons.errors.unauthorized_error import UnauthorizedError
from langfuse.callback import CallbackHandler

from langfuse.client import Langfuse
from langfuse.model import CreateTrace
from tests.test_task_manager import get_host
from langfuse.openai import _is_openai_v1, auth_check, openai

chat_func = openai.chat.completions.create if _is_openai_v1() else openai.ChatCompletion.create


def test_langfuse_release():
    # Backup environment variables to restore them later
    backup_environ = os.environ.copy()

    # Clearing the environment variables
    os.environ.clear()

    # These key are required
    client = Langfuse(public_key="test", secret_key="test")
    assert client.release is None

    # If neither the LANGFUSE_RELEASE env var nor the release parameter is given,
    # it should fall back to get_common_release_envs
    os.environ["CIRCLE_SHA1"] = "mock-sha1"
    client = Langfuse(public_key="test", secret_key="test")
    assert client.release == "mock-sha1"

    # If LANGFUSE_RELEASE env var is set, it should take precedence
    os.environ["LANGFUSE_RELEASE"] = "mock-langfuse-release"
    client = Langfuse(public_key="test", secret_key="test")
    assert client.release == "mock-langfuse-release"

    # If the release parameter is given during initialization, it should take the highest precedence
    client = Langfuse(public_key="test", secret_key="test", release="parameter-release")
    assert client.release == "parameter-release"

    # Restoring the environment variables
    os.environ.update(backup_environ)


# langfuse sdk
def test_setup_without_any_keys():
    public_key, secret_key, host = (
        os.environ["LANGFUSE_PUBLIC_KEY"],
        os.environ["LANGFUSE_SECRET_KEY"],
        os.environ["LANGFUSE_HOST"],
    )
    os.environ.pop("LANGFUSE_PUBLIC_KEY")
    os.environ.pop("LANGFUSE_SECRET_KEY")
    os.environ.pop("LANGFUSE_HOST")
    with pytest.raises(ValueError):
        Langfuse()

    os.environ["LANGFUSE_PUBLIC_KEY"] = public_key
    os.environ["LANGFUSE_SECRET_KEY"] = secret_key
    os.environ["LANGFUSE_HOST"] = host


def test_setup_without_pk():
    public_key = os.environ["LANGFUSE_PUBLIC_KEY"]
    os.environ.pop("LANGFUSE_PUBLIC_KEY")
    with pytest.raises(ValueError):
        Langfuse()
    os.environ["LANGFUSE_PUBLIC_KEY"] = public_key


def test_setup_without_sk():
    secret_key = os.environ["LANGFUSE_SECRET_KEY"]
    os.environ.pop("LANGFUSE_SECRET_KEY")
    with pytest.raises(ValueError):
        Langfuse()
    os.environ["LANGFUSE_SECRET_KEY"] = secret_key


def test_init_precedence_pk():
    langfuse = Langfuse(public_key="test_LANGFUSE_PUBLIC_KEY")
    assert langfuse.client._client_wrapper._x_langfuse_public_key == "test_LANGFUSE_PUBLIC_KEY"
    assert langfuse.client._client_wrapper._username == "test_LANGFUSE_PUBLIC_KEY"


def test_init_precedence_sk():
    langfuse = Langfuse(secret_key="test_LANGFUSE_SECRET_KEY")
    assert langfuse.client._client_wrapper._password == "test_LANGFUSE_SECRET_KEY"


def test_init_precedence_env():
    langfuse = Langfuse(host="http://localhost:8000/")
    assert langfuse.client._client_wrapper._base_url == "http://localhost:8000/"


def test_sdk_default_host():
    _, _, host = get_env_variables()
    os.environ.pop("LANGFUSE_HOST")

    langfuse = Langfuse()
    assert langfuse.base_url == "https://cloud.langfuse.com"
    os.environ["LANGFUSE_HOST"] = host


def test_sdk_default():
    public_key, secret_key, host = get_env_variables()

    langfuse = Langfuse()

    assert langfuse.client._client_wrapper._username == public_key
    assert langfuse.client._client_wrapper._password == secret_key
    assert langfuse.client._client_wrapper._base_url == host


# callback
def test_callback_setup_without_keys():
    public_key, secret_key, host = get_env_variables()
    os.environ.pop("LANGFUSE_PUBLIC_KEY")
    os.environ.pop("LANGFUSE_SECRET_KEY")
    os.environ.pop("LANGFUSE_HOST")
    with pytest.raises(ValueError):
        CallbackHandler()

    os.environ["LANGFUSE_PUBLIC_KEY"] = public_key
    os.environ["LANGFUSE_SECRET_KEY"] = secret_key
    os.environ["LANGFUSE_HOST"] = host


def test_callback_default_host():
    _, _, host = get_env_variables()
    os.environ.pop("LANGFUSE_HOST")

    handler = CallbackHandler(debug=True)
    assert handler.langfuse.client._client_wrapper._base_url == "https://cloud.langfuse.com"
    os.environ["LANGFUSE_HOST"] = host


def test_callback_setup():
    public_key, secret_key, host = get_env_variables()

    callback_handler = CallbackHandler()

    assert callback_handler.langfuse.client._client_wrapper._username == public_key
    assert callback_handler.langfuse.client._client_wrapper._base_url == host
    assert callback_handler.langfuse.client._client_wrapper._password == secret_key


def test_callback_setup_without_pk():
    public_key = os.environ["LANGFUSE_PUBLIC_KEY"]
    os.environ.pop("LANGFUSE_PUBLIC_KEY")
    with pytest.raises(ValueError):
        CallbackHandler()
    os.environ["LANGFUSE_PUBLIC_KEY"] = public_key


def test_callback_setup_without_sk():
    secret_key = os.environ["LANGFUSE_SECRET_KEY"]
    os.environ.pop("LANGFUSE_SECRET_KEY")
    with pytest.raises(ValueError):
        CallbackHandler()
    os.environ["LANGFUSE_SECRET_KEY"] = secret_key


def test_callback_init_precedence_pk():
    handler = CallbackHandler(public_key="test_LANGFUSE_PUBLIC_KEY")
    assert handler.langfuse.client._client_wrapper._x_langfuse_public_key == "test_LANGFUSE_PUBLIC_KEY"
    assert handler.langfuse.client._client_wrapper._username == "test_LANGFUSE_PUBLIC_KEY"


def test_callback_init_precedence_sk():
    handler = CallbackHandler(secret_key="test_LANGFUSE_SECRET_KEY")
    assert handler.langfuse.client._client_wrapper._password == "test_LANGFUSE_SECRET_KEY"


def test_callback_init_precedence_host():
    handler = CallbackHandler(host="http://localhost:8000/")
    assert handler.langfuse.client._client_wrapper._base_url == "http://localhost:8000/"


def test_callback_init_workers():
    handler = CallbackHandler()
    assert handler.langfuse.task_manager._threads == 1


def test_callback_init_workers_5():
    handler = CallbackHandler(threads=5)
    assert handler.langfuse.task_manager._threads == 5


def test_client_init_workers():
    langfuse = Langfuse()
    assert langfuse.task_manager._threads == 1


def test_openai_default():
    from langfuse.openai import modifier, openai

    importlib.reload(langfuse)
    importlib.reload(langfuse.openai)

    chat_func = openai.chat.completions.create if _is_openai_v1() else openai.ChatCompletion.create

    public_key, secret_key, host = (
        os.environ["LANGFUSE_PUBLIC_KEY"],
        os.environ["LANGFUSE_SECRET_KEY"],
        os.environ["LANGFUSE_HOST"],
    )

    chat_func(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "1 + 1 = "}],
        temperature=0,
        metadata={"someKey": "someResponse"},
    )

    openai.flush_langfuse()
    assert modifier._langfuse.client._client_wrapper._username == public_key
    assert modifier._langfuse.client._client_wrapper._password == secret_key
    assert modifier._langfuse.client._client_wrapper._base_url == host

    os.environ["LANGFUSE_PUBLIC_KEY"] = public_key
    os.environ["LANGFUSE_SECRET_KEY"] = secret_key
    os.environ["LANGFUSE_HOST"] = host


def test_openai_auth_check():
    assert auth_check() is True


def test_openai_auth_check_failing_key():
    secret_key = os.environ["LANGFUSE_SECRET_KEY"]
    os.environ.pop("LANGFUSE_SECRET_KEY")

    importlib.reload(langfuse)
    importlib.reload(langfuse.openai)

    from langfuse.openai import openai

    openai.langfuse_secret_key = "test"

    with pytest.raises(UnauthorizedError):
        auth_check()

    os.environ["LANGFUSE_SECRET_KEY"] = secret_key


def test_openai_configured(httpserver: HTTPServer):
    httpserver.expect_request("/api/public/ingestion", method="POST").respond_with_response(Response(status=200))
    host = get_host(httpserver.url_for("/api/public/ingestion"))

    importlib.reload(langfuse)
    importlib.reload(langfuse.openai)
    from langfuse.openai import modifier, openai

    chat_func = openai.chat.completions.create if _is_openai_v1() else openai.ChatCompletion.create

    public_key, secret_key, original_host = (
        os.environ["LANGFUSE_PUBLIC_KEY"],
        os.environ["LANGFUSE_SECRET_KEY"],
        os.environ["LANGFUSE_HOST"],
    )

    os.environ.pop("LANGFUSE_PUBLIC_KEY")
    os.environ.pop("LANGFUSE_SECRET_KEY")
    os.environ.pop("LANGFUSE_HOST")

    openai.langfuse_public_key = "pk-lf-asdfghjkl"
    openai.langfuse_secret_key = "sk-lf-asdfghjkl"
    openai.langfuse_host = host

    chat_func(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "1 + 1 = "}],
        temperature=0,
        metadata={"someKey": "someResponse"},
    )
    openai.flush_langfuse()

    assert modifier._langfuse.client._client_wrapper._username == "pk-lf-asdfghjkl"
    assert modifier._langfuse.client._client_wrapper._password == "sk-lf-asdfghjkl"
    assert modifier._langfuse.client._client_wrapper._base_url == host
    assert modifier._langfuse.task_manager._client._base_url == host

    os.environ["LANGFUSE_PUBLIC_KEY"] = public_key
    os.environ["LANGFUSE_SECRET_KEY"] = secret_key
    os.environ["LANGFUSE_HOST"] = original_host


def test_client_init_workers_5():
    langfuse = Langfuse(threads=5)
    langfuse.flush()

    assert langfuse.task_manager._threads == 5


def get_env_variables():
    return (
        os.environ["LANGFUSE_PUBLIC_KEY"],
        os.environ["LANGFUSE_SECRET_KEY"],
        os.environ["LANGFUSE_HOST"],
    )


def test_auth_check():
    langfuse = Langfuse(debug=False)

    assert langfuse.auth_check() is True

    langfuse.flush()


def test_wrong_key_auth_check():
    langfuse = Langfuse(debug=False, secret_key="test")

    with pytest.raises(UnauthorizedError):
        langfuse.auth_check()

    langfuse.flush()


def test_auth_check_callback():
    langfuse = CallbackHandler(debug=False)

    assert langfuse.auth_check() is True
    langfuse.flush()


def test_auth_check_callback_stateful():
    langfuse = Langfuse(debug=False)
    trace = langfuse.trace(CreateTrace(name="name"))
    handler = trace.get_langchain_handler()

    assert handler.auth_check() is True
    handler.flush()


def test_wrong_key_auth_check_callback():
    langfuse = CallbackHandler(debug=False, secret_key="test")

    with pytest.raises(UnauthorizedError):
        langfuse.auth_check()
    langfuse.flush()


def test_wrong_url_auth_check():
    langfuse = Langfuse(debug=False, host="http://localhost:4000/")

    with pytest.raises(httpx.ConnectError):
        langfuse.auth_check()

    langfuse.flush()


def test_wrong_url_auth_check_callback():
    langfuse = CallbackHandler(debug=False, host="http://localhost:4000/")

    with pytest.raises(httpx.ConnectError):
        langfuse.auth_check()
    langfuse.flush()
