import os

import pytest
from langfuse.callback.langchain import CallbackHandler

from langfuse.client import Langfuse


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
    assert langfuse.client.x_langfuse_public_key == "test_LANGFUSE_PUBLIC_KEY"
    assert langfuse.client._username == "test_LANGFUSE_PUBLIC_KEY"


def test_init_precedence_sk():
    langfuse = Langfuse(secret_key="test_LANGFUSE_SECRET_KEY")
    assert langfuse.client._password == "test_LANGFUSE_SECRET_KEY"


def test_init_precedence_env():
    langfuse = Langfuse(host="http://localhost:8000/")
    assert langfuse.client._environment == "http://localhost:8000/"


def test_sdk_default_host():
    _, _, host = get_env_variables()
    os.environ.pop("LANGFUSE_HOST")

    langfuse = Langfuse(debug=True)
    assert langfuse.base_url == "https://cloud.langfuse.com"
    os.environ["LANGFUSE_HOST"] = host


def test_sdk_default():
    public_key, secret_key, host = get_env_variables()

    langfuse = Langfuse()

    assert langfuse.client._username == public_key
    assert langfuse.client._password == secret_key
    assert langfuse.client._environment == host


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
    assert handler.langfuse.base_url == "https://cloud.langfuse.com"
    os.environ["LANGFUSE_HOST"] = host


def test_callback_setup():
    public_key, secret_key, host = get_env_variables()

    callback_handler = CallbackHandler()

    assert callback_handler.langfuse.client._username == public_key
    assert callback_handler.langfuse.client._environment == host
    assert callback_handler.langfuse.client._password == secret_key


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
    assert handler.langfuse.client.x_langfuse_public_key == "test_LANGFUSE_PUBLIC_KEY"
    assert handler.langfuse.client._username == "test_LANGFUSE_PUBLIC_KEY"


def test_callback_init_precedence_sk():
    handler = CallbackHandler(secret_key="test_LANGFUSE_SECRET_KEY")
    assert handler.langfuse.client._password == "test_LANGFUSE_SECRET_KEY"


def test_callback_init_precedence_host():
    handler = CallbackHandler(host="http://localhost:8000/")
    assert handler.langfuse.client._environment == "http://localhost:8000/"


def get_env_variables():
    return (
        os.environ["LANGFUSE_PUBLIC_KEY"],
        os.environ["LANGFUSE_SECRET_KEY"],
        os.environ["LANGFUSE_HOST"],
    )
