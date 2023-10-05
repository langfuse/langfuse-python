import os

import pytest
from langfuse.callback import CallbackHandler

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


def test_public_key_in_header_and_client():
    langfuse = Langfuse(public_key="test_LANGFUSE_PUBLIC_KEY")
    assert langfuse.client.x_langfuse_public_key == "test_LANGFUSE_PUBLIC_KEY"
    assert langfuse.client._username == "test_LANGFUSE_PUBLIC_KEY"


def test_secret_key_in_password():
    langfuse = Langfuse(secret_key="test_LANGFUSE_SECRET_KEY")
    assert langfuse.client._password == "test_LANGFUSE_SECRET_KEY"


def test_host_in_environment():
    langfuse = Langfuse(host="http://localhost:8000/")
    assert langfuse.client._environment == "http://localhost:8000/"


def test_set_via_constructor():
    langfuse = Langfuse(public_key="a", secret_key="b", host="http://host.com")
    assert langfuse.client._username == "a"
    assert langfuse.client._password == "b"
    assert langfuse.client._environment == "http://host.com"


def get_env_variables():
    return (
        os.environ["LANGFUSE_PUBLIC_KEY"],
        os.environ["LANGFUSE_SECRET_KEY"],
        os.environ["LANGFUSE_HOST"],
    )


def test_setup_without_keys():
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


def test_langfuse_default_host():
    _, _, host = get_env_variables()
    os.environ.pop("LANGFUSE_HOST")

    langfuse = Langfuse(debug=True)
    assert langfuse.base_url == "https://cloud.langfuse.com"
    os.environ["LANGFUSE_HOST"] = host


def test_langfuse_init():
    callback = CallbackHandler(debug=True)
    assert callback.trace is None
    assert not callback.runs


def test_langchain_setup_without_keys():
    public_key, secret_key, host = get_env_variables()
    os.environ.pop("LANGFUSE_PUBLIC_KEY")
    os.environ.pop("LANGFUSE_SECRET_KEY")
    os.environ.pop("LANGFUSE_HOST")

    os.environ["LANGFUSE_PUBLIC_KEY"] = "public_key"
    os.environ["LANGFUSE_SECRET_KEY"] = "secret_key"
    os.environ["LANGFUSE_HOST"] = "http://host.com"

    callback_handler = CallbackHandler()

    assert callback_handler.langfuse.client._username == "public_key"
    assert callback_handler.langfuse.client._environment == "http://host.com"
    assert callback_handler.langfuse.client._password == "secret_key"

    os.environ["LANGFUSE_PUBLIC_KEY"] = public_key
    os.environ["LANGFUSE_SECRET_KEY"] = secret_key
    os.environ["LANGFUSE_HOST"] = host


def test_setup_with_different_keys():
    public_key, secret_key, host = get_env_variables()
    os.environ.pop("LANGFUSE_PUBLIC_KEY")
    os.environ.pop("LANGFUSE_SECRET_KEY")
    os.environ.pop("LANGFUSE_HOST")
    with pytest.raises(ValueError):
        CallbackHandler()

    os.environ["LANGFUSE_PUBLIC_KEY"] = public_key
    os.environ["LANGFUSE_SECRET_KEY"] = secret_key
    os.environ["LANGFUSE_HOST"] = host


def test_langchain_setup_without_pk():
    public_key = os.environ["LANGFUSE_PUBLIC_KEY"]
    os.environ.pop("LANGFUSE_PUBLIC_KEY")
    with pytest.raises(ValueError):
        CallbackHandler()
    os.environ["LANGFUSE_PUBLIC_KEY"] = public_key


def test_langchain_setup_without_sk():
    secret_key = os.environ["LANGFUSE_SECRET_KEY"]
    os.environ.pop("LANGFUSE_SECRET_KEY")
    with pytest.raises(ValueError):
        CallbackHandler()
    os.environ["LANGFUSE_SECRET_KEY"] = secret_key


def test_public_key_in_header_and_password():
    handler = CallbackHandler(public_key="test_LANGFUSE_PUBLIC_KEY")
    assert handler.langfuse.client.x_langfuse_public_key == "test_LANGFUSE_PUBLIC_KEY"
    assert handler.langfuse.client._username == "test_LANGFUSE_PUBLIC_KEY"


def test_langchain_secret_key_in_password():
    handler = CallbackHandler(secret_key="test_LANGFUSE_SECRET_KEY")
    assert handler.langfuse.client._password == "test_LANGFUSE_SECRET_KEY"


def test_host_in_header():
    handler = CallbackHandler(host="http://localhost:8000/")
    assert handler.langfuse.client._environment == "http://localhost:8000/"
