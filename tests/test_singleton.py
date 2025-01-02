import threading
from unittest.mock import patch

import pytest

from langfuse.utils.langfuse_singleton import LangfuseSingleton


@pytest.fixture(autouse=True)
def reset_singleton():
    LangfuseSingleton._instance = None
    LangfuseSingleton._langfuse = None
    yield
    LangfuseSingleton._instance = None
    LangfuseSingleton._langfuse = None


def test_singleton_instance():
    """Test that the LangfuseSingleton class truly implements singleton behavior."""
    instance1 = LangfuseSingleton()
    instance2 = LangfuseSingleton()

    assert instance1 is instance2


def test_singleton_thread_safety():
    """Test the thread safety of the LangfuseSingleton class."""

    def get_instance(results):
        instance = LangfuseSingleton()
        results.append(instance)

    results = []
    threads = [
        threading.Thread(target=get_instance, args=(results,)) for _ in range(10)
    ]

    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    for instance in results:
        assert instance is results[0]


@patch("langfuse.utils.langfuse_singleton.Langfuse")
def test_langfuse_initialization(mock_langfuse):
    instance = LangfuseSingleton()
    created = instance.get(public_key="key123", secret_key="secret", debug=True)
    mock_langfuse.assert_called_once_with(
        public_key="key123",
        secret_key="secret",
        debug=True,
    )

    assert created is mock_langfuse.return_value


@patch("langfuse.utils.langfuse_singleton.Langfuse")
def test_reset_functionality(mock_langfuse):
    """Test the reset functionality of the LangfuseSingleton."""
    instance = LangfuseSingleton()
    instance.get(public_key="key123")
    instance.reset()

    assert instance._langfuse is None

    mock_langfuse.return_value.shutdown.assert_called_once()
