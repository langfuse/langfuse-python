import os
from langfuse import Langfuse
from langfuse.callback import CallbackHandler


"""
Level	Numeric value
logging.DEBUG	10
logging.INFO	20
logging.WARNING	30
logging.ERROR	40
"""


def test_via_env():
    os.environ["LANGFUSE_DEBUG"] = "True"

    langfuse = Langfuse()

    assert langfuse.log.level == 10
    assert langfuse.task_manager._log.level == 10

    os.environ.pop("LANGFUSE_DEBUG")


def test_via_env_callback():
    os.environ["LANGFUSE_DEBUG"] = "True"

    callback = CallbackHandler()

    assert callback.log.level == 10
    assert callback.langfuse.log.level == 10
    assert callback.langfuse.task_manager._log.level == 10
    os.environ.pop("LANGFUSE_DEBUG")


def test_debug_langfuse():
    langfuse = Langfuse(debug=True)
    assert langfuse.log.level == 10
    assert langfuse.task_manager._log.level == 10


def test_default_langfuse():
    langfuse = Langfuse()
    assert langfuse.log.level == 30
    assert langfuse.task_manager._log.level == 30


def test_default_langfuse_callback():
    callback = CallbackHandler()
    assert callback.log.level == 30
    assert callback.log.level == 30
    assert callback.langfuse.log.level == 30


def test_debug_langfuse_callback():
    callback = CallbackHandler(debug=True)
    assert callback.log.level == 10
    assert callback.log.level == 10
    assert callback.langfuse.log.level == 10


def test_default_langfuse_trace_callback():
    langfuse = Langfuse()
    trace = langfuse.trace(name="test")
    callback = trace.getNewHandler()

    assert callback.log.level == 30
    assert callback.log.level == 30
    assert callback.trace.log.level == 30
    assert callback.trace.task_manager._log.level == 30


def test_debug_langfuse_trace_callback():
    langfuse = Langfuse(debug=True)
    trace = langfuse.trace(name="test")
    callback = trace.getNewHandler()

    assert callback.log.level == 10
    assert callback.log.level == 10
    assert callback.trace.log.level == 10
    assert callback.trace.task_manager._log.level == 10
