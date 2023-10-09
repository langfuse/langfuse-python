from langfuse import Langfuse
from langfuse.callback.langchain import CallbackHandler
from langfuse.model import (
    CreateTrace,
)


"""
Level	Numeric value
logging.DEBUG	10
logging.INFO	20
logging.WARNING	30
logging.ERROR	40
"""


def test_debug_langfuse():
    langfuse = Langfuse(debug=True)
    assert langfuse.log.level == 10
    assert langfuse.task_manager.log.level == 10


def test_default_langfuse():
    langfuse = Langfuse()
    assert langfuse.log.level == 30
    assert langfuse.task_manager.log.level == 30


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
    trace = langfuse.trace(CreateTrace(name="test"))
    callback = trace.getNewHandler()

    assert callback.log.level == 30
    assert callback.log.level == 30
    assert callback.trace.log.level == 30
    assert callback.trace.task_manager.log.level == 30


def test_debug_langfuse_trace_callback():
    langfuse = Langfuse(debug=True)
    trace = langfuse.trace(CreateTrace(name="test"))
    callback = trace.getNewHandler()

    assert callback.log.level == 10
    assert callback.log.level == 10
    assert callback.trace.log.level == 10
    assert callback.trace.task_manager.log.level == 10
