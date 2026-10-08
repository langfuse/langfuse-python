import importlib
import logging
import os

from langfuse import Langfuse
from langfuse.logger import langfuse_logger

"""
Level	Numeric value
logging.DEBUG	10
logging.INFO	20
logging.WARNING	30
logging.ERROR	40
"""


def test_default_langfuse():
    Langfuse()

    assert langfuse_logger.level == 30


def test_via_env():
    os.environ["LANGFUSE_DEBUG"] = "True"

    Langfuse()

    assert langfuse_logger.level == 10

    os.environ.pop("LANGFUSE_DEBUG")


def test_debug_langfuse():
    Langfuse(debug=True)
    assert langfuse_logger.level == 10

    # Reset
    langfuse_logger.setLevel("WARNING")


def test_httpx_handler_not_duplicated_when_handler_exists():
    import langfuse.logger as lf_logger

    httpx_logger = logging.getLogger("httpx")
    saved = httpx_logger.handlers[:]
    httpx_logger.handlers = []
    pre_existing = logging.NullHandler()
    httpx_logger.addHandler(pre_existing)
    try:
        importlib.reload(lf_logger)
        assert httpx_logger.handlers == [pre_existing]
    finally:
        httpx_logger.handlers = saved
