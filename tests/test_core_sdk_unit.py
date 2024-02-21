import os
import time
from asyncio import gather
from datetime import datetime, timezone
from unittest.mock import Mock
from langfuse.api.client import FernLangfuse
from langfuse.client import (
    StatefulClient,
    StatefulGenerationClient,
    StatefulSpanClient,
    StatefulTraceClient,
)

from langfuse.utils import _get_timestamp


import pytest

from langfuse import Langfuse
from tests.api_wrapper import LangfuseAPI


@pytest.fixture
def langfuse():
    langfuse_instance = Langfuse(debug=False)
    langfuse_instance.client = Mock()
    langfuse_instance.task_manager = Mock()
    langfuse_instance.log = Mock()

    return langfuse_instance


@pytest.fixture
def stateful_client():
    stateful_client = StatefulClient(Mock(), "test_id", Mock(), "test_trace", Mock())

    return stateful_client


@pytest.mark.parametrize(
    "trace_method, expected_client, kwargs",
    [
        (Langfuse.trace, StatefulTraceClient, {}),
        (Langfuse.generation, StatefulGenerationClient, {}),
        (Langfuse.span, StatefulSpanClient, {}),
        (Langfuse.score, StatefulClient, {"value": 1, "trace_id": "test_trace"}),
    ],
)
def test_langfuse_returning_if_taskmanager_fails(
    langfuse, trace_method, expected_client, kwargs
):
    trace_name = "test_trace"

    mock_task_manager = langfuse.task_manager.add_task
    mock_task_manager.return_value = Exception("Task manager unable to process event")


    body = {
        "name": trace_name,
        **kwargs,
    }

    result = trace_method(langfuse, **body)
    mock_task_manager.assert_called()

    assert isinstance(result, expected_client)


@pytest.mark.parametrize(
    "trace_method, expected_client, kwargs",
    [
        (StatefulClient.generation, StatefulGenerationClient, {}),
        (StatefulClient.span, StatefulSpanClient, {}),
        (StatefulClient.score, StatefulClient, {"value": 1}),
    ],
)
def test_stateful_client_returning_if_taskmanager_fails(
    stateful_client, trace_method, expected_client, kwargs
):
    trace_name = "test_trace"

    mock_task_manager = stateful_client.task_manager.add_task
    mock_task_manager.return_value = Exception("Task manager unable to process event")
    mock_client = stateful_client.client
    mock_client.return_value = FernLangfuse(base_url="http://localhost:8000")

    body = {
        "name": trace_name,
        **kwargs,
    }

    result = trace_method(stateful_client, **body)
    mock_task_manager.assert_called()

    assert isinstance(result, expected_client)
