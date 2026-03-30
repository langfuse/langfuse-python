import logging
from importlib import import_module
from unittest.mock import MagicMock
from uuid import uuid4

from langfuse._client import propagation
from langfuse._client.propagation import propagate_attributes
from langfuse.langchain.CallbackHandler import LangchainCallbackHandler


def test_propagate_attributes_swallows_context_mismatch(monkeypatch, caplog):
    fake_token = object()

    monkeypatch.setattr(
        propagation.otel_context_api,
        "attach",
        lambda *, context: fake_token,
    )

    def raise_context_mismatch(token):
        assert token is fake_token
        raise ValueError("token was created in a different Context")

    monkeypatch.setattr(propagation._RUNTIME_CONTEXT, "detach", raise_context_mismatch)

    caplog.set_level(logging.ERROR, logger="opentelemetry.context")

    with propagate_attributes(user_id="test-user"):
        pass

    assert not [
        record
        for record in caplog.records
        if record.name == "opentelemetry.context"
        and "Failed to detach context" in record.getMessage()
    ]


def test_on_chain_error_exits_root_propagation_context(monkeypatch):
    mock_client = MagicMock()
    callback_handler_module = import_module("langfuse.langchain.CallbackHandler")
    monkeypatch.setattr(
        callback_handler_module, "get_client", lambda public_key=None: mock_client
    )

    handler = LangchainCallbackHandler()
    manager = MagicMock()
    observation = MagicMock()
    observation.update.return_value = observation
    run_id = uuid4()

    handler._propagation_context_manager = manager
    monkeypatch.setattr(handler, "_detach_observation", lambda _: observation)

    handler.on_chain_error(GeneratorExit(), run_id=run_id, parent_run_id=None)

    manager.__exit__.assert_called_once_with(None, None, None)
    observation.end.assert_called_once()
    assert handler._propagation_context_manager is None
