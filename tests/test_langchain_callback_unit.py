"""Unit tests for CallbackHandler trace-name propagation.

These tests cover the fix for on_chain_start not passing trace_name to
propagate_attributes, which caused non-deterministic trace names on LangGraph
resume (e.g. after a human-in-the-loop interrupt).

No real API calls are made — propagate_attributes and get_client are mocked.
"""

import uuid
from contextlib import contextmanager
from typing import Any, Dict, Optional
from unittest.mock import MagicMock, call, patch

import pytest

from langfuse.langchain import CallbackHandler
from langfuse.langchain.CallbackHandler import (
    LangchainCallbackHandler,
    _strip_langfuse_keys_from_dict,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_handler() -> CallbackHandler:
    """Return a CallbackHandler with Langfuse SDK calls mocked out."""
    with patch("langfuse.langchain.CallbackHandler.get_client") as mock_get_client:
        mock_client = MagicMock()
        mock_client.start_observation.return_value = MagicMock(trace_id="trace-123")
        mock_get_client.return_value = mock_client
        handler = CallbackHandler()
    # Keep a reference so tests can inspect it
    handler._langfuse_client = MagicMock()
    handler._langfuse_client.start_observation.return_value = MagicMock(
        trace_id="trace-123"
    )
    return handler


def _make_run_id() -> uuid.UUID:
    return uuid.uuid4()


# ---------------------------------------------------------------------------
# Tests: _parse_langfuse_trace_attributes
# ---------------------------------------------------------------------------


class TestParseLangfuseTraceAttributes:
    def _parse(self, handler, metadata=None, tags=None):
        return handler._parse_langfuse_trace_attributes(
            metadata=metadata, tags=tags
        )

    def test_extracts_trace_name_from_metadata(self):
        handler = _make_handler()
        result = self._parse(
            handler, metadata={"langfuse_trace_name": "my-agent"}
        )
        assert result["trace_name"] == "my-agent"

    def test_ignores_non_string_trace_name(self):
        handler = _make_handler()
        result = self._parse(handler, metadata={"langfuse_trace_name": 42})
        assert "trace_name" not in result

    def test_does_not_set_trace_name_when_absent(self):
        handler = _make_handler()
        result = self._parse(handler, metadata={"langfuse_session_id": "s1"})
        assert "trace_name" not in result

    def test_extracts_all_attributes_together(self):
        handler = _make_handler()
        result = self._parse(
            handler,
            metadata={
                "langfuse_trace_name": "agent",
                "langfuse_session_id": "sess-1",
                "langfuse_user_id": "user-1",
            },
        )
        assert result["trace_name"] == "agent"
        assert result["session_id"] == "sess-1"
        assert result["user_id"] == "user-1"


# ---------------------------------------------------------------------------
# Tests: on_chain_start passes trace_name to propagate_attributes
# ---------------------------------------------------------------------------


class TestOnChainStartTraceNamePropagation:
    """Verify that on_chain_start forwards trace_name to propagate_attributes."""

    def _run_on_chain_start(
        self,
        handler: CallbackHandler,
        serialized: Optional[Dict[str, Any]] = None,
        name: Optional[str] = None,
        parent_run_id: Optional[uuid.UUID] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> uuid.UUID:
        run_id = _make_run_id()
        kwargs: Dict[str, Any] = {}
        if name is not None:
            kwargs["name"] = name
        handler.on_chain_start(
            serialized=serialized or {},
            inputs={},
            run_id=run_id,
            parent_run_id=parent_run_id,
            metadata=metadata,
            **kwargs,
        )
        return run_id

    def test_trace_name_passed_to_propagate_attributes(self):
        """span_name derived from serialized['name'] is forwarded as trace_name."""
        handler = _make_handler()

        @contextmanager
        def _noop_ctx(*args, **kwargs):
            yield

        with patch(
            "langfuse.langchain.CallbackHandler.propagate_attributes"
        ) as mock_pa:
            mock_pa.return_value = MagicMock(
                __enter__=MagicMock(return_value=None),
                __exit__=MagicMock(return_value=False),
            )
            self._run_on_chain_start(
                handler,
                serialized={"name": "my-agent"},
                parent_run_id=None,
            )

        mock_pa.assert_called_once()
        _, kwargs = mock_pa.call_args
        assert kwargs.get("trace_name") == "my-agent"

    def test_trace_name_uses_kwargs_name_over_serialized(self):
        """The 'name' kwarg takes priority over serialized dict (LangChain convention)."""
        handler = _make_handler()

        with patch(
            "langfuse.langchain.CallbackHandler.propagate_attributes"
        ) as mock_pa:
            mock_pa.return_value = MagicMock(
                __enter__=MagicMock(return_value=None),
                __exit__=MagicMock(return_value=False),
            )
            self._run_on_chain_start(
                handler,
                serialized={"name": "fallback-name"},
                name="explicit-name",
                parent_run_id=None,
            )

        _, kwargs = mock_pa.call_args
        assert kwargs.get("trace_name") == "explicit-name"

    def test_metadata_langfuse_trace_name_overrides_span_name(self):
        """langfuse_trace_name in metadata takes priority over computed span_name."""
        handler = _make_handler()

        with patch(
            "langfuse.langchain.CallbackHandler.propagate_attributes"
        ) as mock_pa:
            mock_pa.return_value = MagicMock(
                __enter__=MagicMock(return_value=None),
                __exit__=MagicMock(return_value=False),
            )
            self._run_on_chain_start(
                handler,
                serialized={"name": "computed-name"},
                metadata={"langfuse_trace_name": "override-name"},
                parent_run_id=None,
            )

        _, kwargs = mock_pa.call_args
        assert kwargs.get("trace_name") == "override-name"

    def test_propagate_attributes_not_called_for_child_runs(self):
        """propagate_attributes must only be called at the root (parent_run_id=None)."""
        handler = _make_handler()
        root_run_id = _make_run_id()
        handler._child_to_parent_run_id_map[root_run_id] = None

        with patch(
            "langfuse.langchain.CallbackHandler.propagate_attributes"
        ) as mock_pa:
            mock_pa.return_value = MagicMock(
                __enter__=MagicMock(return_value=None),
                __exit__=MagicMock(return_value=False),
            )
            # Child run — parent_run_id is set
            self._run_on_chain_start(
                handler,
                serialized={"name": "child-node"},
                parent_run_id=root_run_id,
            )

        mock_pa.assert_not_called()

    def test_empty_span_name_still_propagated(self):
        """Even when span_name resolves to '', it should still be forwarded."""
        handler = _make_handler()

        with patch(
            "langfuse.langchain.CallbackHandler.propagate_attributes"
        ) as mock_pa:
            mock_pa.return_value = MagicMock(
                __enter__=MagicMock(return_value=None),
                __exit__=MagicMock(return_value=False),
            )
            # Empty serialized — span_name will be '<unknown>'
            self._run_on_chain_start(
                handler,
                serialized=None,
                parent_run_id=None,
            )

        _, kwargs = mock_pa.call_args
        # '<unknown>' is what get_langchain_run_name returns for None serialized
        assert kwargs.get("trace_name") == "<unknown>"


# ---------------------------------------------------------------------------
# Tests: _strip_langfuse_keys_from_dict strips langfuse_trace_name
# ---------------------------------------------------------------------------


class TestStripLangfuseKeys:
    def test_strips_langfuse_trace_name(self):
        metadata = {
            "langfuse_trace_name": "my-agent",
            "other_key": "value",
        }
        result = _strip_langfuse_keys_from_dict(metadata, keep_langfuse_trace_attributes=False)
        assert "langfuse_trace_name" not in result
        assert result["other_key"] == "value"

    def test_keeps_langfuse_trace_name_when_flag_set(self):
        metadata = {
            "langfuse_trace_name": "my-agent",
            "other_key": "value",
        }
        result = _strip_langfuse_keys_from_dict(metadata, keep_langfuse_trace_attributes=True)
        assert result["langfuse_trace_name"] == "my-agent"

    def test_strips_all_trace_attribute_keys_together(self):
        metadata = {
            "langfuse_trace_name": "n",
            "langfuse_session_id": "s",
            "langfuse_user_id": "u",
            "langfuse_tags": ["t"],
            "keep_me": 1,
        }
        result = _strip_langfuse_keys_from_dict(metadata, keep_langfuse_trace_attributes=False)
        for key in ("langfuse_trace_name", "langfuse_session_id", "langfuse_user_id", "langfuse_tags"):
            assert key not in result
        assert result["keep_me"] == 1
