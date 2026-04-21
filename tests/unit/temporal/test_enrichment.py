"""Unit tests for the span-enrichment helper.

Exercises the ``_Enrichment`` class directly with a fake span so the tests
do not need a running Temporal worker. This covers the attributes we write
onto the current OTel span, the session/user/tag/metadata factories, the
default session = workflow_id behaviour, and the payload capture path.
"""

from __future__ import annotations

import json
from typing import Any, Dict

import pytest

from langfuse.temporal.attributes import (
    LANGFUSE_INPUT,
    LANGFUSE_METADATA_PREFIX,
    LANGFUSE_OUTPUT,
    LANGFUSE_SESSION_ID,
    LANGFUSE_TAGS,
    LANGFUSE_USER_ID,
    TEMPORAL_ACTIVITY_TYPE,
    TEMPORAL_IS_LOCAL_ACTIVITY,
    TEMPORAL_RUN_ID,
    TEMPORAL_WORKFLOW_ID,
    TEMPORAL_WORKFLOW_TYPE,
)
from langfuse.temporal.config import CaptureConfig, LangfusePluginConfig
from langfuse.temporal.interceptor import _Enrichment


class FakeSpan:
    """Minimal stand-in for an OTel span for assertion purposes."""

    def __init__(self) -> None:
        self.attributes: Dict[str, Any] = {}

    def set_attribute(self, key: str, value: Any) -> None:
        self.attributes[key] = value


class FakeWorkflowInfo:
    def __init__(self) -> None:
        self.workflow_id = "wf-123"
        self.run_id = "run-abc"
        self.workflow_type = "OrderWorkflow"
        self.namespace = "default"
        self.task_queue = "main-tq"
        self.attempt = 1


class FakeActivityInfo:
    def __init__(self) -> None:
        self.workflow_id = "wf-123"
        self.workflow_run_id = "run-abc"
        self.activity_id = "act-1"
        self.activity_type = "ChargeCardActivity"
        self.workflow_namespace = "default"
        self.task_queue = "main-tq"
        self.attempt = 2


@pytest.fixture()
def patched_current_span(monkeypatch):
    """Patch the current-span lookup on ``_Enrichment`` to return a fake."""
    span = FakeSpan()
    monkeypatch.setattr(_Enrichment, "_current_span", lambda self: span)
    return span


@pytest.mark.unit
def test_workflow_execute_attaches_temporal_identifiers(patched_current_span):
    cfg = LangfusePluginConfig()
    enrichment = _Enrichment(cfg)
    info = FakeWorkflowInfo()

    enrichment.on_workflow_execute(info, args=("input",))

    attrs = patched_current_span.attributes
    assert attrs[TEMPORAL_WORKFLOW_ID] == "wf-123"
    assert attrs[TEMPORAL_RUN_ID] == "run-abc"
    assert attrs[TEMPORAL_WORKFLOW_TYPE] == "OrderWorkflow"
    # Default session id = workflow_id per requirements doc §9.1.
    assert attrs[LANGFUSE_SESSION_ID] == "wf-123"


@pytest.mark.unit
def test_workflow_execute_runs_session_and_user_factories(patched_current_span):
    cfg = LangfusePluginConfig(
        session_id_factory=lambda ctx: f"session-{ctx.info.workflow_id}",
        user_id_factory=lambda ctx: "user-42",
    )
    enrichment = _Enrichment(cfg)
    enrichment.on_workflow_execute(FakeWorkflowInfo(), args=())

    attrs = patched_current_span.attributes
    assert attrs[LANGFUSE_SESSION_ID] == "session-wf-123"
    assert attrs[LANGFUSE_USER_ID] == "user-42"


@pytest.mark.unit
def test_tags_factory_merges_with_static_tags(patched_current_span):
    cfg = LangfusePluginConfig(
        static_tags=["prod"],
        tags_factory=lambda ctx: ["temporal", "order"],
    )
    enrichment = _Enrichment(cfg)
    enrichment.on_workflow_execute(FakeWorkflowInfo(), args=())

    tags = json.loads(patched_current_span.attributes[LANGFUSE_TAGS])
    assert set(tags) == {"prod", "temporal", "order"}


@pytest.mark.unit
def test_metadata_factory_keys_are_prefixed(patched_current_span):
    cfg = LangfusePluginConfig(
        metadata_factory=lambda ctx: {"customer_tier": "gold", "region": "us"},
    )
    enrichment = _Enrichment(cfg)
    enrichment.on_workflow_execute(FakeWorkflowInfo(), args=())

    attrs = patched_current_span.attributes
    assert attrs[LANGFUSE_METADATA_PREFIX + "customer_tier"] == "gold"
    assert attrs[LANGFUSE_METADATA_PREFIX + "region"] == "us"


@pytest.mark.unit
def test_capture_workflow_inputs_only_when_enabled(patched_current_span):
    cfg = LangfusePluginConfig(capture=CaptureConfig(capture_workflow_inputs=False))
    enrichment = _Enrichment(cfg)
    enrichment.on_workflow_execute(FakeWorkflowInfo(), args=("x",))
    assert LANGFUSE_INPUT not in patched_current_span.attributes


@pytest.mark.unit
def test_capture_workflow_inputs_attaches_serialized_payload(patched_current_span):
    cfg = LangfusePluginConfig(capture=CaptureConfig(capture_workflow_inputs=True))
    enrichment = _Enrichment(cfg)
    enrichment.on_workflow_execute(FakeWorkflowInfo(), args=("hello", 42))

    payload = patched_current_span.attributes[LANGFUSE_INPUT]
    assert json.loads(payload) == ["hello", 42]


@pytest.mark.unit
def test_capture_respects_workflow_denylist(patched_current_span):
    cfg = LangfusePluginConfig(
        capture=CaptureConfig(
            capture_workflow_inputs=True,
            workflow_denylist=["OrderWorkflow"],
        )
    )
    enrichment = _Enrichment(cfg)
    enrichment.on_workflow_execute(FakeWorkflowInfo(), args=("hello",))
    assert LANGFUSE_INPUT not in patched_current_span.attributes


@pytest.mark.unit
def test_activity_execute_sets_activity_attributes(patched_current_span):
    cfg = LangfusePluginConfig()
    enrichment = _Enrichment(cfg)
    enrichment.on_activity_execute(FakeActivityInfo(), args=("arg",), is_local=True)

    attrs = patched_current_span.attributes
    assert attrs[TEMPORAL_ACTIVITY_TYPE] == "ChargeCardActivity"
    assert attrs[TEMPORAL_WORKFLOW_ID] == "wf-123"
    assert attrs[TEMPORAL_IS_LOCAL_ACTIVITY] is True


@pytest.mark.unit
def test_activity_complete_captures_output_when_enabled(patched_current_span):
    cfg = LangfusePluginConfig(
        capture=CaptureConfig(capture_activity_outputs=True),
    )
    enrichment = _Enrichment(cfg)
    enrichment.on_activity_complete(FakeActivityInfo(), result={"status": "ok"})

    payload = patched_current_span.attributes[LANGFUSE_OUTPUT]
    assert json.loads(payload) == {"status": "ok"}


@pytest.mark.unit
def test_enrichment_never_raises_even_if_factory_is_broken(patched_current_span):
    def bad_factory(ctx):
        raise RuntimeError("boom")

    cfg = LangfusePluginConfig(
        session_id_factory=bad_factory,
        user_id_factory=bad_factory,
        tags_factory=bad_factory,
        metadata_factory=bad_factory,
    )
    enrichment = _Enrichment(cfg)
    # Should not raise; we fall back to default session = workflow_id.
    enrichment.on_workflow_execute(FakeWorkflowInfo(), args=())
    assert patched_current_span.attributes[LANGFUSE_SESSION_ID] == "wf-123"
