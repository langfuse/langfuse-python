"""Unit tests for the configuration dataclasses."""

from __future__ import annotations

import pytest

from langfuse.temporal.config import (
    CaptureConfig,
    FactoryContext,
    LangfusePluginConfig,
    TracingConfig,
)


@pytest.mark.unit
def test_defaults_are_metadata_only():
    cfg = LangfusePluginConfig()
    assert cfg.capture.capture_workflow_inputs is False
    assert cfg.capture.capture_workflow_outputs is False
    assert cfg.capture.capture_activity_inputs is False
    assert cfg.capture.capture_activity_outputs is False


@pytest.mark.unit
def test_tracing_defaults_disable_noisy_surfaces():
    cfg = TracingConfig()
    assert cfg.add_temporal_spans is True
    assert cfg.trace_signals is False
    assert cfg.trace_queries is False


@pytest.mark.unit
def test_capture_allowlist_denylist_logic():
    cap = CaptureConfig(
        workflow_allowlist=["OrderWorkflow"],
        activity_denylist=["ChargeCardActivity"],
    )

    assert cap.should_capture_workflow("OrderWorkflow") is True
    assert cap.should_capture_workflow("OtherWorkflow") is False
    # No activity allowlist → only denylist applies.
    assert cap.should_capture_activity("SendEmailActivity") is True
    assert cap.should_capture_activity("ChargeCardActivity") is False


@pytest.mark.unit
def test_capture_denylist_wins_over_allowlist():
    cap = CaptureConfig(
        workflow_allowlist=["OrderWorkflow", "SecretWorkflow"],
        workflow_denylist=["SecretWorkflow"],
    )
    assert cap.should_capture_workflow("SecretWorkflow") is False
    assert cap.should_capture_workflow("OrderWorkflow") is True


@pytest.mark.unit
def test_unknown_workflow_name_defaults_sane():
    cap = CaptureConfig()
    # Without any lists, unknown workflow names are captured (because
    # the flag would still gate capture at a higher level).
    assert cap.should_capture_workflow(None) is True
    cap_allow = CaptureConfig(workflow_allowlist=["A"])
    # Missing name + allowlist = default deny.
    assert cap_allow.should_capture_workflow(None) is False


@pytest.mark.unit
def test_factory_context_is_plain_dataclass():
    ctx = FactoryContext(kind="workflow", info=None, input=(1, 2))
    assert ctx.kind == "workflow"
    assert ctx.input == (1, 2)
