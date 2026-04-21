"""Unit tests for the payload redaction / size-limit helper.

These tests do not require ``temporalio`` to be installed.
"""

from __future__ import annotations

import json

import pytest

from langfuse.temporal.redaction import prepare_payload


@pytest.mark.unit
def test_prepare_payload_serializes_primitive_types():
    assert prepare_payload({"a": 1}) == '{"a": 1}'
    assert prepare_payload([1, 2, 3]) == "[1, 2, 3]"
    assert prepare_payload("hello") == '"hello"'


@pytest.mark.unit
def test_prepare_payload_coerces_non_json_values():
    class Opaque:
        def __repr__(self) -> str:
            return "<Opaque>"

    result = prepare_payload(Opaque())
    assert result is not None
    assert "Opaque" in result


@pytest.mark.unit
def test_prepare_payload_honours_size_limit():
    big = {"k": "x" * 10_000}
    result = prepare_payload(big, size_limit_bytes=256)
    assert result is not None
    assert len(result.encode("utf-8")) <= 256 + len("…[truncated]".encode("utf-8"))
    assert result.endswith("…[truncated]")


@pytest.mark.unit
def test_prepare_payload_runs_redact_before_serialize():
    seen = {}

    def redact(value):
        seen["value"] = value
        return {"masked": True}

    result = prepare_payload({"secret": "hunter2"}, redact=redact)
    assert seen["value"] == {"secret": "hunter2"}
    assert json.loads(result) == {"masked": True}


@pytest.mark.unit
def test_prepare_payload_drops_when_redact_returns_none():
    assert prepare_payload({"secret": "x"}, redact=lambda v: None) is None


@pytest.mark.unit
def test_prepare_payload_never_raises_on_bad_redact():
    def bad(value):
        raise ValueError("boom")

    result = prepare_payload({"x": 1}, redact=bad)
    assert result == "[redaction-error]"


@pytest.mark.unit
def test_prepare_payload_handles_pydantic_like_objects():
    class FakeModel:
        def __init__(self) -> None:
            self.foo = "bar"

        def model_dump(self):
            return {"foo": "bar"}

    out = prepare_payload(FakeModel())
    assert json.loads(out) == {"foo": "bar"}
