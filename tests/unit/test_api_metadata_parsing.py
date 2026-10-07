"""Response parsing of read-side ``metadata`` on generated API models.

The server returns read-side metadata as a JSON object, ``null``, or omits it.
These tests pin that contract so a future regeneration that makes the field
required or narrows its values is caught.
"""

import typing

import pydantic
import pytest

from langfuse.api import (
    BaseScore,
    BaseScoreV1,
    GetScoresResponseData,
    Observation,
    ObservationV2,
    Score,
    ScoreV1,
    Trace,
)
from langfuse.api.core import parse_obj_as

TIMESTAMP = "2026-01-01T00:00:00.000Z"

SCORE_BASE: typing.Dict[str, typing.Any] = {
    "id": "score-1",
    "name": "quality",
    "source": "API",
    "timestamp": TIMESTAMP,
    "createdAt": TIMESTAMP,
    "updatedAt": TIMESTAMP,
    "environment": "default",
}
SCORE_V1_BASE = {**SCORE_BASE, "traceId": "trace-1"}

SCORE_VARIANTS: typing.Dict[str, typing.Dict[str, typing.Any]] = {
    "NUMERIC": {"value": 0.5},
    "CATEGORICAL": {"value": 1, "stringValue": "good"},
    "BOOLEAN": {"value": 1, "stringValue": "True"},
    "CORRECTION": {"value": 0, "stringValue": "corrected output"},
    "TEXT": {"stringValue": "free text"},
}
SCORE_V1_DATA_TYPES = ["NUMERIC", "CATEGORICAL", "BOOLEAN", "TEXT"]


def _score_payload(
    base: typing.Dict[str, typing.Any], data_type: str
) -> typing.Dict[str, typing.Any]:
    return {**base, "dataType": data_type, **SCORE_VARIANTS[data_type]}


CASES: typing.List[typing.Any] = [
    pytest.param(
        Trace,
        {
            "id": "trace-1",
            "timestamp": TIMESTAMP,
            "tags": [],
            "public": False,
            "environment": "default",
        },
        id="Trace",
    ),
    pytest.param(
        Observation,
        {
            "id": "obs-1",
            "type": "SPAN",
            "startTime": TIMESTAMP,
            "modelParameters": {},
            "input": None,
            "output": None,
            "usage": {"input": 0, "output": 0, "total": 0},
            "level": "DEFAULT",
            "usageDetails": {},
            "costDetails": {},
            "environment": "default",
        },
        id="Observation",
    ),
    pytest.param(
        ObservationV2,
        {
            "id": "obs-1",
            "startTime": TIMESTAMP,
            "projectId": "project-1",
            "type": "SPAN",
        },
        id="ObservationV2",
    ),
    pytest.param(BaseScore, SCORE_BASE, id="BaseScore"),
    pytest.param(BaseScoreV1, SCORE_V1_BASE, id="BaseScoreV1"),
    *[
        pytest.param(Score, _score_payload(SCORE_BASE, dt), id=f"Score-{dt}")
        for dt in SCORE_VARIANTS
    ],
    *[
        pytest.param(
            GetScoresResponseData,
            _score_payload(SCORE_BASE, dt),
            id=f"GetScoresResponseData-{dt}",
        )
        for dt in SCORE_VARIANTS
    ],
    *[
        pytest.param(ScoreV1, _score_payload(SCORE_V1_BASE, dt), id=f"ScoreV1-{dt}")
        for dt in SCORE_V1_DATA_TYPES
    ],
]

NESTED_METADATA = {
    "str": "value",
    "int": 1,
    "float": 1.5,
    "bool": True,
    "null": None,
    "list": [1, "two", {"three": 3}],
    "nested": {"deeper": {"key": ["a", "b"]}},
}


@pytest.mark.parametrize(("type_", "payload"), CASES)
def test_metadata_omitted_parses_as_none(type_, payload):
    assert "metadata" not in payload

    parsed = parse_obj_as(type_, payload)

    assert parsed.metadata is None


@pytest.mark.parametrize(("type_", "payload"), CASES)
def test_metadata_null_parses_as_none(type_, payload):
    parsed = parse_obj_as(type_, {**payload, "metadata": None})

    assert parsed.metadata is None


@pytest.mark.parametrize(("type_", "payload"), CASES)
def test_metadata_empty_object_parses(type_, payload):
    parsed = parse_obj_as(type_, {**payload, "metadata": {}})

    assert parsed.metadata == {}


@pytest.mark.parametrize(("type_", "payload"), CASES)
def test_metadata_object_with_nested_json_values_parses(type_, payload):
    parsed = parse_obj_as(type_, {**payload, "metadata": NESTED_METADATA})

    assert parsed.metadata == NESTED_METADATA


@pytest.mark.parametrize(("type_", "payload"), CASES)
@pytest.mark.parametrize(
    "metadata",
    ["a string", 1, 1.5, True, ["a", "list"]],
    ids=["str", "int", "float", "bool", "list"],
)
def test_metadata_non_object_is_rejected(type_, payload, metadata):
    with pytest.raises(pydantic.ValidationError) as exc_info:
        parse_obj_as(type_, {**payload, "metadata": metadata})

    # The discriminated unions prefix the location with the variant tag.
    assert exc_info.value.errors()
    assert all("metadata" in error["loc"] for error in exc_info.value.errors())
