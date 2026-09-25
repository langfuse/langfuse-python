import json
import threading
from dataclasses import dataclass
from datetime import date, datetime, timezone
from enum import Enum
from pathlib import Path
from uuid import UUID

import pytest
from pydantic import BaseModel, SecretBytes, SecretStr

from langfuse._utils.serializer import (
    EventSerializer,
)
from langfuse.media import LangfuseMediaReference


class TestEnum(Enum):
    A = 1
    B = 2


@dataclass
class TestDataclass:
    field: str


class TestBaseModel(BaseModel):
    field: str


class SecretBaseModel(BaseModel):
    api_key: SecretStr
    token: SecretBytes


def test_datetime():
    dt = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    serializer = EventSerializer()

    assert serializer.encode(dt) == '"2023-01-01T12:00:00Z"'


def test_date():
    d = date(2023, 1, 1)
    serializer = EventSerializer()
    assert serializer.encode(d) == '"2023-01-01"'


def test_enum():
    serializer = EventSerializer()
    assert serializer.encode(TestEnum.A) == "1"


def test_uuid():
    uuid = UUID("123e4567-e89b-12d3-a456-426614174000")
    serializer = EventSerializer()
    assert serializer.encode(uuid) == '"123e4567-e89b-12d3-a456-426614174000"'


def test_bytes():
    b = b"hello"
    serializer = EventSerializer()
    assert serializer.encode(b) == '"hello"'


def test_dataclass():
    dc = TestDataclass(field="test")
    serializer = EventSerializer()
    assert json.loads(serializer.encode(dc)) == {"field": "test"}


def test_pydantic_model():
    model = TestBaseModel(field="test")
    serializer = EventSerializer()
    assert json.loads(serializer.encode(model)) == {"field": "test"}


@pytest.mark.parametrize(
    "secret",
    [
        SecretStr("not-a-real-api-key"),
        SecretBytes(b"not-a-real-token"),
    ],
)
def test_pydantic_secret(secret):
    serializer = EventSerializer()

    assert serializer.encode(secret) == '"<secret>"'


def test_pydantic_model_with_secrets():
    model = SecretBaseModel(
        api_key=SecretStr("not-a-real-api-key"),
        token=SecretBytes(b"not-a-real-token"),
    )
    serializer = EventSerializer()

    assert json.loads(serializer.encode(model)) == {
        "api_key": "<secret>",
        "token": "<secret>",
    }


def test_langfuse_media_reference_serializes_to_reference_string():
    # Resolved references must round-trip back to their original reference string
    # rather than falling through to asdict() and emitting an opaque dict.
    reference_string = "@@@langfuseMedia:type=image/png|id=media-id|source=bytes@@@"
    ref = LangfuseMediaReference(
        media_id="media-id",
        content_type="image/png",
        url="https://example.com/image.png",
        reference_string=reference_string,
    )
    serializer = EventSerializer()
    assert serializer.encode(ref) == f'"{reference_string}"'


def test_langfuse_media_reference_without_reference_string_falls_back_to_dict():
    ref = LangfuseMediaReference(
        media_id="media-id",
        content_type="image/png",
        url="https://example.com/image.png",
    )
    serializer = EventSerializer()
    assert json.loads(serializer.encode(ref))["media_id"] == "media-id"


def test_path():
    path = Path("/tmp/test.txt")
    serializer = EventSerializer()
    assert serializer.encode(path) == '"/tmp/test.txt"'


def test_tuple_set_frozenset():
    data = (1, 2, 3)
    serializer = EventSerializer()
    assert serializer.encode(data) == "[1, 2, 3]"

    data = {1, 2, 3}
    assert serializer.encode(data) == "[1, 2, 3]"

    data = frozenset([1, 2, 3])
    assert json.loads(serializer.encode(data)) == [1, 2, 3]


def test_dict():
    data = {"a": 1, "b": "two"}
    serializer = EventSerializer()

    assert json.loads(serializer.encode(data)) == data


def test_list():
    data = [1, "two", 3.0]
    serializer = EventSerializer()

    assert json.loads(serializer.encode(data)) == data


def test_nested_structures():
    data = {"list": [1, 2, 3], "dict": {"a": 1, "b": 2}, "tuple": (4, 5, 6)}
    serializer = EventSerializer()

    assert json.loads(serializer.encode(data)) == {
        "list": [1, 2, 3],
        "dict": {"a": 1, "b": 2},
        "tuple": [4, 5, 6],
    }


def test_custom_object():
    class CustomObject:
        def __init__(self):
            self.field = "value"

    obj = CustomObject()
    serializer = EventSerializer()

    assert json.loads(serializer.encode(obj)) == {"field": "value"}


def test_circular_reference():
    class Node:
        def __init__(self):
            self.next = None

    node1 = Node()
    node2 = Node()
    node1.next = node2
    node2.next = node1

    serializer = EventSerializer()
    result = json.loads(serializer.encode(node1))

    assert result == {"next": {"next": "Node"}}


def test_not_serializable():
    class NotSerializable:
        def __init__(self):
            self.lock = threading.Lock()

        def __repr__(self):
            raise Exception("Cannot represent")

    obj = NotSerializable()
    serializer = EventSerializer()

    assert serializer.encode(obj) == '{"lock": "<lock>"}'


def test_exception():
    ex = ValueError("Test exception")
    serializer = EventSerializer()
    assert serializer.encode(ex) == '"ValueError: Test exception"'


def test_none():
    serializer = EventSerializer()
    assert serializer.encode(None) == "null"


def test_infinity_floats():
    serializer = EventSerializer()
    assert serializer.encode(float("inf")) == '"Infinity"'
    assert serializer.encode(float("-inf")) == '"-Infinity"'


def _reject_json_constant(token):
    # json.loads accepts bare NaN/Infinity by default; reject them the way
    # the ingestion server's strict parser does.
    raise ValueError(f"invalid JSON constant emitted: {token}")


def _strict_loads(encoded: str):
    return json.loads(encoded, parse_constant=_reject_json_constant)


def test_pydantic_model_with_non_finite_floats():
    # Non-finite floats nested inside a pydantic model must be converted to
    # safe string tokens rather than emitted as bare NaN/Infinity, which are
    # invalid JSON and are rejected by the ingestion server's strict parser.
    class ModelWithFloats(BaseModel):
        nan: float
        inf: float
        neg_inf: float
        finite: float

    model = ModelWithFloats(
        nan=float("nan"),
        inf=float("inf"),
        neg_inf=float("-inf"),
        finite=1.5,
    )
    serializer = EventSerializer()
    parsed = _strict_loads(serializer.encode(model))
    assert parsed == {
        "nan": "NaN",
        "inf": "Infinity",
        "neg_inf": "-Infinity",
        "finite": 1.5,
    }


def test_tuple_set_frozenset_with_non_finite_floats():
    serializer = EventSerializer()

    assert _strict_loads(serializer.encode((float("nan"), 1.0, float("inf")))) == [
        "NaN",
        1.0,
        "Infinity",
    ]
    assert _strict_loads(serializer.encode({float("nan")})) == ["NaN"]
    assert _strict_loads(serializer.encode(frozenset([float("-inf")]))) == ["-Infinity"]


def test_dataclass_with_non_finite_floats():
    @dataclass
    class Point:
        x: float
        y: float

    serializer = EventSerializer()
    parsed = _strict_loads(serializer.encode(Point(float("nan"), float("inf"))))
    assert parsed == {"x": "NaN", "y": "Infinity"}


def test_enum_with_non_finite_float_value():
    class NonFiniteEnum(Enum):
        NAN = float("nan")
        INF = float("inf")

    serializer = EventSerializer()
    assert _strict_loads(serializer.encode(NonFiniteEnum.NAN)) == "NaN"
    assert _strict_loads(serializer.encode(NonFiniteEnum.INF)) == "Infinity"


def test_numpy_array_and_generic_with_non_finite_floats(monkeypatch):
    # Numpy is an optional runtime dependency; fake the types EventSerializer
    # checks for so the ndarray / generic branches are covered without numpy.
    class FakeGeneric:
        def __init__(self, value):
            self._value = value

        def item(self):
            return self._value

    class FakeNdarray:
        def __init__(self, data):
            self._data = data

        def tolist(self):
            return self._data

    class FakeNP:
        generic = FakeGeneric
        ndarray = FakeNdarray

    from langfuse._utils import serializer as serializer_mod

    monkeypatch.setattr(serializer_mod, "np", FakeNP)

    serializer = EventSerializer()
    assert _strict_loads(
        serializer.encode(FakeNdarray([float("nan"), 1.0, float("inf")]))
    ) == ["NaN", 1.0, "Infinity"]
    assert _strict_loads(
        serializer.encode(FakeNdarray([[float("nan"), 1.0], [float("-inf"), 2.0]]))
    ) == [["NaN", 1.0], ["-Infinity", 2.0]]
    assert _strict_loads(serializer.encode(FakeGeneric(float("nan")))) == "NaN"
    assert _strict_loads(serializer.encode(FakeGeneric(float("inf")))) == "Infinity"


def test_langchain_serializable_to_json_with_non_finite_floats(monkeypatch):
    # langchain_core.Serializable is a BaseModel, so real messages take the
    # pydantic path. Patch the type EventSerializer checks so the to_json()
    # branch is covered independently.
    class FakeSerializable:
        def to_json(self):
            return {"score": float("nan"), "ok": 1.5}

    from langfuse._utils import serializer as serializer_mod

    monkeypatch.setattr(serializer_mod, "Serializable", FakeSerializable)

    parsed = _strict_loads(EventSerializer().encode(FakeSerializable()))
    assert parsed == {"score": "NaN", "ok": 1.5}


def test_tuple_with_js_unsafe_integer():
    # The same conversion branches that leaked non-finite floats also skip
    # JS-safe integer coercion unless values are routed back through default().
    unsafe = 2**53
    serializer = EventSerializer()
    assert _strict_loads(serializer.encode((unsafe,))) == [str(unsafe)]


def test_slots():
    class SlotClass:
        __slots__ = ["field"]

        def __init__(self):
            self.field = "value"

    obj = SlotClass()
    serializer = EventSerializer()
    assert json.loads(serializer.encode(obj)) == {"field": "value"}


def test_deeply_nested_object_does_not_hang():
    class Inner:
        def __init__(self):
            self.lock = threading.Lock()
            self.value = "deep"

    class Connection:
        def __init__(self):
            self._inner = Inner()
            self._pool = [Inner() for _ in range(3)]

    class Client:
        def __init__(self):
            self._connection = Connection()
            self._config = {"key": "value"}

    class Platform:
        def __init__(self):
            self._client = Client()

    obj = {"args": (Platform(),), "kwargs": {}}
    serializer = EventSerializer()
    result = serializer.encode(obj)

    # Must complete without hanging and produce valid JSON
    parsed = json.loads(result)
    assert "args" in parsed


def test_max_depth_returns_type_name():
    class Level:
        def __init__(self, child=None):
            self.child = child

    # Build a chain deeper than _MAX_DEPTH
    obj = None
    for _ in range(EventSerializer._MAX_DEPTH + 10):
        obj = Level(child=obj)

    serializer = EventSerializer()
    result = json.loads(serializer.encode(obj))

    # Walk down the chain — at some point it should be truncated to "Level"
    node = result
    found_truncation = False
    while isinstance(node, dict) and "child" in node:
        if node["child"] == "Level" or node["child"] == "<Level>":
            found_truncation = True
            break
        node = node["child"]

    assert found_truncation, "Expected depth limit to truncate deep nesting"


def test_deeply_nested_slots_object_is_truncated():
    class SlotLevel:
        __slots__ = ["child"]

        def __init__(self, child=None):
            self.child = child

    obj = None
    for _ in range(EventSerializer._MAX_DEPTH + 10):
        obj = SlotLevel(child=obj)

    serializer = EventSerializer()
    result = json.loads(serializer.encode(obj))

    # Walk the nested structure and verify it terminates
    node = result
    depth = 0
    while isinstance(node, dict):
        depth += 1
        if "child" in node:
            node = node["child"]
        else:
            break

    assert EventSerializer._MAX_DEPTH - 2 <= depth <= EventSerializer._MAX_DEPTH + 2, (
        f"Nesting depth {depth} not near _MAX_DEPTH ({EventSerializer._MAX_DEPTH}) — "
        "serializer truncated too early or too late"
    )


def test_deeply_nested_dict_preserves_keys_at_depth_boundary(monkeypatch):
    monkeypatch.setattr(EventSerializer, "_MAX_DEPTH", 3)

    input_obj = {"a": {"b": {"c": "leaf"}}}
    expected = {"a": {"b": "<dict>"}}

    serializer = EventSerializer()
    result = json.loads(serializer.encode(input_obj))

    assert result == expected


class _Color(Enum):
    RED = "red"
    NUMERIC = 7


@pytest.mark.parametrize(
    "input_obj, expected",
    [
        (
            {datetime(2024, 1, 1, tzinfo=timezone.utc): "v"},
            {"2024-01-01T00:00:00Z": "v"},
        ),
        (
            {UUID("12345678-1234-5678-1234-567812345678"): "v"},
            {"12345678-1234-5678-1234-567812345678": "v"},
        ),
        ({_Color.RED: "v"}, {"red": "v"}),
        ({_Color.NUMERIC: "v"}, {"7": "v"}),
    ],
    ids=["datetime", "uuid", "enum_str_value", "enum_int_value"],
)
def test_dict_with_non_string_keys_is_serialized(input_obj, expected):
    result = json.loads(EventSerializer().encode(input_obj))

    assert result == expected
