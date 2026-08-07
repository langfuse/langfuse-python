import json
import threading
from dataclasses import dataclass
from datetime import date, datetime, timezone
from decimal import Decimal
from enum import Enum
from pathlib import Path
from uuid import UUID

import pytest
from pydantic import BaseModel

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


def test_decimal_is_preserved_exactly():
    # Serialized to its exact string form (never the "<Decimal>" fallback)
    assert json.loads(EventSerializer().encode(Decimal("19.99"))) == "19.99"
    assert json.loads(EventSerializer().encode({"price": Decimal("19.99")})) == {
        "price": "19.99"
    }
    # High-precision values are preserved exactly (a float() conversion would
    # silently round these).
    assert (
        json.loads(EventSerializer().encode(Decimal("1.0000000000000001")))
        == "1.0000000000000001"
    )
    assert (
        json.loads(EventSerializer().encode(Decimal("123456789012345678")))
        == "123456789012345678"
    )


def test_decimal_special_values():
    assert EventSerializer().encode(Decimal("NaN")) == '"NaN"'
    assert EventSerializer().encode(Decimal("Infinity")) == '"Infinity"'
    assert EventSerializer().encode(Decimal("-Infinity")) == '"-Infinity"'


def test_dict_with_non_primitive_keys_preserves_values():
    # A tuple key must not discard the entire dict (previously the whole dict
    # serialized to a "<not serializable ...>" string).
    result = json.loads(EventSerializer().encode({(1, 2): "v", "other": "data"}))
    assert result == {"(1, 2)": "v", "other": "data"}


def test_dict_with_custom_object_key_uses_str():
    class _Key:
        def __str__(self) -> str:
            return "custom-key"

    result = json.loads(EventSerializer().encode({_Key(): "val", "k2": "x"}))
    assert result == {"custom-key": "val", "k2": "x"}
