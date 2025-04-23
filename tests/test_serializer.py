from datetime import datetime, date, timezone
from uuid import UUID
from enum import Enum
from dataclasses import dataclass
from pathlib import Path
from pydantic import BaseModel
import json
import pytest
import threading
import langfuse.serializer
from langfuse.serializer import (
    EventSerializer,
)


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


def test_none_without_langchain(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(langfuse.serializer, "Serializable", type(None), raising=True)
    serializer = EventSerializer()
    assert serializer.encode(None) == "null"


def test_slots():
    class SlotClass:
        __slots__ = ["field"]

        def __init__(self):
            self.field = "value"

    obj = SlotClass()
    serializer = EventSerializer()
    assert json.loads(serializer.encode(obj)) == {"field": "value"}


def test_numpy_float32():
    import numpy as np

    data = np.float32(1.0)
    serializer = EventSerializer()

    assert serializer.encode(data) == "1.0"
