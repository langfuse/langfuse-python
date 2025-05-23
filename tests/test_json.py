import builtins
import importlib
import json
import uuid
from dataclasses import dataclass
from datetime import date, datetime, timezone
from unittest.mock import patch

import pytest
from bson import ObjectId
from langchain.schema.messages import HumanMessage
from pydantic import BaseModel

import langfuse
from langfuse._utils.serializer import EventSerializer
from langfuse.api.resources.commons.types.observation_level import ObservationLevel


class TestModel(BaseModel):
    foo: str
    bar: datetime


def test_json_encoder():
    """Test that the JSON encoder encodes datetimes correctly."""
    message = HumanMessage(content="I love programming!")
    obj = {
        "foo": "bar",
        "bar": datetime(2021, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
        "date": date(2024, 1, 1),
        "messages": [message],
    }

    result = json.dumps(obj, cls=EventSerializer)
    assert (
        '{"foo": "bar", "bar": "2021-01-01T00:00:00Z", "date": "2024-01-01", "messages": [{"content": "I love programming!", "additional_kwargs": {}, "response_metadata": {}, "type": "human", "name": null, "id": null, "example": false}]}'
        in result
    )


def test_json_decoder_pydantic():
    obj = TestModel(foo="bar", bar=datetime(2021, 1, 1, 0, 0, 0, tzinfo=timezone.utc))
    assert (
        json.dumps(obj, cls=EventSerializer)
        == '{"foo": "bar", "bar": "2021-01-01T00:00:00Z"}'
    )


@pytest.fixture
def event_serializer():
    return EventSerializer()


def test_json_decoder_without_langchain_serializer():
    with patch.dict("sys.modules", {"langchain.load.serializable": None}):
        model = TestModel(
            foo="John", bar=datetime(2021, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        )
        result = json.dumps(model, cls=EventSerializer)
        assert result == '{"foo": "John", "bar": "2021-01-01T00:00:00Z"}'


@pytest.fixture
def hide_available_langchain(monkeypatch):
    import_orig = builtins.__import__

    def mocked_import(name, *args, **kwargs):
        if name == "langchain" or name == "langchain.load.serializable":
            raise ImportError()
        return import_orig(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mocked_import)


@pytest.mark.usefixtures("hide_available_langchain")
def test_json_decoder_without_langchain_serializer_with_langchain_message():
    with pytest.raises(ImportError):
        import langchain  # noqa

    with pytest.raises(ImportError):
        from langchain.load.serializable import Serializable  # noqa

    importlib.reload(langfuse)
    obj = TestModel(foo="bar", bar=datetime(2021, 1, 1, 0, 0, 0, tzinfo=timezone.utc))
    result = json.dumps(obj, cls=EventSerializer)
    assert result == '{"foo": "bar", "bar": "2021-01-01T00:00:00Z"}'


@pytest.mark.usefixtures("hide_available_langchain")
def test_json_decoder_without_langchain_serializer_with_none():
    with pytest.raises(ImportError):
        import langchain  # noqa

    with pytest.raises(ImportError):
        from langchain.load.serializable import Serializable  # noqa

    importlib.reload(langfuse)
    result = json.dumps(None, cls=EventSerializer)
    default = json.dumps(None)
    assert result == "null"
    assert result == default


def test_data_class():
    @dataclass
    class InventoryItem:
        """Class for keeping track of an item in inventory."""

        name: str
        unit_price: float
        quantity_on_hand: int = 0

    item = InventoryItem("widget", 3.0, 10)

    result = json.dumps(item, cls=EventSerializer)

    assert result == '{"name": "widget", "unit_price": 3.0, "quantity_on_hand": 10}'


def test_data_uuid():
    test_id = uuid.uuid4()

    result = json.dumps(test_id, cls=EventSerializer)

    assert result == f'"{str(test_id)}"'


def test_observation_level():
    result = json.dumps(ObservationLevel.ERROR, cls=EventSerializer)

    assert result == '"ERROR"'


def test_mongo_cursor():
    test_id = ObjectId("5f3e3e3e3e3e3e3e3e3e3e3e")

    result = json.dumps(test_id, cls=EventSerializer)

    assert isinstance(result, str)
