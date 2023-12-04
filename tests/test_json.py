from datetime import datetime, timezone
import json
from typing import List
from langchain.schema.messages import HumanMessage
from pydantic import BaseModel

from langfuse.serializer import EventSerializer


def test_json_encoder():
    """Test that the JSON encoder encodes datetimes correctly."""

    class TestModel(BaseModel):
        foo: str
        bar: datetime
        messages: List[HumanMessage]

    message = HumanMessage(content="I love programming!")
    obj = {"foo": "bar", "bar": datetime(2021, 1, 1, 0, 0, 0, tzinfo=timezone.utc), "messages": [message]}

    # model = TestModel(foo="bar", bar=datetime(2021, 1, 1, 0, 0, 0, tzinfo=timezone.utc), messages=[message])

    assert (
        json.dumps(obj, cls=EventSerializer)
        == '{"foo": "bar", "bar": "2021-01-01T00:00:00+00:00", "messages": [{"lc": 1, "type": "constructor", "id": ["langchain", "schema", "messages", "HumanMessage"], "kwargs": {"content": "I love programming!"}}]}'
    )
