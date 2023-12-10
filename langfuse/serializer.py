from datetime import date, datetime, timezone
from json import JSONEncoder
from typing import Any
from langchain.load.serializable import Serializable

from pydantic import BaseModel


class EventSerializer(JSONEncoder):
    def default(self, obj: Any):
        if isinstance(obj, (date, datetime)):
            # make the datetime object timezone-aware if it isn't already
            if obj.tzinfo is None or obj.tzinfo.utcoffset(obj) is None:
                obj = obj.replace(tzinfo=timezone.utc)
            return obj.isoformat()
        if isinstance(obj, BaseModel):
            return obj.dict()
        if isinstance(obj, Serializable):
            return obj.to_json()

        # If the object is of a type that the JSON encoder can handle, return the original object
        if isinstance(obj, (dict, list, str, int, float, type(None))):
            return obj

        return JSONEncoder.default(self, obj)
