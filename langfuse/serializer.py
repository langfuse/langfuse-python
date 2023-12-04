from datetime import date, datetime, timezone
from json import JSONEncoder
import logging
from typing import Any
from langchain.schema.messages import Serializable

from pydantic import BaseModel

log = logging.getLogger("langfuse")


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

        return JSONEncoder.default(self, obj)
