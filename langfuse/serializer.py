from datetime import date, datetime, timezone
from typing import Any
from json import JSONEncoder
from pydantic import BaseModel

# Attempt to import Serializable
try:
    from langchain.load.serializable import Serializable
except ImportError:
    # If Serializable is not available, set it to NoneType
    Serializable = type(None)


class EventSerializer(JSONEncoder):
    def default(self, obj: Any):
        if isinstance(obj, (date, datetime)):
            # Timezone-awareness check
            if obj.tzinfo is None or obj.tzinfo.utcoffset(obj) is None:
                obj = obj.replace(tzinfo=timezone.utc)
            return obj.isoformat()
        if isinstance(obj, BaseModel):
            return obj.dict()
        # if langchain is not available, the Serializable type is NoneType
        if isinstance(obj, Serializable):
            return obj.to_json()

        # Standard JSON-encodable types
        if isinstance(obj, (dict, list, str, int, float, type(None))):
            return obj

        return JSONEncoder.default(self, obj)
