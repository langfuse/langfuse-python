from datetime import date, datetime
from dataclasses import is_dataclass, asdict
from json import JSONEncoder
from typing import Any
from uuid import UUID

from langfuse.api.core import serialize_datetime

from pydantic import BaseModel

# Attempt to import Serializable
try:
    from langchain.load.serializable import Serializable
except ImportError:
    # If Serializable is not available, set it to NoneType
    Serializable = type(None)


class EventSerializer(JSONEncoder):
    def default(self, obj: Any):
        if isinstance(obj, (datetime)):
            # Timezone-awareness check
            return serialize_datetime(obj)
        if is_dataclass(obj):
            return asdict(obj)
        if isinstance(obj, UUID):
            return str(obj)
        if isinstance(obj, bytes):
            return obj.decode("utf-8")
        if isinstance(obj, (date)):
            return obj.isoformat()
        if isinstance(obj, BaseModel):
            return obj.dict()
        # if langchain is not available, the Serializable type is NoneType
        if Serializable is not None and isinstance(obj, Serializable):
            return obj.to_json()

        # Standard JSON-encodable types
        if isinstance(obj, (dict, list, str, int, float, type(None))):
            return obj

        if hasattr(obj, "__slots__"):
            return self.default(
                {slot: getattr(obj, slot, None) for slot in obj.__slots__}
            )
        elif hasattr(obj, "__dict__"):
            return self.default(vars(obj))
        else:
            return JSONEncoder.default(self, obj)
