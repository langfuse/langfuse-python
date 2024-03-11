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
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.seen = set()  # Track seen objects to detect circular references

    def default(self, obj: Any):
        if isinstance(obj, (datetime)):
            # Timezone-awareness check
            return serialize_datetime(obj)

        # LlamaIndex StreamingAgentChatResponse and StreamingResponse is not serializable by default as it is a generator
        # Attention: These LlamaIndex objects are a also a dataclasses, so check for it first
        if "Streaming" in type(obj).__name__:
            return str(obj)

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

        if isinstance(obj, (tuple, set, frozenset)):
            return list(obj)

        if hasattr(obj, "__slots__"):
            return self.default(
                {slot: getattr(obj, slot, None) for slot in obj.__slots__}
            )
        elif hasattr(obj, "__dict__"):
            obj_id = id(obj)

            if obj_id in self.seen:
                # Break on circular references
                return type(obj).__name__
            else:
                self.seen.add(obj_id)
                result = {k: self.default(v) for k, v in vars(obj).items()}
                self.seen.remove(obj_id)

                return result

        else:
            # Return object type rather than JSONEncoder.default(obj) which simply raises a TypeError
            return type(obj).__name__

    def encode(self, obj: Any) -> str:
        self.seen.clear()  # Clear seen objects before each encode call
        return super().encode(obj)
