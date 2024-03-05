from collections.abc import Sequence
from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from json import JSONEncoder
from typing import Any
from uuid import UUID

from google.protobuf.json_format import MessageToJson
from google.protobuf.message import Message
from pydantic import BaseModel

from langfuse.api.core import serialize_datetime

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

        # LlamaIndex StreamingAgentChatResponse is not serializable by default as it is a generator
        # Attention: StreamingAgentChatResponse is a also a dataclass, so check for it first
        if type(obj).__name__ == "StreamingAgentChatResponse":
            return str(obj)

        if is_dataclass(obj):
            return asdict(obj)
        if isinstance(obj, Message):
            return MessageToJson(obj)
        if isinstance(obj, Sequence) and not isinstance(obj, (str, bytes)):
            return [self.default(item) for item in obj]
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
