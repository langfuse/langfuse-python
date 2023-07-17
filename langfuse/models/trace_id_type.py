from enum import Enum


class TraceIdType(str, Enum):
    EXTERNAL = "EXTERNAL"
    LANGFUSE = "LANGFUSE"

    def __str__(self) -> str:
        return str(self.value)
