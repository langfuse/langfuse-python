from enum import Enum


class TraceIdTypeEvent(str, Enum):
    EXTERNAL = "EXTERNAL"
    LANGFUSE = "LANGFUSE"

    def __str__(self) -> str:
        return str(self.value)
