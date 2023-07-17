from enum import Enum


class TraceIdTypeSpan(str, Enum):
    EXTERNAL = "EXTERNAL"
    LANGFUSE = "LANGFUSE"

    def __str__(self) -> str:
        return str(self.value)
