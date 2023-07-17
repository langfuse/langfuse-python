from enum import Enum


class TraceIdTypeGenerations(str, Enum):
    EXTERNAL = "EXTERNAL"
    LANGFUSE = "LANGFUSE"

    def __str__(self) -> str:
        return str(self.value)
