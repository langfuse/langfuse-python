""".. include:: ../README.md"""

from ._client.attributes import LangfuseOtelSpanAttributes
from ._client.get_client import get_client
from ._client import client as _client
from ._client.observe import observe
from ._client.span import LangfuseEvent, LangfuseGeneration, LangfuseSpan

Langfuse = _client.Langfuse

__all__ = [
    "Langfuse",
    "get_client",
    "observe",
    "LangfuseSpan",
    "LangfuseGeneration",
    "LangfuseEvent",
    "LangfuseOtelSpanAttributes",
]
