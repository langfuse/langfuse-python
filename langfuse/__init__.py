""".. include:: ../README.md"""

from ._client import client as _client_module
from ._client.attributes import LangfuseOtelSpanAttributes
from ._client.get_client import get_client
from ._client.observe import observe
from ._client.span import LangfuseEvent, LangfuseGeneration, LangfuseSpan

Langfuse = _client_module.Langfuse

__all__ = [
    "Langfuse",
    "get_client",
    "observe",
    "LangfuseSpan",
    "LangfuseGeneration",
    "LangfuseEvent",
    "LangfuseOtelSpanAttributes",
]
