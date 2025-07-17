""".. include:: ../README.md"""

from ._client.client import Langfuse  # noqa
from ._client.get_client import get_client  # noqa
from ._client.observe import observe  # noqa
from .version import __version__  # noqa
from ._client.span import LangfuseSpan, LangfuseGeneration, LangfuseEvent
from ._client.attributes import LangfuseOtelSpanAttributes

__all__ = [
    "Langfuse",
    "get_client",
    "observe",
    "LangfuseSpan",
    "LangfuseGeneration",
    "LangfuseEvent",
    "LangfuseOtelSpanAttributes",
]
