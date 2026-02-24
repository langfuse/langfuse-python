"""Public API for all Langfuse types.

This module provides a centralized location for importing commonly used types
from the Langfuse SDK, making them easily accessible without requiring nested imports.

Example:
    ```python
    from langfuse.types import Evaluation, LocalExperimentItem, TaskFunction

    # Define your task function
    def my_task(*, item: LocalExperimentItem, **kwargs) -> str:
        return f"Processed: {item['input']}"

    # Define your evaluator
    def my_evaluator(*, output: str, **kwargs) -> Evaluation:
        return {"name": "length", "value": len(output)}
    ```
"""

from typing import (
    Any,
    Dict,
    Literal,
    Protocol,
    TypedDict,
)

try:
    from typing import NotRequired  # type: ignore
except ImportError:
    from typing_extensions import NotRequired


from langfuse.api import MediaContentType

SpanLevel = Literal["DEBUG", "DEFAULT", "WARNING", "ERROR"]

ScoreDataType = Literal["NUMERIC", "CATEGORICAL", "BOOLEAN"]


class MaskFunction(Protocol):
    """A function that masks data.

    Keyword Args:
        data: The data to mask.

    Returns:
        The masked data that must be serializable to JSON.
    """

    def __call__(self, *, data: Any, **kwargs: Dict[str, Any]) -> Any: ...


class ParsedMediaReference(TypedDict):
    """A parsed media reference.

    Attributes:
        media_id: The media ID.
        source: The original source of the media, e.g. a file path, bytes, base64 data URI, etc.
        content_type: The content type of the media.
    """

    media_id: str
    source: str
    content_type: MediaContentType


class TraceContext(TypedDict):
    trace_id: str
    parent_span_id: NotRequired[str]


__all__ = [
    "SpanLevel",
    "ScoreDataType",
    "MaskFunction",
    "ParsedMediaReference",
    "TraceContext",
]
