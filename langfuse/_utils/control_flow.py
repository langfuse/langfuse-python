"""Utilities for detecting control-flow exceptions that should not be marked as errors.

LangGraph uses GraphBubbleUp subclasses (e.g. GraphInterrupt, NodeInterrupt) for
expected control flow such as human-in-the-loop interrupts and handoffs.  These are
not real errors and should be recorded at "DEFAULT" level in Langfuse, not "ERROR".
"""

from typing import Literal, Set, Type

CONTROL_FLOW_EXCEPTION_TYPES: Set[Type[BaseException]] = set()

try:
    from langgraph.errors import GraphBubbleUp

    CONTROL_FLOW_EXCEPTION_TYPES.add(GraphBubbleUp)
except ImportError:
    pass


def get_error_level(
    error: BaseException,
) -> Literal["DEFAULT", "ERROR"]:
    """Return the appropriate Langfuse observation level for *error*.

    Returns ``"DEFAULT"`` for known control-flow exceptions (e.g.
    ``GraphInterrupt``) and ``"ERROR"`` for everything else.
    """
    if any(isinstance(error, t) for t in CONTROL_FLOW_EXCEPTION_TYPES):
        return "DEFAULT"
    return "ERROR"
