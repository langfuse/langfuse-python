"""Constants used by the Langfuse OpenTelemetry integration.

This module defines constants used throughout the Langfuse OpenTelemetry integration.
"""

import enum
from typing import Literal
from typing_extensions import TypeAlias

LANGFUSE_TRACER_NAME = "langfuse-sdk"


class ObservationType(str, enum.Enum):
    """Enumeration of valid observation types for Langfuse tracing.

    This enum defines all the observation types that can be used with the @observe
    decorator and other Langfuse SDK methods.
    """

    SPAN = "SPAN"
    GENERATION = "GENERATION"
    AGENT = "AGENT"
    TOOL = "TOOL"
    CHAIN = "CHAIN"
    RETRIEVER = "RETRIEVER"
    EMBEDDING = "EMBEDDING"
    EVALUATOR = "EVALUATOR"
    GUARDRAIL = "GUARDRAIL"


ObservationTypeLiteralNoEvent: TypeAlias = Literal[
    "span",
    "generation",
    "agent",
    "tool",
    "chain",
    "retriever",
    "evaluator",
    "embedding",
    "guardrail",
]

ObservationTypeLiteral: TypeAlias = ObservationTypeLiteralNoEvent | Literal["event"]
