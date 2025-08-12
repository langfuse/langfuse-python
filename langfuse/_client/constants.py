"""Constants used by the Langfuse OpenTelemetry integration.

This module defines constants used throughout the Langfuse OpenTelemetry integration.
"""

LANGFUSE_TRACER_NAME = "langfuse-sdk"

# Valid observation types for the @observe decorator
VALID_OBSERVATION_TYPES = {
    "span",
    "event",
    "generation",
    "agent",
    "tool",
    "chain",
    "retriever",
    "embedding",
}
