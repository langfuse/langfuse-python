"""Constants used by the Langfuse OpenTelemetry integration.

This module defines constants used throughout the Langfuse OpenTelemetry integration.
"""

LANGFUSE_TRACER_NAME = "langfuse-sdk"

# Valid observation types for the @observe decorator
VALID_OBSERVATION_TYPES = {
    "SPAN",
    "EVENT",
    "GENERATION",
    "AGENT",
    "TOOL",
    "CHAIN",
    "RETRIEVER",
    "EMBEDDING",
}
