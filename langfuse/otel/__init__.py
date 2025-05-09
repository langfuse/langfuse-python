"""Langfuse OpenTelemetry (OTEL) integration.

This module provides the core OpenTelemetry-based instrumentation for Langfuse,
enabling standardized observability for AI/LLM applications.

The OpenTelemetry integration offers:
- W3C-compliant distributed tracing
- Specialized span types for AI/LLM operations
- Automatic context propagation
- Efficient batching and background processing
- Sampling and filtering capabilities

Classes:
    Langfuse: The main client for Langfuse observability using OpenTelemetry

Functions:
    observe: Decorator for automatic tracing of functions and methods
    get_client: Retrieves or creates a Langfuse client instance
"""

from ._client import Langfuse
from ._get_client import get_client
from ._observe import observe

__all__ = ["Langfuse", "observe", "get_client"]
