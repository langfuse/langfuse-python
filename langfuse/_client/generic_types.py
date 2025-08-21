"""Advanced generic typing system for Langfuse SDK.

This module provides TypeScript-inspired generic typing that eliminates the need
for repetitive @overload methods while maintaining perfect type safety and IDE support.

The system uses TypeVar + Literal mapping to provide precise return types based
on the as_type parameter, enabling single method signatures that return the correct
typed object for each observation type.
"""

from typing import TYPE_CHECKING, Any, Dict, Literal, Type, TypeVar, Union, overload
from typing_extensions import TypedDict

if TYPE_CHECKING:
    # Import observation classes only for type checking to avoid circular imports
    from langfuse._client.span import (
        LangfuseAgent,
        LangfuseChain,
        LangfuseEmbedding,
        LangfuseEvaluator,
        LangfuseGeneration,
        LangfuseGuardrail,
        LangfuseObservationWrapper,
        LangfuseRetriever,
        LangfuseSpan,
        LangfuseTool,
    )

from langfuse._client.constants import ObservationTypeLiteralNoEvent


# Type mapping system - maps observation type strings to their return types
class ObservationTypeMap(TypedDict, total=False):
    """Type mapping dictionary for observation types to their return types."""
    span: "LangfuseSpan"
    generation: "LangfuseGeneration"
    agent: "LangfuseAgent"
    tool: "LangfuseTool"
    chain: "LangfuseChain"
    retriever: "LangfuseRetriever"
    evaluator: "LangfuseEvaluator"
    embedding: "LangfuseEmbedding"
    guardrail: "LangfuseGuardrail"


# TypeVar constrained to valid observation types
ObservationType = TypeVar(
    "ObservationType", 
    bound=ObservationTypeLiteralNoEvent
)


# Generic type for methods that return observation objects based on as_type
def get_observation_return_type(as_type: str) -> Type["LangfuseObservationWrapper"]:
    """Get the return type for a given observation type string.
    
    This function provides runtime type resolution for the factory pattern.
    
    Args:
        as_type: The observation type string
        
    Returns:
        The corresponding observation class type
    """
    # Import classes at runtime to avoid circular imports
    from langfuse._client.span import (
        LangfuseAgent,
        LangfuseChain,
        LangfuseEmbedding,
        LangfuseEvaluator,
        LangfuseGeneration,
        LangfuseGuardrail,
        LangfuseObservationWrapper,
        LangfuseRetriever,
        LangfuseSpan,
        LangfuseTool,
    )
    
    type_map = {
        "span": LangfuseSpan,
        "generation": LangfuseGeneration,
        "agent": LangfuseAgent,
        "tool": LangfuseTool,
        "chain": LangfuseChain,
        "retriever": LangfuseRetriever,
        "evaluator": LangfuseEvaluator,
        "embedding": LangfuseEmbedding,
        "guardrail": LangfuseGuardrail,
    }
    
    return type_map.get(as_type, LangfuseObservationWrapper)


# Overload preservation for IDE support while using generic implementation
@overload
def start_observation_typed(
    as_type: Literal["span"], **kwargs: Any
) -> "LangfuseSpan": ...

@overload  
def start_observation_typed(
    as_type: Literal["generation"], **kwargs: Any
) -> "LangfuseGeneration": ...

@overload
def start_observation_typed(
    as_type: Literal["agent"], **kwargs: Any
) -> "LangfuseAgent": ...

@overload
def start_observation_typed(
    as_type: Literal["tool"], **kwargs: Any
) -> "LangfuseTool": ...

@overload
def start_observation_typed(
    as_type: Literal["chain"], **kwargs: Any
) -> "LangfuseChain": ...

@overload
def start_observation_typed(
    as_type: Literal["retriever"], **kwargs: Any
) -> "LangfuseRetriever": ...

@overload
def start_observation_typed(
    as_type: Literal["evaluator"], **kwargs: Any
) -> "LangfuseEvaluator": ...

@overload
def start_observation_typed(
    as_type: Literal["embedding"], **kwargs: Any
) -> "LangfuseEmbedding": ...

@overload
def start_observation_typed(
    as_type: Literal["guardrail"], **kwargs: Any
) -> "LangfuseGuardrail": ...

def start_observation_typed(
    as_type: ObservationTypeLiteralNoEvent, **kwargs: Any
) -> Union[
    "LangfuseSpan",
    "LangfuseGeneration", 
    "LangfuseAgent",
    "LangfuseTool",
    "LangfuseChain",
    "LangfuseRetriever",
    "LangfuseEvaluator",
    "LangfuseEmbedding",
    "LangfuseGuardrail",
]:
    """Generic typed function for creating observations.
    
    This demonstrates the pattern of maintaining overload signatures for IDE support
    while providing a single implementation. The factory can delegate to this pattern.
    
    Args:
        as_type: The observation type to create
        **kwargs: Additional arguments for the observation
        
    Returns:
        An observation of the appropriate type based on as_type
    """
    # This would delegate to the actual factory implementation
    # Implementation details handled by the actual client method
    pass  # pragma: no cover


# Score overload pattern - simplified to just 2 overloads instead of duplicating everywhere
@overload
def score_typed(
    *, name: str, value: float, **kwargs: Any
) -> None: ...

@overload
def score_typed(
    *, name: str, value: str, **kwargs: Any
) -> None: ...

def score_typed(
    *, name: str, value: Union[float, str], **kwargs: Any
) -> None:
    """Generic typed function for creating scores.
    
    This demonstrates the pattern for score methods - just 2 overloads
    instead of duplicating them everywhere.
    
    Args:
        name: Score name
        value: Score value (float or string)
        **kwargs: Additional score arguments
    """
    # Implementation details handled by the actual client method
    pass  # pragma: no cover