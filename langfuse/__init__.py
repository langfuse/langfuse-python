""".. include:: ../README.md"""

from langfuse.batch_evaluation import (
    BatchEvaluationResult,
    BatchEvaluationResumeToken,
    CompositeEvaluatorFunction,
    EvaluatorInputs,
    EvaluatorStats,
    MapperFunction,
)
from langfuse.experiment import Evaluation

from ._client import client as _client_module
from ._client.attributes import LangfuseOtelSpanAttributes
from ._client.constants import ObservationTypeLiteral
from ._client.get_client import get_client
from ._client.observe import observe
from ._client.propagation import propagate_attributes
from ._client.span import (
    LangfuseAgent,
    LangfuseChain,
    LangfuseEmbedding,
    LangfuseEvaluator,
    LangfuseEvent,
    LangfuseGeneration,
    LangfuseGuardrail,
    LangfuseRetriever,
    LangfuseSpan,
    LangfuseTool,
)
from .span_filter import (
    KNOWN_LLM_INSTRUMENTATION_SCOPES,
    KNOWN_LLM_INSTRUMENTATION_SCOPE_PREFIXES,
    is_default_export_span,
    is_genai_span,
    is_known_llm_instrumentor,
    is_langfuse_span,
)

Langfuse = _client_module.Langfuse

__all__ = [
    "Langfuse",
    "get_client",
    "observe",
    "propagate_attributes",
    "ObservationTypeLiteral",
    "LangfuseSpan",
    "LangfuseGeneration",
    "LangfuseEvent",
    "LangfuseOtelSpanAttributes",
    "LangfuseAgent",
    "LangfuseTool",
    "LangfuseChain",
    "LangfuseEmbedding",
    "LangfuseEvaluator",
    "LangfuseRetriever",
    "LangfuseGuardrail",
    "Evaluation",
    "EvaluatorInputs",
    "MapperFunction",
    "CompositeEvaluatorFunction",
    "EvaluatorStats",
    "BatchEvaluationResumeToken",
    "BatchEvaluationResult",
    "is_default_export_span",
    "is_langfuse_span",
    "is_genai_span",
    "is_known_llm_instrumentor",
    "KNOWN_LLM_INSTRUMENTATION_SCOPES",
    "KNOWN_LLM_INSTRUMENTATION_SCOPE_PREFIXES",
    "experiment",
    "api",
]
