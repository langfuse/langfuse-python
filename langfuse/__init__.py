"""Langfuse Python SDK — observability, evaluation, and prompt management for LLM applications.

Capabilities:

- **Tracing / observability**: `@observe` decorator, `Langfuse.start_observation` /
  `start_as_current_observation` context managers, OpenTelemetry-based; integrations
  for OpenAI (`langfuse.openai`) and LangChain (`langfuse.langchain.CallbackHandler`).
- **Trace attributes**: `propagate_attributes` (top-level function) sets user_id,
  session_id, tags, and metadata on all spans in a context.
- **Datasets & experiments**: `Langfuse.get_dataset`, `Langfuse.run_experiment` for
  offline evaluation and regression testing of prompt/model changes (CI support via
  https://github.com/langfuse/experiment-action and `RegressionError`).
- **Evaluation / LLM-as-a-judge**: `Evaluation` results from custom or model-based
  evaluators; scores via `Langfuse.create_score` / `span.score`.
- **Prompt management**: `Langfuse.get_prompt`, `Langfuse.create_prompt` with
  client-side caching and version/label control.
- **Full REST API**: `Langfuse.api` (sync) / `Langfuse.async_api` (async) clients.

Quickstart:

```python
# env: LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_BASE_URL
from langfuse import get_client

langfuse = get_client()

# Create a span using a context manager
with langfuse.start_as_current_observation(as_type="span", name="process-request") as span:
    # Your processing logic here
    span.update(output="Processing complete")

    # Create a nested generation for an LLM call
    with langfuse.start_as_current_observation(as_type="generation", name="llm-response", model="gpt-3.5-turbo") as generation:
        # Your LLM call logic here
        generation.update(output="Generated response")

# All spans are automatically closed when exiting their context blocks

# Flush events in short-lived applications
langfuse.flush()
```

Configuration is via constructor args or environment variables: `LANGFUSE_PUBLIC_KEY`,
`LANGFUSE_SECRET_KEY`, `LANGFUSE_BASE_URL` (defaults to https://cloud.langfuse.com). See `langfuse._client.environment_variables`
for the full list.

Docs: https://langfuse.com/docs — machine-readable index: https://langfuse.com/llms.txt

.. include:: ../README.md
"""

from langfuse.batch_evaluation import (
    BatchEvaluationResult,
    BatchEvaluationResumeToken,
    CompositeEvaluatorFunction,
    EvaluatorInputs,
    EvaluatorStats,
    MapperFunction,
)
from langfuse.experiment import Evaluation, RegressionError, RunnerContext

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
from ._version import __version__
from .media import LangfuseMedia, LangfuseMediaReference
from .span_filter import (
    KNOWN_LLM_INSTRUMENTATION_SCOPE_PREFIXES,
    is_default_export_span,
    is_genai_span,
    is_known_llm_instrumentor,
    is_langfuse_span,
)
from .types import (
    MaskOtelSpansFunction,
    MaskOtelSpansParams,
    MaskOtelSpansResult,
    OtelSpanData,
    OtelSpanIdentifier,
    OtelSpanPatch,
)

Langfuse = _client_module.Langfuse
LangfuseAuthCheckError = _client_module.LangfuseAuthCheckError

__all__ = [
    "Langfuse",
    "LangfuseAuthCheckError",
    "LangfuseMedia",
    "LangfuseMediaReference",
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
    "RunnerContext",
    "RegressionError",
    "__version__",
    "is_default_export_span",
    "is_langfuse_span",
    "is_genai_span",
    "is_known_llm_instrumentor",
    "KNOWN_LLM_INSTRUMENTATION_SCOPE_PREFIXES",
    "MaskOtelSpansFunction",
    "MaskOtelSpansParams",
    "MaskOtelSpansResult",
    "OtelSpanData",
    "OtelSpanIdentifier",
    "OtelSpanPatch",
    "experiment",
    "api",
]
