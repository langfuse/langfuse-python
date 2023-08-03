from langfuse.api.resources.commons.types.create_generation_request import CreateGenerationRequest
from langfuse.api.resources.commons.types.create_event_request import CreateEventRequest
from langfuse.api.resources.commons.types.create_span_request import CreateSpanRequest
from langfuse.api.resources.commons.types.llm_usage import LlmUsage
from langfuse.api.resources.generations.types.update_generation_request import UpdateGenerationRequest
from langfuse.api.resources.score.types.create_score_request import CreateScoreRequest

from langfuse.api.resources.span.types.update_span_request import UpdateSpanRequest
from langfuse.api.resources.trace.types.create_trace_request import CreateTraceRequest

# these imports need to stay here, otherwise imports from our clients wont work
from langfuse.api.resources.commons.types.trace_id_type_enum import TraceIdTypeEnum
from langfuse.api.resources.commons.types.observation_level import ObservationLevel


class InitialGeneration(CreateGenerationRequest):
    pass


class InitialScoreRequest(CreateScoreRequest):
    __fields__ = {name: field for name, field in CreateScoreRequest.__fields__.items()}


class Span(CreateGenerationRequest):
    __fields__ = {name: field for name, field in CreateGenerationRequest.__fields__.items()}


class CreateScore(InitialScoreRequest):
    __fields__ = {name: field for name, field in InitialScoreRequest.__fields__.items() if name not in ["trace_id", "observation_id"]}


class CreateTrace(CreateTraceRequest):
    __fields__ = {name: field for name, field in CreateTraceRequest.__fields__.items() if name not in ["release"]}


class CreateGeneration(CreateGenerationRequest):
    __fields__ = {name: field for name, field in CreateGenerationRequest.__fields__.items() if name not in ["trace_id", "parent_observation_id", "traceIdType"]}


class CreateSpan(CreateSpanRequest):
    __fields__ = {name: field for name, field in CreateSpanRequest.__fields__.items() if name not in ["trace_id", "parent_observation_id", "traceIdType"]}


class CreateEvent(CreateEventRequest):
    __fields__ = {name: field for name, field in CreateEventRequest.__fields__.items() if name not in ["trace_id", "parent_observation_id", "traceIdType"]}


class UpdateGeneration(UpdateGenerationRequest):
    __fields__ = {name: field for name, field in UpdateGenerationRequest.__fields__.items() if name not in ["generation_id"]}


class UpdateSpan(UpdateSpanRequest):
    __fields__ = {name: field for name, field in UpdateSpanRequest.__fields__.items() if name not in ["spanId", "span_id"]}


class Usage(LlmUsage):
    __fields__ = {name: field for name, field in LlmUsage.__fields__.items()}
