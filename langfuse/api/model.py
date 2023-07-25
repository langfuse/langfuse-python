from langfuse.api.resources.event.types.create_event_request import CreateEventRequest
from langfuse.api.resources.generations.types.create_log import CreateLog
from langfuse.api.resources.generations.types.llm_usage import LlmUsage
from langfuse.api.resources.generations.types.update_generation_request import UpdateGenerationRequest
from langfuse.api.resources.score.types.create_score_request import CreateScoreRequest
from langfuse.api.resources.span.types.create_span_request import CreateSpanRequest
from langfuse.api.resources.span.types.update_span_request import UpdateSpanRequest
from langfuse.api.resources.trace.types.create_trace_request import CreateTraceRequest


class Generation(CreateLog):
    __fields__ = {name: field for name, field in CreateLog.__fields__.items()}


class Span(CreateLog):
    __fields__ = {name: field for name, field in CreateLog.__fields__.items()}


class CreateScore(CreateScoreRequest):
    __fields__ = {name: field for name, field in CreateScoreRequest.__fields__.items() if name not in ["trace_id", "observation_id"]}


class CreateTrace(CreateTraceRequest):
    __fields__ = {name: field for name, field in CreateTraceRequest.__fields__.items()}


class CreateGeneration(CreateLog):
    __fields__ = {name: field for name, field in CreateLog.__fields__.items() if name not in ["trace_id", "parent_observation_id", "traceIdType"]}


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
