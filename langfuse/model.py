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
from langfuse.api.resources.commons.types.dataset_item import DatasetItem
from langfuse.api.resources.commons.types.dataset import Dataset
from langfuse.api.resources.dataset_run_items.types.create_dataset_run_item_request import CreateDatasetRunItemRequest
from langfuse.api.resources.datasets.types.create_dataset_request import CreateDatasetRequest
from langfuse.api.resources.dataset_items.types.create_dataset_item_request import CreateDatasetItemRequest
from langfuse.api.resources.commons.types.dataset_run import DatasetRun


class InitialGeneration(CreateGenerationRequest):
    __fields__ = {
        name: field for name, field in CreateGenerationRequest.__fields__.items() if name not in ["traceIdType"]
    }


class InitialScore(CreateScoreRequest):
    __fields__ = {name: field for name, field in CreateScoreRequest.__fields__.items() if name not in ["traceIdType"]}


class InitialSpan(CreateSpanRequest):
    __fields__ = {name: field for name, field in CreateSpanRequest.__fields__.items() if name not in ["traceIdType"]}


class CreateScore(CreateScoreRequest):
    __fields__ = {
        name: field
        for name, field in CreateScoreRequest.__fields__.items()
        if name not in ["trace_id", "observation_id", "traceIdType"]
    }


class CreateTrace(CreateTraceRequest):
    __fields__ = {
        name: field for name, field in CreateTraceRequest.__fields__.items() if name not in ["release", "external_id"]
    }


class CreateGeneration(CreateGenerationRequest):
    __fields__ = {
        name: field
        for name, field in CreateGenerationRequest.__fields__.items()
        if name not in ["parent_observation_id", "traceIdType"]
    }


class CreateSpan(CreateSpanRequest):
    __fields__ = {
        name: field
        for name, field in CreateSpanRequest.__fields__.items()
        if name not in ["parent_observation_id", "traceIdType"]
    }


class CreateEvent(CreateEventRequest):
    __fields__ = {
        name: field
        for name, field in CreateEventRequest.__fields__.items()
        if name not in ["parent_observation_id", "traceIdType"]
    }


class UpdateGeneration(UpdateGenerationRequest):
    __fields__ = {
        name: field for name, field in UpdateGenerationRequest.__fields__.items() if name not in ["generation_id"]
    }


class UpdateSpan(UpdateSpanRequest):
    __fields__ = {
        name: field for name, field in UpdateSpanRequest.__fields__.items() if name not in ["spanId", "span_id"]
    }


class Usage(LlmUsage):
    __fields__ = {name: field for name, field in LlmUsage.__fields__.items()}
