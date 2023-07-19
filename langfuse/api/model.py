from langfuse.api.resources.event.types.create_event_request import CreateEventRequest
from langfuse.api.resources.generations.types.create_log import CreateLog
from langfuse.api.resources.score.types.create_score_request import CreateScoreRequest
from langfuse.api.resources.span.types.create_span_request import CreateSpanRequest
from langfuse.api.resources.trace.types.create_trace_request import CreateTraceRequest


class CreateScore(CreateScoreRequest):
    class Config:
        pass

    __fields__ = {
        name: field for name, field in CreateScoreRequest.__fields__.items() if name != 'trace_id'
    }


class CreateTrace(CreateTraceRequest):
    class Config:
        pass

    __fields__ = {
        name: field for name, field in CreateTraceRequest.__fields__.items()
    }

class CreateGeneration(CreateLog):
    class Config:
        pass

    __fields__ = {
        name: field for name, field in CreateLog.__fields__.items()
    }

class CreateSpan(CreateSpanRequest):
    class Config:
        pass

    __fields__ = {
        name: field for name, field in CreateLog.__fields__.items()
    }

class CreateEvent(CreateEventRequest):
    class Config:
        pass

    __fields__ = {
        name: field for name, field in CreateLog.__fields__.items()
    }