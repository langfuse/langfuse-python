# This file was auto-generated by Fern from our API Definition.

from __future__ import annotations

import typing

import typing_extensions

from .create_event_event import CreateEventEvent
from .create_generation_event import CreateGenerationEvent
from .create_observation_event import CreateObservationEvent
from .create_span_event import CreateSpanEvent
from .score_event import ScoreEvent
from .sdk_log_event import SdkLogEvent
from .trace_event import TraceEvent
from .update_generation_event import UpdateGenerationEvent
from .update_observation_event import UpdateObservationEvent
from .update_span_event import UpdateSpanEvent


class IngestionEvent_TraceCreate(TraceEvent):
    type: typing_extensions.Literal["trace-create"]

    class Config:
        frozen = True
        smart_union = True
        allow_population_by_field_name = True


class IngestionEvent_ScoreCreate(ScoreEvent):
    type: typing_extensions.Literal["score-create"]

    class Config:
        frozen = True
        smart_union = True
        allow_population_by_field_name = True


class IngestionEvent_EventCreate(CreateEventEvent):
    type: typing_extensions.Literal["event-create"]

    class Config:
        frozen = True
        smart_union = True
        allow_population_by_field_name = True


class IngestionEvent_GenerationCreate(CreateGenerationEvent):
    type: typing_extensions.Literal["generation-create"]

    class Config:
        frozen = True
        smart_union = True
        allow_population_by_field_name = True


class IngestionEvent_GenerationUpdate(UpdateGenerationEvent):
    type: typing_extensions.Literal["generation-update"]

    class Config:
        frozen = True
        smart_union = True
        allow_population_by_field_name = True


class IngestionEvent_SpanCreate(CreateSpanEvent):
    type: typing_extensions.Literal["span-create"]

    class Config:
        frozen = True
        smart_union = True
        allow_population_by_field_name = True


class IngestionEvent_SpanUpdate(UpdateSpanEvent):
    type: typing_extensions.Literal["span-update"]

    class Config:
        frozen = True
        smart_union = True
        allow_population_by_field_name = True


class IngestionEvent_SdkLog(SdkLogEvent):
    type: typing_extensions.Literal["sdk-log"]

    class Config:
        frozen = True
        smart_union = True
        allow_population_by_field_name = True


class IngestionEvent_ObservationCreate(CreateObservationEvent):
    type: typing_extensions.Literal["observation-create"]

    class Config:
        frozen = True
        smart_union = True
        allow_population_by_field_name = True


class IngestionEvent_ObservationUpdate(UpdateObservationEvent):
    type: typing_extensions.Literal["observation-update"]

    class Config:
        frozen = True
        smart_union = True
        allow_population_by_field_name = True


IngestionEvent = typing.Union[
    IngestionEvent_TraceCreate,
    IngestionEvent_ScoreCreate,
    IngestionEvent_EventCreate,
    IngestionEvent_GenerationCreate,
    IngestionEvent_GenerationUpdate,
    IngestionEvent_SpanCreate,
    IngestionEvent_SpanUpdate,
    IngestionEvent_SdkLog,
    IngestionEvent_ObservationCreate,
    IngestionEvent_ObservationUpdate,
]
