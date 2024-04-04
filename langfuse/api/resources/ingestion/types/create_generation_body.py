# This file was auto-generated by Fern from our API Definition.

import datetime as dt
import typing

from ....core.datetime_utils import serialize_datetime
from ....core.pydantic_utilities import pydantic_v1
from ...commons.types.map_value import MapValue
from .create_span_body import CreateSpanBody
from .ingestion_usage import IngestionUsage


class CreateGenerationBody(CreateSpanBody):
    completion_start_time: typing.Optional[dt.datetime] = pydantic_v1.Field(alias="completionStartTime", default=None)
    model: typing.Optional[str] = None
    model_parameters: typing.Optional[typing.Dict[str, MapValue]] = pydantic_v1.Field(
        alias="modelParameters", default=None
    )
    usage: typing.Optional[IngestionUsage] = None
    prompt_name: typing.Optional[str] = pydantic_v1.Field(alias="promptName", default=None)
    prompt_version: typing.Optional[int] = pydantic_v1.Field(alias="promptVersion", default=None)

    def json(self, **kwargs: typing.Any) -> str:
        kwargs_with_defaults: typing.Any = {"by_alias": True, "exclude_unset": True, **kwargs}
        return super().json(**kwargs_with_defaults)

    def dict(self, **kwargs: typing.Any) -> typing.Dict[str, typing.Any]:
        kwargs_with_defaults: typing.Any = {"by_alias": True, "exclude_unset": True, **kwargs}
        return super().dict(**kwargs_with_defaults)

    class Config:
        frozen = True
        smart_union = True
        allow_population_by_field_name = True
        populate_by_name = True
        extra = pydantic_v1.Extra.allow
        json_encoders = {dt.datetime: serialize_datetime}
