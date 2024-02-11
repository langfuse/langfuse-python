# This file was auto-generated by Fern from our API Definition.

import datetime as dt
import typing

from ....core.datetime_utils import serialize_datetime
from .model_usage_unit import ModelUsageUnit

try:
    import pydantic.v1 as pydantic  # type: ignore
except ImportError:
    import pydantic  # type: ignore


class Usage(pydantic.BaseModel):
    input: typing.Optional[int] = None
    output: typing.Optional[int] = None
    total: typing.Optional[int] = None
    unit: typing.Optional[ModelUsageUnit] = None
    input_cost: typing.Optional[float] = pydantic.Field(alias="inputCost", default=None)
    output_cost: typing.Optional[float] = pydantic.Field(
        alias="outputCost", default=None
    )
    total_cost: typing.Optional[float] = pydantic.Field(alias="totalCost", default=None)

    def json(self, **kwargs: typing.Any) -> str:
        kwargs_with_defaults: typing.Any = {
            "by_alias": True,
            "exclude_unset": True,
            **kwargs,
        }
        return super().json(**kwargs_with_defaults)

    def dict(self, **kwargs: typing.Any) -> typing.Dict[str, typing.Any]:
        kwargs_with_defaults: typing.Any = {
            "by_alias": True,
            "exclude_unset": True,
            **kwargs,
        }
        return super().dict(**kwargs_with_defaults)

    class Config:
        frozen = True
        smart_union = True
        allow_population_by_field_name = True
        json_encoders = {dt.datetime: serialize_datetime}
