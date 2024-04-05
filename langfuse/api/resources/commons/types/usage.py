# This file was auto-generated by Fern from our API Definition.

import datetime as dt
import typing

from ....core.datetime_utils import serialize_datetime
from ....core.pydantic_utilities import pydantic_v1
from .model_usage_unit import ModelUsageUnit


class Usage(pydantic_v1.BaseModel):
    """Standard interface for usage and cost
    """

    input: typing.Optional[int] = pydantic_v1.Field(default=None)
    """
    Number of input units (e.g. tokens)
    """

    output: typing.Optional[int] = pydantic_v1.Field(default=None)
    """
    Number of output units (e.g. tokens)
    """

    total: typing.Optional[int] = pydantic_v1.Field(default=None)
    """
    Defaults to input+output if not set
    """

    unit: typing.Optional[ModelUsageUnit] = None
    input_cost: typing.Optional[float] = pydantic_v1.Field(alias="inputCost", default=None)
    """
    USD input cost
    """

    output_cost: typing.Optional[float] = pydantic_v1.Field(alias="outputCost", default=None)
    """
    USD output cost
    """

    total_cost: typing.Optional[float] = pydantic_v1.Field(alias="totalCost", default=None)
    """
    USD total cost, defaults to input+output
    """

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
