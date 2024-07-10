# This file was auto-generated by Fern from our API Definition.

import datetime as dt
import typing

from ....core.datetime_utils import serialize_datetime
from ....core.pydantic_utilities import pydantic_v1
from .observation import Observation


class ObservationsView(Observation):
    model_id: typing.Optional[str] = pydantic_v1.Field(alias="modelId", default=None)
    input_price: typing.Optional[float] = pydantic_v1.Field(
        alias="inputPrice", default=None
    )
    output_price: typing.Optional[float] = pydantic_v1.Field(
        alias="outputPrice", default=None
    )
    total_price: typing.Optional[float] = pydantic_v1.Field(
        alias="totalPrice", default=None
    )
    calculated_input_cost: typing.Optional[float] = pydantic_v1.Field(
        alias="calculatedInputCost", default=None
    )
    calculated_output_cost: typing.Optional[float] = pydantic_v1.Field(
        alias="calculatedOutputCost", default=None
    )
    calculated_total_cost: typing.Optional[float] = pydantic_v1.Field(
        alias="calculatedTotalCost", default=None
    )
    latency: typing.Optional[float] = None
    time_to_first_token: typing.Optional[float] = pydantic_v1.Field(
        alias="timeToFirstToken", default=None
    )

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
        populate_by_name = True
        extra = pydantic_v1.Extra.allow
        json_encoders = {dt.datetime: serialize_datetime}
