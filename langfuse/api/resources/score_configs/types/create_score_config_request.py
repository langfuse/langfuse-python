# This file was auto-generated by Fern from our API Definition.

import datetime as dt
import typing

from ....core.datetime_utils import serialize_datetime
from ....core.pydantic_utilities import pydantic_v1
from ...commons.types.config_category import ConfigCategory
from ...commons.types.score_data_type import ScoreDataType


class CreateScoreConfigRequest(pydantic_v1.BaseModel):
    name: str
    data_type: ScoreDataType = pydantic_v1.Field(alias="dataType")
    categories: typing.Optional[typing.List[ConfigCategory]] = pydantic_v1.Field(
        default=None
    )
    """
    Configure custom categories for categorical scores. Pass a list of objects with `label` and `value` properties. Categories are autogenerated for boolean configs and cannot be passed
    """

    min_value: typing.Optional[float] = pydantic_v1.Field(
        alias="minValue", default=None
    )
    """
    Configure a minimum value for numerical scores. If not set, the minimum value defaults to -∞
    """

    max_value: typing.Optional[float] = pydantic_v1.Field(
        alias="maxValue", default=None
    )
    """
    Configure a maximum value for numerical scores. If not set, the maximum value defaults to +∞
    """

    description: typing.Optional[str] = pydantic_v1.Field(default=None)
    """
    Description is shown across the Langfuse UI and can be used to e.g. explain the config categories in detail, why a numeric range was set, or provide additional context on config name or usage
    """

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
