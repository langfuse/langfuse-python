# This file was auto-generated by Fern from our API Definition.

import datetime as dt
import typing

from ....core.datetime_utils import serialize_datetime
from ....core.pydantic_utilities import deep_union_pydantic_dicts, pydantic_v1
from ...commons.types.create_score_value import CreateScoreValue
from ...commons.types.score_data_type import ScoreDataType


class CreateScoreRequest(pydantic_v1.BaseModel):
    """
    Examples
    --------
    from langfuse import CreateScoreRequest

    CreateScoreRequest(
        name="novelty",
        value=0.9,
        trace_id="cdef-1234-5678-90ab",
    )
    """

    id: typing.Optional[str] = None
    trace_id: str = pydantic_v1.Field(alias="traceId")
    name: str
    value: CreateScoreValue = pydantic_v1.Field()
    """
    The value of the score. Must be passed as string for categorical scores, and numeric for boolean and numeric scores. Boolean score values must equal either 1 or 0 (true or false)
    """

    observation_id: typing.Optional[str] = pydantic_v1.Field(
        alias="observationId", default=None
    )
    comment: typing.Optional[str] = None
    data_type: typing.Optional[ScoreDataType] = pydantic_v1.Field(
        alias="dataType", default=None
    )
    """
    The data type of the score. When passing a configId this field is inferred. Otherwise, this field must be passed or will default to numeric.
    """

    config_id: typing.Optional[str] = pydantic_v1.Field(alias="configId", default=None)
    """
    Reference a score config on a score. The unique langfuse identifier of a score config. When passing this field, the dataType and stringValue fields are automatically populated.
    """

    def json(self, **kwargs: typing.Any) -> str:
        kwargs_with_defaults: typing.Any = {
            "by_alias": True,
            "exclude_unset": True,
            **kwargs,
        }
        return super().json(**kwargs_with_defaults)

    def dict(self, **kwargs: typing.Any) -> typing.Dict[str, typing.Any]:
        kwargs_with_defaults_exclude_unset: typing.Any = {
            "by_alias": True,
            "exclude_unset": True,
            **kwargs,
        }
        kwargs_with_defaults_exclude_none: typing.Any = {
            "by_alias": True,
            "exclude_none": True,
            **kwargs,
        }

        return deep_union_pydantic_dicts(
            super().dict(**kwargs_with_defaults_exclude_unset),
            super().dict(**kwargs_with_defaults_exclude_none),
        )

    class Config:
        frozen = True
        smart_union = True
        allow_population_by_field_name = True
        populate_by_name = True
        extra = pydantic_v1.Extra.allow
        json_encoders = {dt.datetime: serialize_datetime}
