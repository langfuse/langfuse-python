# This file was auto-generated by Fern from our API Definition.

import datetime as dt
import typing

from ....core.datetime_utils import serialize_datetime
from ....core.pydantic_utilities import pydantic_v1
from ...commons.types.dataset_status import DatasetStatus


class CreateDatasetItemRequest(pydantic_v1.BaseModel):
    dataset_name: str = pydantic_v1.Field(alias="datasetName")
    input: typing.Optional[typing.Any] = None
    expected_output: typing.Optional[typing.Any] = pydantic_v1.Field(
        alias="expectedOutput", default=None
    )
    metadata: typing.Optional[typing.Any] = None
    source_trace_id: typing.Optional[str] = pydantic_v1.Field(
        alias="sourceTraceId", default=None
    )
    source_observation_id: typing.Optional[str] = pydantic_v1.Field(
        alias="sourceObservationId", default=None
    )
    id: typing.Optional[str] = pydantic_v1.Field(default=None)
    """
    Dataset items are upserted on their id
    """

    status: typing.Optional[DatasetStatus] = pydantic_v1.Field(default=None)
    """
    Defaults to ACTIVE for newly created items
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
