import datetime
from typing import Any, Dict, List, Type, TypeVar, Union

import attr
from dateutil.parser import isoparse

from ..models.observation_level_span import ObservationLevelSpan
from ..types import UNSET, Unset

T = TypeVar("T", bound="Span")


@attr.s(auto_attribs=True)
class Span:
    """
    Attributes:
        id (str):
        trace_id (str):
        type (str):
        start_time (datetime.datetime):
        level (ObservationLevelSpan):
        name (Union[Unset, None, str]):
        end_time (Union[Unset, None, datetime.datetime]):
        metadata (Union[Unset, Any]):
        input_ (Union[Unset, Any]):
        output (Union[Unset, Any]):
        status_message (Union[Unset, None, str]):
        parent_observation_id (Union[Unset, None, str]):
    """

    id: str
    trace_id: str
    type: str
    start_time: datetime.datetime
    level: ObservationLevelSpan
    name: Union[Unset, None, str] = UNSET
    end_time: Union[Unset, None, datetime.datetime] = UNSET
    metadata: Union[Unset, Any] = UNSET
    input_: Union[Unset, Any] = UNSET
    output: Union[Unset, Any] = UNSET
    status_message: Union[Unset, None, str] = UNSET
    parent_observation_id: Union[Unset, None, str] = UNSET
    additional_properties: Dict[str, Any] = attr.ib(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        id = self.id
        trace_id = self.trace_id
        type = self.type
        start_time = self.start_time.isoformat()

        level = self.level.value

        name = self.name
        end_time: Union[Unset, None, str] = UNSET
        if not isinstance(self.end_time, Unset):
            end_time = self.end_time.isoformat() if self.end_time else None

        metadata = self.metadata
        input_ = self.input_
        output = self.output
        status_message = self.status_message
        parent_observation_id = self.parent_observation_id

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "id": id,
                "traceId": trace_id,
                "type": type,
                "startTime": start_time,
                "level": level,
            }
        )
        if name is not UNSET:
            field_dict["name"] = name
        if end_time is not UNSET:
            field_dict["endTime"] = end_time
        if metadata is not UNSET:
            field_dict["metadata"] = metadata
        if input_ is not UNSET:
            field_dict["input"] = input_
        if output is not UNSET:
            field_dict["output"] = output
        if status_message is not UNSET:
            field_dict["statusMessage"] = status_message
        if parent_observation_id is not UNSET:
            field_dict["parentObservationId"] = parent_observation_id

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        d = src_dict.copy()
        id = d.pop("id")

        trace_id = d.pop("traceId")

        type = d.pop("type")

        start_time = isoparse(d.pop("startTime"))

        level = ObservationLevelSpan(d.pop("level"))

        name = d.pop("name", UNSET)

        _end_time = d.pop("endTime", UNSET)
        end_time: Union[Unset, None, datetime.datetime]
        if _end_time is None:
            end_time = None
        elif isinstance(_end_time, Unset):
            end_time = UNSET
        else:
            end_time = isoparse(_end_time)

        metadata = d.pop("metadata", UNSET)

        input_ = d.pop("input", UNSET)

        output = d.pop("output", UNSET)

        status_message = d.pop("statusMessage", UNSET)

        parent_observation_id = d.pop("parentObservationId", UNSET)

        span = cls(
            id=id,
            trace_id=trace_id,
            type=type,
            start_time=start_time,
            level=level,
            name=name,
            end_time=end_time,
            metadata=metadata,
            input_=input_,
            output=output,
            status_message=status_message,
            parent_observation_id=parent_observation_id,
        )

        span.additional_properties = d
        return span

    @property
    def additional_keys(self) -> List[str]:
        return list(self.additional_properties.keys())

    def __getitem__(self, key: str) -> Any:
        return self.additional_properties[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self.additional_properties[key] = value

    def __delitem__(self, key: str) -> None:
        del self.additional_properties[key]

    def __contains__(self, key: str) -> bool:
        return key in self.additional_properties
