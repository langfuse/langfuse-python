import datetime
from typing import Any, Dict, List, Type, TypeVar, Union

import attr
from dateutil.parser import isoparse

from ..models.observation_level_span import ObservationLevelSpan
from ..models.trace_id_type_span import TraceIdTypeSpan
from ..types import UNSET, Unset

T = TypeVar("T", bound="CreateSpanRequest")


@attr.s(auto_attribs=True)
class CreateSpanRequest:
    """
    Attributes:
        trace_id (Union[Unset, None, str]):
        trace_id_type (Union[Unset, TraceIdTypeSpan]):
        name (Union[Unset, None, str]):
        start_time (Union[Unset, None, datetime.datetime]):
        end_time (Union[Unset, None, datetime.datetime]):
        metadata (Union[Unset, Any]):
        input_ (Union[Unset, Any]):
        output (Union[Unset, Any]):
        level (Union[Unset, ObservationLevelSpan]):
        status_message (Union[Unset, None, str]):
        parent_observation_id (Union[Unset, None, str]):
    """

    trace_id: Union[Unset, None, str] = UNSET
    trace_id_type: Union[Unset, TraceIdTypeSpan] = UNSET
    name: Union[Unset, None, str] = UNSET
    start_time: Union[Unset, None, datetime.datetime] = UNSET
    end_time: Union[Unset, None, datetime.datetime] = UNSET
    metadata: Union[Unset, Any] = UNSET
    input_: Union[Unset, Any] = UNSET
    output: Union[Unset, Any] = UNSET
    level: Union[Unset, ObservationLevelSpan] = UNSET
    status_message: Union[Unset, None, str] = UNSET
    parent_observation_id: Union[Unset, None, str] = UNSET
    additional_properties: Dict[str, Any] = attr.ib(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        trace_id = self.trace_id
        trace_id_type: Union[Unset, str] = UNSET
        if not isinstance(self.trace_id_type, Unset):
            trace_id_type = self.trace_id_type.value

        name = self.name
        start_time: Union[Unset, None, str] = UNSET
        if not isinstance(self.start_time, Unset):
            start_time = self.start_time.isoformat() if self.start_time else None

        end_time: Union[Unset, None, str] = UNSET
        if not isinstance(self.end_time, Unset):
            end_time = self.end_time.isoformat() if self.end_time else None

        metadata = self.metadata
        input_ = self.input_
        output = self.output
        level: Union[Unset, str] = UNSET
        if not isinstance(self.level, Unset):
            level = self.level.value

        status_message = self.status_message
        parent_observation_id = self.parent_observation_id

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if trace_id is not UNSET:
            field_dict["traceId"] = trace_id
        if trace_id_type is not UNSET:
            field_dict["traceIdType"] = trace_id_type
        if name is not UNSET:
            field_dict["name"] = name
        if start_time is not UNSET:
            field_dict["startTime"] = start_time
        if end_time is not UNSET:
            field_dict["endTime"] = end_time
        if metadata is not UNSET:
            field_dict["metadata"] = metadata
        if input_ is not UNSET:
            field_dict["input"] = input_
        if output is not UNSET:
            field_dict["output"] = output
        if level is not UNSET:
            field_dict["level"] = level
        if status_message is not UNSET:
            field_dict["statusMessage"] = status_message
        if parent_observation_id is not UNSET:
            field_dict["parentObservationId"] = parent_observation_id

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        d = src_dict.copy()
        trace_id = d.pop("traceId", UNSET)

        _trace_id_type = d.pop("traceIdType", UNSET)
        trace_id_type: Union[Unset, TraceIdTypeSpan]
        if isinstance(_trace_id_type, Unset):
            trace_id_type = UNSET
        else:
            trace_id_type = TraceIdTypeSpan(_trace_id_type)

        name = d.pop("name", UNSET)

        _start_time = d.pop("startTime", UNSET)
        start_time: Union[Unset, None, datetime.datetime]
        if _start_time is None:
            start_time = None
        elif isinstance(_start_time, Unset):
            start_time = UNSET
        else:
            start_time = isoparse(_start_time)

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

        _level = d.pop("level", UNSET)
        level: Union[Unset, ObservationLevelSpan]
        if isinstance(_level, Unset):
            level = UNSET
        else:
            level = ObservationLevelSpan(_level)

        status_message = d.pop("statusMessage", UNSET)

        parent_observation_id = d.pop("parentObservationId", UNSET)

        create_span_request = cls(
            trace_id=trace_id,
            trace_id_type=trace_id_type,
            name=name,
            start_time=start_time,
            end_time=end_time,
            metadata=metadata,
            input_=input_,
            output=output,
            level=level,
            status_message=status_message,
            parent_observation_id=parent_observation_id,
        )

        create_span_request.additional_properties = d
        return create_span_request

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
