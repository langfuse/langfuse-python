from typing import Any, Dict, List, Type, TypeVar, Union

import attr

from ..models.trace_id_type import TraceIdType
from ..types import UNSET, Unset

T = TypeVar("T", bound="CreateScoreRequest")


@attr.s(auto_attribs=True)
class CreateScoreRequest:
    """
    Attributes:
        trace_id (str):
        name (str):
        value (int):
        trace_id_type (Union[Unset, TraceIdType]):
        observation_id (Union[Unset, None, str]):
        comment (Union[Unset, None, str]):
    """

    trace_id: str
    name: str
    value: int
    trace_id_type: Union[Unset, TraceIdType] = UNSET
    observation_id: Union[Unset, None, str] = UNSET
    comment: Union[Unset, None, str] = UNSET
    additional_properties: Dict[str, Any] = attr.ib(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        trace_id = self.trace_id
        name = self.name
        value = self.value
        trace_id_type: Union[Unset, str] = UNSET
        if not isinstance(self.trace_id_type, Unset):
            trace_id_type = self.trace_id_type.value

        observation_id = self.observation_id
        comment = self.comment

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "traceId": trace_id,
                "name": name,
                "value": value,
            }
        )
        if trace_id_type is not UNSET:
            field_dict["traceIdType"] = trace_id_type
        if observation_id is not UNSET:
            field_dict["observationId"] = observation_id
        if comment is not UNSET:
            field_dict["comment"] = comment

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        d = src_dict.copy()
        trace_id = d.pop("traceId")

        name = d.pop("name")

        value = d.pop("value")

        _trace_id_type = d.pop("traceIdType", UNSET)
        trace_id_type: Union[Unset, TraceIdType]
        if isinstance(_trace_id_type, Unset):
            trace_id_type = UNSET
        else:
            trace_id_type = TraceIdType(_trace_id_type)

        observation_id = d.pop("observationId", UNSET)

        comment = d.pop("comment", UNSET)

        create_score_request = cls(
            trace_id=trace_id,
            name=name,
            value=value,
            trace_id_type=trace_id_type,
            observation_id=observation_id,
            comment=comment,
        )

        create_score_request.additional_properties = d
        return create_score_request

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
