import datetime
from typing import Any, Dict, List, Type, TypeVar, Union

import attr
from dateutil.parser import isoparse

from ..types import UNSET, Unset

T = TypeVar("T", bound="Score")


@attr.s(auto_attribs=True)
class Score:
    """
    Attributes:
        id (str):
        trace_id (str):
        name (str):
        value (int):
        timestamp (datetime.datetime):
        observation_id (Union[Unset, None, str]):
        comment (Union[Unset, None, str]):
    """

    id: str
    trace_id: str
    name: str
    value: int
    timestamp: datetime.datetime
    observation_id: Union[Unset, None, str] = UNSET
    comment: Union[Unset, None, str] = UNSET
    additional_properties: Dict[str, Any] = attr.ib(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        id = self.id
        trace_id = self.trace_id
        name = self.name
        value = self.value
        timestamp = self.timestamp.isoformat()

        observation_id = self.observation_id
        comment = self.comment

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "id": id,
                "traceId": trace_id,
                "name": name,
                "value": value,
                "timestamp": timestamp,
            }
        )
        if observation_id is not UNSET:
            field_dict["observationId"] = observation_id
        if comment is not UNSET:
            field_dict["comment"] = comment

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        d = src_dict.copy()
        id = d.pop("id")

        trace_id = d.pop("traceId")

        name = d.pop("name")

        value = d.pop("value")

        timestamp = isoparse(d.pop("timestamp"))

        observation_id = d.pop("observationId", UNSET)

        comment = d.pop("comment", UNSET)

        score = cls(
            id=id,
            trace_id=trace_id,
            name=name,
            value=value,
            timestamp=timestamp,
            observation_id=observation_id,
            comment=comment,
        )

        score.additional_properties = d
        return score

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
