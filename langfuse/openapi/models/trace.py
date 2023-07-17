import datetime
from typing import Any, Dict, List, Type, TypeVar, Union

import attr
from dateutil.parser import isoparse

from ..types import UNSET, Unset

T = TypeVar("T", bound="Trace")


@attr.s(auto_attribs=True)
class Trace:
    """
    Attributes:
        id (str):
        timestamp (datetime.datetime):
        external_id (Union[Unset, None, str]):
        name (Union[Unset, None, str]):
        user_id (Union[Unset, None, str]):
        metadata (Union[Unset, Any]):
    """

    id: str
    timestamp: datetime.datetime
    external_id: Union[Unset, None, str] = UNSET
    name: Union[Unset, None, str] = UNSET
    user_id: Union[Unset, None, str] = UNSET
    metadata: Union[Unset, Any] = UNSET
    additional_properties: Dict[str, Any] = attr.ib(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        id = self.id
        timestamp = self.timestamp.isoformat()

        external_id = self.external_id
        name = self.name
        user_id = self.user_id
        metadata = self.metadata

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "id": id,
                "timestamp": timestamp,
            }
        )
        if external_id is not UNSET:
            field_dict["externalId"] = external_id
        if name is not UNSET:
            field_dict["name"] = name
        if user_id is not UNSET:
            field_dict["userId"] = user_id
        if metadata is not UNSET:
            field_dict["metadata"] = metadata

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        d = src_dict.copy()
        id = d.pop("id")

        timestamp = isoparse(d.pop("timestamp"))

        external_id = d.pop("externalId", UNSET)

        name = d.pop("name", UNSET)

        user_id = d.pop("userId", UNSET)

        metadata = d.pop("metadata", UNSET)

        trace = cls(
            id=id,
            timestamp=timestamp,
            external_id=external_id,
            name=name,
            user_id=user_id,
            metadata=metadata,
        )

        trace.additional_properties = d
        return trace

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
