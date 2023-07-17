from typing import Any, Dict, List, Type, TypeVar, Union

import attr

from ..types import UNSET, Unset

T = TypeVar("T", bound="CreateTraceRequest")


@attr.s(auto_attribs=True)
class CreateTraceRequest:
    """
    Attributes:
        name (Union[Unset, None, str]):
        user_id (Union[Unset, None, str]):
        external_id (Union[Unset, None, str]):
        metadata (Union[Unset, Any]):
    """

    name: Union[Unset, None, str] = UNSET
    user_id: Union[Unset, None, str] = UNSET
    external_id: Union[Unset, None, str] = UNSET
    metadata: Union[Unset, Any] = UNSET
    additional_properties: Dict[str, Any] = attr.ib(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        name = self.name
        user_id = self.user_id
        external_id = self.external_id
        metadata = self.metadata

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if name is not UNSET:
            field_dict["name"] = name
        if user_id is not UNSET:
            field_dict["userId"] = user_id
        if external_id is not UNSET:
            field_dict["externalId"] = external_id
        if metadata is not UNSET:
            field_dict["metadata"] = metadata

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        d = src_dict.copy()
        name = d.pop("name", UNSET)

        user_id = d.pop("userId", UNSET)

        external_id = d.pop("externalId", UNSET)

        metadata = d.pop("metadata", UNSET)

        create_trace_request = cls(
            name=name,
            user_id=user_id,
            external_id=external_id,
            metadata=metadata,
        )

        create_trace_request.additional_properties = d
        return create_trace_request

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
