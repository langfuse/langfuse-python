from typing import Any, Dict, List, Type, TypeVar, Union, cast

import attr

T = TypeVar("T", bound="CreateLogModelParameters")


@attr.s(auto_attribs=True)
class CreateLogModelParameters:
    """ """

    additional_properties: Dict[str, Union[bool, int, str]] = attr.ib(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        field_dict: Dict[str, Any] = {}
        for prop_name, prop in self.additional_properties.items():
            field_dict[prop_name] = prop

        field_dict.update({})

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        d = src_dict.copy()
        create_log_model_parameters = cls()

        additional_properties = {}
        for prop_name, prop_dict in d.items():

            def _parse_additional_property(data: object) -> Union[bool, int, str]:
                return cast(Union[bool, int, str], data)

            additional_property = _parse_additional_property(prop_dict)

            additional_properties[prop_name] = additional_property

        create_log_model_parameters.additional_properties = additional_properties
        return create_log_model_parameters

    @property
    def additional_keys(self) -> List[str]:
        return list(self.additional_properties.keys())

    def __getitem__(self, key: str) -> Union[bool, int, str]:
        return self.additional_properties[key]

    def __setitem__(self, key: str, value: Union[bool, int, str]) -> None:
        self.additional_properties[key] = value

    def __delitem__(self, key: str) -> None:
        del self.additional_properties[key]

    def __contains__(self, key: str) -> bool:
        return key in self.additional_properties
