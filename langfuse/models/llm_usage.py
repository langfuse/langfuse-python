from typing import Any, Dict, List, Type, TypeVar, Union

import attr

from ..types import UNSET, Unset

T = TypeVar("T", bound="LLMUsage")


@attr.s(auto_attribs=True)
class LLMUsage:
    """
    Attributes:
        prompt_tokens (Union[Unset, None, int]):
        completion_tokens (Union[Unset, None, int]):
        total_tokens (Union[Unset, None, int]):
    """

    prompt_tokens: Union[Unset, None, int] = UNSET
    completion_tokens: Union[Unset, None, int] = UNSET
    total_tokens: Union[Unset, None, int] = UNSET
    additional_properties: Dict[str, Any] = attr.ib(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        prompt_tokens = self.prompt_tokens
        completion_tokens = self.completion_tokens
        total_tokens = self.total_tokens

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if prompt_tokens is not UNSET:
            field_dict["promptTokens"] = prompt_tokens
        if completion_tokens is not UNSET:
            field_dict["completionTokens"] = completion_tokens
        if total_tokens is not UNSET:
            field_dict["totalTokens"] = total_tokens

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        d = src_dict.copy()
        prompt_tokens = d.pop("promptTokens", UNSET)

        completion_tokens = d.pop("completionTokens", UNSET)

        total_tokens = d.pop("totalTokens", UNSET)

        llm_usage = cls(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )

        llm_usage.additional_properties = d
        return llm_usage

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
