import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Type, TypeVar, Union

import attr
from dateutil.parser import isoparse

from ..models.observation_level_generation import ObservationLevelGeneration
from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.llm_usage import LLMUsage
    from ..models.update_generation_request_model_parameters import UpdateGenerationRequestModelParameters


T = TypeVar("T", bound="UpdateGenerationRequest")


@attr.s(auto_attribs=True)
class UpdateGenerationRequest:
    """
    Attributes:
        generation_id (str):
        name (Union[Unset, None, str]):
        end_time (Union[Unset, None, datetime.datetime]):
        completion_start_time (Union[Unset, None, datetime.datetime]):
        model (Union[Unset, None, str]):
        model_parameters (Union[Unset, None, UpdateGenerationRequestModelParameters]):
        prompt (Union[Unset, Any]):
        metadata (Union[Unset, Any]):
        completion (Union[Unset, None, str]):
        usage (Union[Unset, LLMUsage]):
        level (Union[Unset, ObservationLevelGeneration]):
        status_message (Union[Unset, None, str]):
    """

    generation_id: str
    name: Union[Unset, None, str] = UNSET
    end_time: Union[Unset, None, datetime.datetime] = UNSET
    completion_start_time: Union[Unset, None, datetime.datetime] = UNSET
    model: Union[Unset, None, str] = UNSET
    model_parameters: Union[Unset, None, "UpdateGenerationRequestModelParameters"] = UNSET
    prompt: Union[Unset, Any] = UNSET
    metadata: Union[Unset, Any] = UNSET
    completion: Union[Unset, None, str] = UNSET
    usage: Union[Unset, "LLMUsage"] = UNSET
    level: Union[Unset, ObservationLevelGeneration] = UNSET
    status_message: Union[Unset, None, str] = UNSET
    additional_properties: Dict[str, Any] = attr.ib(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        generation_id = self.generation_id
        name = self.name
        end_time: Union[Unset, None, str] = UNSET
        if not isinstance(self.end_time, Unset):
            end_time = self.end_time.isoformat() if self.end_time else None

        completion_start_time: Union[Unset, None, str] = UNSET
        if not isinstance(self.completion_start_time, Unset):
            completion_start_time = self.completion_start_time.isoformat() if self.completion_start_time else None

        model = self.model
        model_parameters: Union[Unset, None, Dict[str, Any]] = UNSET
        if not isinstance(self.model_parameters, Unset):
            model_parameters = self.model_parameters.to_dict() if self.model_parameters else None

        prompt = self.prompt
        metadata = self.metadata
        completion = self.completion
        usage: Union[Unset, Dict[str, Any]] = UNSET
        if not isinstance(self.usage, Unset):
            usage = self.usage.to_dict()

        level: Union[Unset, str] = UNSET
        if not isinstance(self.level, Unset):
            level = self.level.value

        status_message = self.status_message

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "generationId": generation_id,
            }
        )
        if name is not UNSET:
            field_dict["name"] = name
        if end_time is not UNSET:
            field_dict["endTime"] = end_time
        if completion_start_time is not UNSET:
            field_dict["completionStartTime"] = completion_start_time
        if model is not UNSET:
            field_dict["model"] = model
        if model_parameters is not UNSET:
            field_dict["modelParameters"] = model_parameters
        if prompt is not UNSET:
            field_dict["prompt"] = prompt
        if metadata is not UNSET:
            field_dict["metadata"] = metadata
        if completion is not UNSET:
            field_dict["completion"] = completion
        if usage is not UNSET:
            field_dict["usage"] = usage
        if level is not UNSET:
            field_dict["level"] = level
        if status_message is not UNSET:
            field_dict["statusMessage"] = status_message

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        from ..models.llm_usage import LLMUsage
        from ..models.update_generation_request_model_parameters import UpdateGenerationRequestModelParameters

        d = src_dict.copy()
        generation_id = d.pop("generationId")

        name = d.pop("name", UNSET)

        _end_time = d.pop("endTime", UNSET)
        end_time: Union[Unset, None, datetime.datetime]
        if _end_time is None:
            end_time = None
        elif isinstance(_end_time, Unset):
            end_time = UNSET
        else:
            end_time = isoparse(_end_time)

        _completion_start_time = d.pop("completionStartTime", UNSET)
        completion_start_time: Union[Unset, None, datetime.datetime]
        if _completion_start_time is None:
            completion_start_time = None
        elif isinstance(_completion_start_time, Unset):
            completion_start_time = UNSET
        else:
            completion_start_time = isoparse(_completion_start_time)

        model = d.pop("model", UNSET)

        _model_parameters = d.pop("modelParameters", UNSET)
        model_parameters: Union[Unset, None, UpdateGenerationRequestModelParameters]
        if _model_parameters is None:
            model_parameters = None
        elif isinstance(_model_parameters, Unset):
            model_parameters = UNSET
        else:
            model_parameters = UpdateGenerationRequestModelParameters.from_dict(_model_parameters)

        prompt = d.pop("prompt", UNSET)

        metadata = d.pop("metadata", UNSET)

        completion = d.pop("completion", UNSET)

        _usage = d.pop("usage", UNSET)
        usage: Union[Unset, LLMUsage]
        if isinstance(_usage, Unset):
            usage = UNSET
        else:
            usage = LLMUsage.from_dict(_usage)

        _level = d.pop("level", UNSET)
        level: Union[Unset, ObservationLevelGeneration]
        if isinstance(_level, Unset):
            level = UNSET
        else:
            level = ObservationLevelGeneration(_level)

        status_message = d.pop("statusMessage", UNSET)

        update_generation_request = cls(
            generation_id=generation_id,
            name=name,
            end_time=end_time,
            completion_start_time=completion_start_time,
            model=model,
            model_parameters=model_parameters,
            prompt=prompt,
            metadata=metadata,
            completion=completion,
            usage=usage,
            level=level,
            status_message=status_message,
        )

        update_generation_request.additional_properties = d
        return update_generation_request

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
