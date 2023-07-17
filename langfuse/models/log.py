import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Type, TypeVar, Union

import attr
from dateutil.parser import isoparse

from ..models.observation_level_generation import ObservationLevelGeneration
from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.llm_usage import LLMUsage
    from ..models.log_model_parameters import LogModelParameters


T = TypeVar("T", bound="Log")


@attr.s(auto_attribs=True)
class Log:
    """
    Attributes:
        id (str):
        trace_id (str):
        type (str):
        start_time (datetime.datetime):
        level (ObservationLevelGeneration):
        name (Union[Unset, None, str]):
        end_time (Union[Unset, None, datetime.datetime]):
        completion_start_time (Union[Unset, None, datetime.datetime]):
        model (Union[Unset, None, str]):
        model_parameters (Union[Unset, None, LogModelParameters]):
        prompt (Union[Unset, Any]):
        metadata (Union[Unset, Any]):
        completion (Union[Unset, None, str]):
        usage (Union[Unset, LLMUsage]):
        status_message (Union[Unset, None, str]):
        parent_observation_id (Union[Unset, None, str]):
    """

    id: str
    trace_id: str
    type: str
    start_time: datetime.datetime
    level: ObservationLevelGeneration
    name: Union[Unset, None, str] = UNSET
    end_time: Union[Unset, None, datetime.datetime] = UNSET
    completion_start_time: Union[Unset, None, datetime.datetime] = UNSET
    model: Union[Unset, None, str] = UNSET
    model_parameters: Union[Unset, None, "LogModelParameters"] = UNSET
    prompt: Union[Unset, Any] = UNSET
    metadata: Union[Unset, Any] = UNSET
    completion: Union[Unset, None, str] = UNSET
    usage: Union[Unset, "LLMUsage"] = UNSET
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
        if status_message is not UNSET:
            field_dict["statusMessage"] = status_message
        if parent_observation_id is not UNSET:
            field_dict["parentObservationId"] = parent_observation_id

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        from ..models.llm_usage import LLMUsage
        from ..models.log_model_parameters import LogModelParameters

        d = src_dict.copy()
        id = d.pop("id")

        trace_id = d.pop("traceId")

        type = d.pop("type")

        start_time = isoparse(d.pop("startTime"))

        level = ObservationLevelGeneration(d.pop("level"))

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
        model_parameters: Union[Unset, None, LogModelParameters]
        if _model_parameters is None:
            model_parameters = None
        elif isinstance(_model_parameters, Unset):
            model_parameters = UNSET
        else:
            model_parameters = LogModelParameters.from_dict(_model_parameters)

        prompt = d.pop("prompt", UNSET)

        metadata = d.pop("metadata", UNSET)

        completion = d.pop("completion", UNSET)

        _usage = d.pop("usage", UNSET)
        usage: Union[Unset, LLMUsage]
        if isinstance(_usage, Unset):
            usage = UNSET
        else:
            usage = LLMUsage.from_dict(_usage)

        status_message = d.pop("statusMessage", UNSET)

        parent_observation_id = d.pop("parentObservationId", UNSET)

        log = cls(
            id=id,
            trace_id=trace_id,
            type=type,
            start_time=start_time,
            level=level,
            name=name,
            end_time=end_time,
            completion_start_time=completion_start_time,
            model=model,
            model_parameters=model_parameters,
            prompt=prompt,
            metadata=metadata,
            completion=completion,
            usage=usage,
            status_message=status_message,
            parent_observation_id=parent_observation_id,
        )

        log.additional_properties = d
        return log

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
