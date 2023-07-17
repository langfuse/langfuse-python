import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Type, TypeVar, Union

import attr
from dateutil.parser import isoparse

from ..models.observation_level_generation import ObservationLevelGeneration
from ..models.trace_id_type_generations import TraceIdTypeGenerations
from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.create_log_model_parameters import CreateLogModelParameters
    from ..models.llm_usage import LLMUsage


T = TypeVar("T", bound="CreateLog")


@attr.s(auto_attribs=True)
class CreateLog:
    """
    Attributes:
        trace_id (Union[Unset, None, str]):
        trace_id_type (Union[Unset, TraceIdTypeGenerations]):
        name (Union[Unset, None, str]):
        start_time (Union[Unset, None, datetime.datetime]):
        end_time (Union[Unset, None, datetime.datetime]):
        completion_start_time (Union[Unset, None, datetime.datetime]):
        model (Union[Unset, None, str]):
        model_parameters (Union[Unset, None, CreateLogModelParameters]):
        prompt (Union[Unset, Any]):
        metadata (Union[Unset, Any]):
        completion (Union[Unset, None, str]):
        usage (Union[Unset, LLMUsage]):
        level (Union[Unset, ObservationLevelGeneration]):
        status_message (Union[Unset, None, str]):
        parent_observation_id (Union[Unset, None, str]):
    """

    trace_id: Union[Unset, None, str] = UNSET
    trace_id_type: Union[Unset, TraceIdTypeGenerations] = UNSET
    name: Union[Unset, None, str] = UNSET
    start_time: Union[Unset, None, datetime.datetime] = UNSET
    end_time: Union[Unset, None, datetime.datetime] = UNSET
    completion_start_time: Union[Unset, None, datetime.datetime] = UNSET
    model: Union[Unset, None, str] = UNSET
    model_parameters: Union[Unset, None, "CreateLogModelParameters"] = UNSET
    prompt: Union[Unset, Any] = UNSET
    metadata: Union[Unset, Any] = UNSET
    completion: Union[Unset, None, str] = UNSET
    usage: Union[Unset, "LLMUsage"] = UNSET
    level: Union[Unset, ObservationLevelGeneration] = UNSET
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
        if parent_observation_id is not UNSET:
            field_dict["parentObservationId"] = parent_observation_id

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        from ..models.create_log_model_parameters import CreateLogModelParameters
        from ..models.llm_usage import LLMUsage

        d = src_dict.copy()
        trace_id = d.pop("traceId", UNSET)

        _trace_id_type = d.pop("traceIdType", UNSET)
        trace_id_type: Union[Unset, TraceIdTypeGenerations]
        if isinstance(_trace_id_type, Unset):
            trace_id_type = UNSET
        else:
            trace_id_type = TraceIdTypeGenerations(_trace_id_type)

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
        model_parameters: Union[Unset, None, CreateLogModelParameters]
        if _model_parameters is None:
            model_parameters = None
        elif isinstance(_model_parameters, Unset):
            model_parameters = UNSET
        else:
            model_parameters = CreateLogModelParameters.from_dict(_model_parameters)

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

        parent_observation_id = d.pop("parentObservationId", UNSET)

        create_log = cls(
            trace_id=trace_id,
            trace_id_type=trace_id_type,
            name=name,
            start_time=start_time,
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
            parent_observation_id=parent_observation_id,
        )

        create_log.additional_properties = d
        return create_log

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
