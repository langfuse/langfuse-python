"""@private"""

from abc import ABC, abstractmethod
from typing import Optional, TypedDict, Any, Dict, Union, List
import re

from langfuse.api.resources.commons.types.dataset import (
    Dataset,  # noqa: F401
)

# these imports need to stay here, otherwise imports from our clients wont work
from langfuse.api.resources.commons.types.dataset_item import DatasetItem  # noqa: F401

# noqa: F401
from langfuse.api.resources.commons.types.dataset_run import DatasetRun  # noqa: F401

# noqa: F401
from langfuse.api.resources.commons.types.dataset_status import (  # noqa: F401
    DatasetStatus,
)
from langfuse.api.resources.commons.types.map_value import MapValue  # noqa: F401
from langfuse.api.resources.commons.types.observation import Observation  # noqa: F401
from langfuse.api.resources.commons.types.trace_with_full_details import (  # noqa: F401
    TraceWithFullDetails,
)

# noqa: F401
from langfuse.api.resources.dataset_items.types.create_dataset_item_request import (  # noqa: F401
    CreateDatasetItemRequest,
)
from langfuse.api.resources.dataset_run_items.types.create_dataset_run_item_request import (  # noqa: F401
    CreateDatasetRunItemRequest,
)

# noqa: F401
from langfuse.api.resources.datasets.types.create_dataset_request import (  # noqa: F401
    CreateDatasetRequest,
)
from langfuse.api.resources.prompts import Prompt, ChatMessage, Prompt_Chat, Prompt_Text


class ModelUsage(TypedDict):
    unit: Optional[str]
    input: Optional[int]
    output: Optional[int]
    total: Optional[int]
    input_cost: Optional[float]
    output_cost: Optional[float]
    total_cost: Optional[float]


class ChatMessageDict(TypedDict):
    role: str
    content: str


class BasePromptClient(ABC):
    name: str
    version: int
    config: Dict[str, Any]
    labels: List[str]

    def __init__(self, prompt: Prompt):
        self.name = prompt.name
        self.version = prompt.version
        self.config = prompt.config
        self.labels = prompt.labels

    @abstractmethod
    def compile(self, **kwargs) -> Union[str, List[ChatMessage]]:
        pass

    @abstractmethod
    def __eq__(self, other):
        pass

    @abstractmethod
    def get_langchain_prompt(self):
        pass

    @staticmethod
    def _get_langchain_prompt_string(content: str):
        return re.sub(r"\{\{(.*?)\}\}", r"{\g<1>}", content)

    @staticmethod
    def _compile_template_string(content: str, data: Dict[str, Any] = {}) -> str:
        opening = "{{"
        closing = "}}"

        result_list = []
        curr_idx = 0

        while curr_idx < len(content):
            # Find the next opening tag
            var_start = content.find(opening, curr_idx)

            if var_start == -1:
                result_list.append(content[curr_idx:])
                break

            # Find the next closing tag
            var_end = content.find(closing, var_start)

            if var_end == -1:
                result_list.append(content[curr_idx:])
                break

            # Append the content before the variable
            result_list.append(content[curr_idx:var_start])

            # Extract the variable name
            variable_name = content[var_start + len(opening) : var_end].strip()

            # Append the variable value
            if variable_name in data:
                result_list.append(
                    str(data[variable_name]) if data[variable_name] is not None else ""
                )
            else:
                result_list.append(content[var_start : var_end + len(closing)])

            curr_idx = var_end + len(closing)

        return "".join(result_list)


class TextPromptClient(BasePromptClient):
    def __init__(self, prompt: Prompt_Text):
        super().__init__(prompt)
        self.prompt = prompt.prompt

    def compile(self, **kwargs) -> str:
        return self._compile_template_string(self.prompt, kwargs)

    def __eq__(self, other):
        if isinstance(self, other.__class__):
            return (
                self.name == other.name
                and self.version == other.version
                and self.prompt == other.prompt
                and self.config == other.config
            )

        return False

    def get_langchain_prompt(self):
        """Convert Langfuse prompt into string compatible with Langchain PromptTemplate.

        It specifically adapts the mustache-style double curly braces {{variable}} used in Langfuse
        to the single curly brace {variable} format expected by Langchain.

        Returns:
            str: The string that can be plugged into Langchain's PromptTemplate.
        """
        return self._get_langchain_prompt_string(self.prompt)


class ChatPromptClient(BasePromptClient):
    def __init__(self, prompt: Prompt_Chat):
        super().__init__(prompt)
        self.prompt = [
            ChatMessageDict(role=p.role, content=p.content) for p in prompt.prompt
        ]

    def compile(self, **kwargs) -> List[ChatMessageDict]:
        return [
            ChatMessageDict(
                content=self._compile_template_string(chat_message["content"], kwargs),
                role=chat_message["role"],
            )
            for chat_message in self.prompt
        ]

    def __eq__(self, other):
        if isinstance(self, other.__class__):
            return (
                self.name == other.name
                and self.version == other.version
                and all(
                    m1["role"] == m2["role"] and m1["content"] == m2["content"]
                    for m1, m2 in zip(self.prompt, other.prompt)
                )
                and self.config == other.config
            )

        return False

    def get_langchain_prompt(self):
        """Convert Langfuse prompt into string compatible with Langchain ChatPromptTemplate.

        It specifically adapts the mustache-style double curly braces {{variable}} used in Langfuse
        to the single curly brace {variable} format expected by Langchain.

        Returns:
            List of messages in the format expected by Langchain's ChatPromptTemplate: (role, content) tuple.
        """
        return [
            (msg["role"], self._get_langchain_prompt_string(msg["content"]))
            for msg in self.prompt
        ]


PromptClient = Union[TextPromptClient, ChatPromptClient]
