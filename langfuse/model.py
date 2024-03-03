from typing import Optional, TypedDict, Any, Dict
import chevron
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
from langfuse.api.resources.prompts.types.prompt import Prompt


class ModelUsage(TypedDict):
    unit: Optional[str]
    input: Optional[int]
    output: Optional[int]
    total: Optional[int]
    input_cost: Optional[float]
    output_cost: Optional[float]
    total_cost: Optional[float]


class PromptClient:
    name: str
    version: int
    prompt: str
    # same to input/output of the observations, this is typed to Any.
    config: Dict[str, Any]

    def __init__(self, prompt: Prompt):
        self.name = prompt.name
        self.version = prompt.version
        self.prompt = prompt.prompt
        self.config = prompt.config

    def compile(self, **kwargs) -> str:
        return chevron.render(self.prompt, kwargs)

    def __eq__(self, other):
        if isinstance(self, other.__class__):
            return (
                self.name == other.name
                and self.version == other.version
                and self.prompt == other.prompt
                and self.config == other.config
            )

        return False

    def get_langchain_prompt(self) -> str:
        """Converts Langfuse prompt into string compatible with Langchain PromptTemplate.

        It specifically adapts the mustache-style double curly braces {{variable}} used in Langfuse
        to the single curly brace {variable} format expected by Langchain.

        Returns:
            str: The string that can be plugged into Langchain's PromptTemplate.
        """
        return re.sub(r"\{\{(.*?)\}\}", r"{\g<1>}", self.prompt)
