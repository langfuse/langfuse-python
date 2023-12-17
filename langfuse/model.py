from typing import Optional, TypedDict

from langfuse.api.resources.commons.types.dataset import Dataset  # noqa: F401

# these imports need to stay here, otherwise imports from our clients wont work
from langfuse.api.resources.commons.types.dataset_item import DatasetItem  # noqa: F401
from langfuse.api.resources.commons.types.dataset_run import DatasetRun  # noqa: F401
from langfuse.api.resources.commons.types.map_value import MapValue  # noqa: F401
from langfuse.api.resources.commons.types.observation import Observation  # noqa: F401
from langfuse.api.resources.dataset_run_items.types.create_dataset_run_item_request import (  # noqa: F401
    CreateDatasetRunItemRequest,
)


class ModelUsage(TypedDict):
    unit: Optional[str]
    input: Optional[int]
    output: Optional[int]
    total: Optional[int]
