from typing import TypedDict, Optional


# these imports need to stay here, otherwise imports from our clients wont work
from langfuse.api.resources.commons.types.dataset_item import DatasetItem  # noqa: F401
from langfuse.api.resources.commons.types.dataset import Dataset  # noqa: F401
from langfuse.api.resources.dataset_run_items.types.create_dataset_run_item_request import CreateDatasetRunItemRequest  # noqa: F401
from langfuse.api.resources.datasets.types.create_dataset_request import CreateDatasetRequest  # noqa: F401
from langfuse.api.resources.dataset_items.types.create_dataset_item_request import CreateDatasetItemRequest  # noqa: F401
from langfuse.api.resources.commons.types.dataset_run import DatasetRun  # noqa: F401
from langfuse.api.resources.commons.types.observation import Observation  # noqa: F401
from langfuse.api.resources.commons.types.map_value import MapValue  # noqa: F401
from langfuse.api.resources.commons.types.trace_with_full_details import TraceWithFullDetails  # noqa: F401
from langfuse.api.resources.commons.types.dataset_status import DatasetStatus  # noqa: F401


class ModelUsage(TypedDict):
    unit: Optional[str]
    input: Optional[int]
    output: Optional[int]
    total: Optional[int]
