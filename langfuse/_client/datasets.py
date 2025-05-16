import datetime as dt
import logging
from typing import TYPE_CHECKING, Any, List, Optional

from opentelemetry.util._decorator import _agnosticcontextmanager

from langfuse.model import (
    CreateDatasetRunItemRequest,
    Dataset,
    DatasetItem,
    DatasetStatus,
)

if TYPE_CHECKING:
    from langfuse._client.client import Langfuse


class DatasetItemClient:
    """Class for managing dataset items in Langfuse.

    Args:
        id (str): Unique identifier of the dataset item.
        status (DatasetStatus): The status of the dataset item. Can be either 'ACTIVE' or 'ARCHIVED'.
        input (Any): Input data of the dataset item.
        expected_output (Optional[Any]): Expected output of the dataset item.
        metadata (Optional[Any]): Additional metadata of the dataset item.
        source_trace_id (Optional[str]): Identifier of the source trace.
        source_observation_id (Optional[str]): Identifier of the source observation.
        dataset_id (str): Identifier of the dataset to which this item belongs.
        dataset_name (str): Name of the dataset to which this item belongs.
        created_at (datetime): Timestamp of dataset item creation.
        updated_at (datetime): Timestamp of the last update to the dataset item.
        langfuse (Langfuse): Instance of Langfuse client for API interactions.

    Example:
        ```python
        from langfuse import Langfuse

        langfuse = Langfuse()

        dataset = langfuse.get_dataset("<dataset_name>")

        for item in dataset.items:
            # Generate a completion using the input of every item
            completion, generation = llm_app.run(item.input)

            # Evaluate the completion
            generation.score(
                name="example-score",
                value=1
            )
        ```
    """

    log = logging.getLogger("langfuse")

    id: str
    status: DatasetStatus
    input: Any
    expected_output: Optional[Any]
    metadata: Optional[Any]
    source_trace_id: Optional[str]
    source_observation_id: Optional[str]
    dataset_id: str
    dataset_name: str
    created_at: dt.datetime
    updated_at: dt.datetime

    langfuse: "Langfuse"

    def __init__(self, dataset_item: DatasetItem, langfuse: "Langfuse"):
        """Initialize the DatasetItemClient."""
        self.id = dataset_item.id
        self.status = dataset_item.status
        self.input = dataset_item.input
        self.expected_output = dataset_item.expected_output
        self.metadata = dataset_item.metadata
        self.source_trace_id = dataset_item.source_trace_id
        self.source_observation_id = dataset_item.source_observation_id
        self.dataset_id = dataset_item.dataset_id
        self.dataset_name = dataset_item.dataset_name
        self.created_at = dataset_item.created_at
        self.updated_at = dataset_item.updated_at

        self.langfuse = langfuse

    @_agnosticcontextmanager
    def run(
        self,
        *,
        run_name: str,
        run_metadata: Optional[Any] = None,
        run_description: Optional[str] = None,
    ):
        """Create a context manager for the dataset item run that links the execution to a Langfuse trace.

        This method is a context manager that creates a trace for the dataset run and yields a span
        that can be used to track the execution of the run.

        Args:
            run_name (str): The name of the dataset run.
            run_metadata (Optional[Any]): Additional metadata to include in dataset run.
            run_description (Optional[str]): Description of the dataset run.

        Yields:
            span: A LangfuseSpan that can be used to trace the execution of the run.
        """
        trace_name = f"Dataset run: {run_name}"

        with self.langfuse.start_as_current_span(name=trace_name) as span:
            span.update_trace(
                name=trace_name,
                metadata={
                    "dataset_item_id": self.id,
                    "run_name": run_name,
                    "dataset_id": self.dataset_id,
                },
            )

            self.log.debug(
                f"Creating dataset run item: run_name={run_name} id={self.id} trace_id={span.trace_id}"
            )

            self.langfuse.api.dataset_run_items.create(
                request=CreateDatasetRunItemRequest(
                    runName=run_name,
                    datasetItemId=self.id,
                    traceId=span.trace_id,
                    metadata=run_metadata,
                    runDescription=run_description,
                )
            )

            yield span


class DatasetClient:
    """Class for managing datasets in Langfuse.

    Attributes:
        id (str): Unique identifier of the dataset.
        name (str): Name of the dataset.
        description (Optional[str]): Description of the dataset.
        metadata (Optional[typing.Any]): Additional metadata of the dataset.
        project_id (str): Identifier of the project to which the dataset belongs.
        created_at (datetime): Timestamp of dataset creation.
        updated_at (datetime): Timestamp of the last update to the dataset.
        items (List[DatasetItemClient]): List of dataset items associated with the dataset.

    Example:
        Print the input of each dataset item in a dataset.
        ```python
        from langfuse import Langfuse

        langfuse = Langfuse()

        dataset = langfuse.get_dataset("<dataset_name>")

        for item in dataset.items:
            print(item.input)
        ```
    """

    id: str
    name: str
    description: Optional[str]
    project_id: str
    metadata: Optional[Any]
    created_at: dt.datetime
    updated_at: dt.datetime
    items: List[DatasetItemClient]

    def __init__(self, dataset: Dataset, items: List[DatasetItemClient]):
        """Initialize the DatasetClient."""
        self.id = dataset.id
        self.name = dataset.name
        self.description = dataset.description
        self.project_id = dataset.project_id
        self.metadata = dataset.metadata
        self.created_at = dataset.created_at
        self.updated_at = dataset.updated_at
        self.items = items
