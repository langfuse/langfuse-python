from datetime import datetime, timezone

from langfuse._client.datasets import DatasetClient
from langfuse.api import Dataset, DatasetItem, DatasetRunItem, DatasetStatus


def test_dataset_experiment_links_run_item_to_root_observation(
    langfuse_memory_client, get_span, monkeypatch
):
    created_run_item = {}

    def create_dataset_run_item(**kwargs):
        created_run_item.update(kwargs)
        return DatasetRunItem(
            id="run-item-id",
            dataset_run_id="dataset-run-id",
            dataset_run_name=kwargs["run_name"],
            dataset_item_id=kwargs["dataset_item_id"],
            trace_id=kwargs["trace_id"],
            observation_id=kwargs["observation_id"],
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )

    monkeypatch.setattr(
        langfuse_memory_client.api.dataset_run_items,
        "create",
        create_dataset_run_item,
    )

    now = datetime.now(timezone.utc)
    dataset = DatasetClient(
        dataset=Dataset(
            id="dataset-id",
            name="dataset",
            description=None,
            metadata=None,
            project_id="project-id",
            created_at=now,
            updated_at=now,
        ),
        items=[
            DatasetItem(
                id="dataset-item-id",
                status=DatasetStatus.ACTIVE,
                input="input",
                expected_output="expected",
                metadata=None,
                source_trace_id=None,
                source_observation_id=None,
                dataset_id="dataset-id",
                dataset_name="dataset",
                created_at=now,
                updated_at=now,
            )
        ],
        langfuse_client=langfuse_memory_client,
    )

    dataset.run_experiment(name="experiment", task=lambda *, item, **kwargs: "output")
    langfuse_memory_client.flush()

    root_span = get_span("experiment-item-run")
    assert created_run_item["trace_id"] == format(root_span.context.trace_id, "032x")
    assert created_run_item["observation_id"] == format(
        root_span.context.span_id,
        "016x",
    )
