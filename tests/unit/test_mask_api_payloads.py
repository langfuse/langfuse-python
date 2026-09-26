from unittest.mock import Mock

from langfuse import Langfuse


def _client():
    client = Langfuse(
        public_key="pk",
        secret_key="sk",
        host="https://mock-host.com",
        tracing_enabled=False,
        mask=lambda data: "masked",
    )
    client.api = Mock()
    client._resources.add_score_task = Mock()
    return client


def test_create_dataset_item_masks_payload():
    client = _client()
    client.create_dataset_item(
        dataset_name="ds", input="secret", expected_output="secret", metadata="secret"
    )
    kwargs = client.api.dataset_items.create.call_args.kwargs
    assert (kwargs["input"], kwargs["expected_output"], kwargs["metadata"]) == (
        "masked",
        "masked",
        "masked",
    )


def test_create_dataset_masks_metadata():
    client = _client()
    client.create_dataset(name="ds", metadata="secret")
    assert client.api.datasets.create.call_args.kwargs["metadata"] == "masked"


def test_create_score_masks_comment():
    client = _client()
    client.create_score(name="s", value=1, trace_id="t" * 32, comment="secret")
    assert (
        client._resources.add_score_task.call_args.args[0]["body"].comment == "masked"
    )
