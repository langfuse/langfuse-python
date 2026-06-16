from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from langfuse._client.client import Langfuse
from langfuse.api import (
    DatasetItem,
    DatasetItemMediaReference,
    DatasetItemMediaReferenceField,
    DatasetItemMediaReferenceMedia,
    DatasetStatus,
)
from langfuse.media import LangfuseMedia, LangfuseMediaReference


def test_hydrate_dataset_item_media_references_replaces_matching_fields():
    reference_string = "@@@langfuseMedia:type=image/png|id=media-id|source=bytes@@@"
    item = DatasetItem(
        id="item-id",
        status=DatasetStatus.ACTIVE,
        input={
            "image": reference_string,
            "duplicate": reference_string,
            "text": "keep",
        },
        expected_output=[reference_string],
        metadata={"nested": {"image": reference_string}},
        dataset_id="dataset-id",
        dataset_name="dataset-name",
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        media_references=[
            DatasetItemMediaReference(
                field=DatasetItemMediaReferenceField.INPUT,
                reference_string=reference_string,
                json_path="$['image']",
                media=DatasetItemMediaReferenceMedia(
                    media_id="media-id",
                    content_type="image/png",
                    content_length=7,
                    url="https://example.com/image.png",
                    url_expiry="2026-06-15T12:00:00.000Z",
                ),
            ),
            DatasetItemMediaReference(
                field=DatasetItemMediaReferenceField.EXPECTED_OUTPUT,
                reference_string=reference_string,
                json_path="$[0]",
                media=DatasetItemMediaReferenceMedia(
                    media_id="media-id",
                    content_type="image/png",
                    content_length=7,
                    url="https://example.com/image.png",
                    url_expiry="2026-06-15T12:00:00.000Z",
                ),
            ),
            DatasetItemMediaReference(
                field=DatasetItemMediaReferenceField.METADATA,
                reference_string=reference_string,
                json_path="$['nested']['image']",
                media=DatasetItemMediaReferenceMedia(
                    media_id="media-id",
                    content_type="image/png",
                    content_length=7,
                    url="https://example.com/image.png",
                    url_expiry="2026-06-15T12:00:00.000Z",
                ),
            ),
        ],
    )

    client = object.__new__(Langfuse)

    hydrated = client._hydrate_dataset_item_media_references(item)

    assert hydrated.input["text"] == "keep"
    assert isinstance(hydrated.input["image"], LangfuseMediaReference)
    assert hydrated.input["duplicate"] == reference_string
    assert isinstance(hydrated.expected_output[0], LangfuseMediaReference)
    assert isinstance(hydrated.metadata["nested"]["image"], LangfuseMediaReference)
    assert hydrated.input["image"].media_id == "media-id"


@pytest.mark.parametrize(
    ("field", "field_value", "json_path", "assert_resolved"),
    [
        (
            DatasetItemMediaReferenceField.INPUT,
            "@@@langfuseMedia:type=image/png|id=media-id|source=bytes@@@",
            "$",
            lambda item: isinstance(item.input, LangfuseMediaReference),
        ),
        (
            DatasetItemMediaReferenceField.INPUT,
            {
                "image": "@@@langfuseMedia:type=image/png|id=media-id|source=bytes@@@"
            },
            "$['image']",
            lambda item: isinstance(item.input["image"], LangfuseMediaReference),
        ),
        (
            DatasetItemMediaReferenceField.EXPECTED_OUTPUT,
            ["@@@langfuseMedia:type=image/png|id=media-id|source=bytes@@@"],
            "$[0]",
            lambda item: isinstance(
                item.expected_output[0], LangfuseMediaReference
            ),
        ),
        (
            DatasetItemMediaReferenceField.METADATA,
            {
                "image'key": "@@@langfuseMedia:type=image/png|id=media-id|source=bytes@@@"
            },
            r"$['image\'key']",
            lambda item: isinstance(
                item.metadata["image'key"], LangfuseMediaReference
            ),
        ),
    ],
)
def test_hydrate_dataset_item_media_references_supports_json_path_cases(
    field,
    field_value,
    json_path,
    assert_resolved,
):
    reference_string = "@@@langfuseMedia:type=image/png|id=media-id|source=bytes@@@"
    item = DatasetItem(
        id="item-id",
        status=DatasetStatus.ACTIVE,
        input=field_value if field == DatasetItemMediaReferenceField.INPUT else None,
        expected_output=field_value
        if field == DatasetItemMediaReferenceField.EXPECTED_OUTPUT
        else None,
        metadata=field_value if field == DatasetItemMediaReferenceField.METADATA else None,
        dataset_id="dataset-id",
        dataset_name="dataset-name",
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        media_references=[
            DatasetItemMediaReference(
                field=field,
                reference_string=reference_string,
                json_path=json_path,
                media=DatasetItemMediaReferenceMedia(
                    media_id="media-id",
                    content_type="image/png",
                    content_length=7,
                    url="https://example.com/image.png",
                    url_expiry="2026-06-15T12:00:00.000Z",
                ),
            )
        ],
    )

    client = object.__new__(Langfuse)

    hydrated = client._hydrate_dataset_item_media_references(item)

    assert assert_resolved(hydrated)


def test_create_dataset_item_processes_media_before_api_call():
    media = LangfuseMedia(content_bytes=b"payload", content_type="image/png")
    root_media = LangfuseMedia(content_bytes=b"root", content_type="image/png")

    media_manager = Mock()
    dataset_items_api = Mock()
    dataset_items_api.create.return_value = "created-item"

    client = object.__new__(Langfuse)
    client._resources = SimpleNamespace(_media_manager=media_manager)
    client.api = SimpleNamespace(dataset_items=dataset_items_api)
    input_data = {"image": media}
    metadata = {"items": [media], "keep": "value"}

    result = client.create_dataset_item(
        dataset_name="dataset",
        input=input_data,
        expected_output=root_media,
        metadata=metadata,
    )

    assert result == "created-item"
    assert input_data == {"image": media}
    assert metadata == {"items": [media], "keep": "value"}
    media_manager._upload_media_sync.assert_any_call(media=media)
    media_manager._upload_media_sync.assert_any_call(media=root_media)
    assert media_manager._upload_media_sync.call_count == 2
    dataset_items_api.create.assert_called_once_with(
        dataset_name="dataset",
        input={"image": media._reference_string},
        expected_output=root_media._reference_string,
        metadata={"items": [media._reference_string], "keep": "value"},
        source_trace_id=None,
        source_observation_id=None,
        status=None,
        id=None,
    )


def test_create_dataset_item_processes_shared_media_subtrees():
    media = LangfuseMedia(content_bytes=b"payload", content_type="image/png")
    shared = {"image": media}

    media_manager = Mock()
    dataset_items_api = Mock()
    dataset_items_api.create.return_value = "created-item"

    client = object.__new__(Langfuse)
    client._resources = SimpleNamespace(_media_manager=media_manager)
    client.api = SimpleNamespace(dataset_items=dataset_items_api)

    client.create_dataset_item(
        dataset_name="dataset",
        input={"a": shared, "b": shared},
    )

    assert shared == {"image": media}
    media_manager._upload_media_sync.assert_called_once_with(media=media)
    assert dataset_items_api.create.call_args.kwargs["input"] == {
        "a": {"image": media._reference_string},
        "b": {"image": media._reference_string},
    }
