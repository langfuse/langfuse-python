import logging
from queue import Empty
from typing import Any, Optional

import requests

from langfuse.api import GetMediaUploadUrlRequest, PatchMediaBody
from langfuse.api.client import FernLangfuse
from langfuse.media import LangfuseMedia
from langfuse.utils import _get_timestamp

from .media_upload_queue import MediaUploadQueue, UploadMediaJob


class MediaManager:
    _log = logging.getLogger(__name__)

    def __init__(
        self, *, api_client: FernLangfuse, media_upload_queue: MediaUploadQueue
    ):
        self._api_client = api_client
        self._queue = media_upload_queue

    def process_next_media_upload(self):
        try:
            upload_job = self._queue.get(block=True, timeout=1)
            self._process_upload_media_job(data=upload_job)

            self._queue.task_done()
        except Empty:
            pass
        except Exception as e:
            self._log.error(f"Error uploading media: {e}")
            self._queue.task_done()

    def process_media_in_event(self, event: dict):
        try:
            if "body" not in event:
                return

            body = event["body"]
            trace_id = body.get("traceId", None) or (
                body.get("id", None)
                if "type" in event and "trace" in event["type"]
                else None
            )

            if trace_id is None:
                raise ValueError("trace_id is required for media upload")

            observation_id = (
                body.get("id", None)
                if "type" in event
                and ("generation" in event["type"] or "span" in event["type"])
                else None
            )

            multimodal_fields = ["input", "output", "metadata"]

            for field in multimodal_fields:
                if field in body:
                    processed_data = self._find_and_process_media(
                        data=body[field],
                        trace_id=trace_id,
                        observation_id=observation_id,
                        field=field,
                    )

                    body[field] = processed_data

        except Exception as e:
            self._log.error(f"Error processing multimodal event: {e}")

    def _find_and_process_media(
        self,
        *,
        data: Any,
        trace_id: str,
        observation_id: Optional[str],
        field: str,
    ):
        seen = set()

        def _process_data_recursively(data: Any):
            if id(data) in seen:
                return data

            seen.add(id(data))

            if isinstance(data, LangfuseMedia):
                self._process_media(
                    media=data,
                    trace_id=trace_id,
                    observation_id=observation_id,
                    field=field,
                )

                return data

            if isinstance(data, str) and data.startswith("data:"):
                media = LangfuseMedia(
                    obj=data,
                    base64_data_uri=data,
                )

                self._process_media(
                    media=media,
                    trace_id=trace_id,
                    observation_id=observation_id,
                    field=field,
                )

                return media

            if isinstance(data, list):
                return [_process_data_recursively(item) for item in data]

            if isinstance(data, dict):
                return {
                    key: _process_data_recursively(value) for key, value in data.items()
                }

            return data

        return _process_data_recursively(data)

    def _process_media(
        self,
        *,
        media: LangfuseMedia,
        trace_id: str,
        observation_id: Optional[str],
        field: str,
    ):
        if (
            media._content_length is None
            or media._content_type is None
            or media._content_sha256_hash is None
            or media._content_bytes is None
        ):
            return

        upload_url_response = self._api_client.media.get_upload_url(
            request=GetMediaUploadUrlRequest(
                contentLength=media._content_length,
                contentType=media._content_type,
                sha256Hash=media._content_sha256_hash,
                field=field,
                traceId=trace_id,
                observationId=observation_id,
            )
        )

        upload_url = upload_url_response.upload_url
        media._media_id = upload_url_response.media_id  # Important as this is will be used in the media reference string in serializer

        if upload_url is not None:
            self._queue.put(
                item={
                    "upload_url": upload_url,
                    "media_id": media._media_id,
                    "content_bytes": media._content_bytes,
                    "content_type": media._content_type,
                    "content_sha256_hash": media._content_sha256_hash,
                },
                block=True,
                timeout=1,
            )

    def _process_upload_media_job(
        self,
        *,
        data: UploadMediaJob,
    ):
        upload_response = requests.put(
            data["upload_url"],
            headers={
                "Content-Type": data["content_type"],
                "x-amz-checksum-sha256": data["content_sha256_hash"],
            },
            data=data["content_bytes"],
        )

        self._api_client.media.patch(
            media_id=data["media_id"],
            request=PatchMediaBody(
                uploadedAt=_get_timestamp(),
                uploadHttpStatus=upload_response.status_code,
                uploadHttpError=upload_response.text,
            ),
        )
