import logging
import time
from queue import Empty, Queue
from typing import Any, Callable, Optional, TypeVar

import backoff
import requests
from typing_extensions import ParamSpec

from langfuse.api import GetMediaUploadUrlRequest, PatchMediaBody
from langfuse.api.client import FernLangfuse
from langfuse.api.core import ApiError
from langfuse.media import LangfuseMedia
from langfuse.utils import _get_timestamp

from .media_upload_queue import UploadMediaJob

T = TypeVar("T")
P = ParamSpec("P")


class MediaManager:
    _log = logging.getLogger(__name__)

    def __init__(
        self,
        *,
        api_client: FernLangfuse,
        media_upload_queue: Queue,
        max_retries: Optional[int] = 3,
    ):
        self._api_client = api_client
        self._queue = media_upload_queue
        self._max_retries = max_retries

    def process_next_media_upload(self):
        try:
            upload_job = self._queue.get(block=True, timeout=1)
            self._log.debug(f"Processing upload for {upload_job['media_id']}")
            self._process_upload_media_job(data=upload_job)

            self._queue.task_done()
        except Empty:
            self._log.debug("Media upload queue is empty")
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
        max_levels = 10

        def _process_data_recursively(data: Any, level: int):
            if id(data) in seen or level > max_levels:
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

            # Anthropic
            if (
                isinstance(data, dict)
                and "type" in data
                and data["type"] == "base64"
                and "media_type" in data
                and "data" in data
            ):
                media = LangfuseMedia(
                    base64_data_uri=f"data:{data['media_type']};base64," + data["data"],
                )

                self._process_media(
                    media=media,
                    trace_id=trace_id,
                    observation_id=observation_id,
                    field=field,
                )

                data["data"] = media

                return data

            # Vertex
            if (
                isinstance(data, dict)
                and "type" in data
                and data["type"] == "media"
                and "mime_type" in data
                and "data" in data
            ):
                media = LangfuseMedia(
                    base64_data_uri=f"data:{data['mime_type']};base64," + data["data"],
                )

                self._process_media(
                    media=media,
                    trace_id=trace_id,
                    observation_id=observation_id,
                    field=field,
                )

                data["data"] = media

                return data

            if isinstance(data, list):
                return [_process_data_recursively(item, level + 1) for item in data]

            if isinstance(data, dict):
                return {
                    key: _process_data_recursively(value, level + 1)
                    for key, value in data.items()
                }

            return data

        return _process_data_recursively(data, 1)

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

        upload_url_response = self._request_with_backoff(
            self._api_client.media.get_upload_url,
            request=GetMediaUploadUrlRequest(
                contentLength=media._content_length,
                contentType=media._content_type,
                sha256Hash=media._content_sha256_hash,
                field=field,
                traceId=trace_id,
                observationId=observation_id,
            ),
        )

        upload_url = upload_url_response.upload_url
        media._media_id = upload_url_response.media_id  # Important as this is will be used in the media reference string in serializer

        if upload_url is not None:
            self._log.debug(f"Scheduling upload for {media._media_id}")
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

        else:
            self._log.debug(f"Media {media._media_id} already uploaded")

    def _process_upload_media_job(
        self,
        *,
        data: UploadMediaJob,
    ):
        upload_start_time = time.time()
        upload_response = self._request_with_backoff(
            requests.put,
            data["upload_url"],
            headers={
                "Content-Type": data["content_type"],
                "x-amz-checksum-sha256": data["content_sha256_hash"],
                "x-ms-blob-type": "BlockBlob",
            },
            data=data["content_bytes"],
        )
        upload_time_ms = int((time.time() - upload_start_time) * 1000)

        self._request_with_backoff(
            self._api_client.media.patch,
            media_id=data["media_id"],
            request=PatchMediaBody(
                uploadedAt=_get_timestamp(),
                uploadHttpStatus=upload_response.status_code,
                uploadHttpError=upload_response.text,
                uploadTimeMs=upload_time_ms,
            ),
        )

        self._log.debug(
            f"Media upload completed for {data['media_id']} in {upload_time_ms}ms"
        )

    def _request_with_backoff(
        self, func: Callable[P, T], *args: P.args, **kwargs: P.kwargs
    ) -> T:
        @backoff.on_exception(
            backoff.expo, Exception, max_tries=self._max_retries, logger=None
        )
        def execute_task_with_backoff() -> T:
            try:
                return func(*args, **kwargs)
            except ApiError as e:
                if (
                    e.status_code is not None
                    and 400 <= e.status_code < 500
                    and (e.status_code) != 429
                ):
                    raise e
            except requests.exceptions.RequestException as e:
                if (
                    e.response is not None
                    and hasattr(e.response, "status_code")
                    and (e.response.status_code >= 500 or e.response.status_code == 429)
                ):
                    raise

                raise e  # break retries for all other status codes

            raise Exception("Failed to execute task")

        return execute_task_with_backoff()
