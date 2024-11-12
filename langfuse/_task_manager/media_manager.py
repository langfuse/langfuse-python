import base64
import hashlib
import logging
from queue import Empty
from typing import Literal

import requests

from langfuse.api import GetMediaUploadUrlRequest, PatchMediaBody
from langfuse.api.client import FernLangfuse
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
            data = self._queue.get(block=True, timeout=1)
            self._process_upload_media_job(data=data)

            self._queue.task_done()
        except Empty:
            pass
        except Exception as e:
            self._log.error(f"Error uploading media: {e}")
            self._queue.task_done()

    def process_multimodal_event_in_place(self, event: dict):
        try:
            if "body" not in event:
                return

            body = event["body"]
            multimodal_fields = ["input", "output"]

            for field in multimodal_fields:
                if field in body:
                    field_data = body[field]

                    if field == "output":
                        self._process_multimodal_message(
                            event=event, body=body, field=field, message=field_data
                        )

                    if isinstance(field_data, list):
                        for message in field_data:
                            self._process_multimodal_message(
                                event=event, body=body, field=field, message=message
                            )

        except Exception as e:
            self._log.error(f"Error processing multimodal event: {e}")

    def _process_multimodal_message(
        self, *, event: dict, body: dict, field: str, message: dict
    ):
        if isinstance(message, dict) and message.get("content", None) is not None:
            content = message["content"]

            for content_part in content:
                if isinstance(content_part, dict):
                    if content_part.get("image_url", None) is not None:
                        base64_data_uri = content_part["image_url"]["url"]
                        if base64_data_uri.startswith("data:"):
                            media_reference_string = self._enqueue_media_upload(
                                event=event,
                                body=body,
                                field=field,
                                base64_data_uri=base64_data_uri,
                            )

                            if media_reference_string:
                                content_part["image_url"]["url"] = (
                                    media_reference_string
                                )

                    if content_part.get("input_audio", None) is not None:
                        base64_data_uri = (
                            f"data:audio/{content_part['input_audio']['format']};base64,"
                            + content_part["input_audio"]["data"]
                        )

                        media_reference_string = self._enqueue_media_upload(
                            event=event,
                            body=body,
                            field=field,
                            base64_data_uri=base64_data_uri,
                        )

                        if media_reference_string:
                            content_part["input_audio"]["data"] = media_reference_string

                    if content_part.get("output_audio", None) is not None:
                        base64_data_uri = (
                            f"data:audio/{content_part['output_audio']['format']};base64,"
                            + content_part["output_audio"]["data"]
                        )

                        media_reference_string = self._enqueue_media_upload(
                            event=event,
                            body=body,
                            field=field,
                            base64_data_uri=base64_data_uri,
                        )

                        if media_reference_string:
                            content_part["output_audio"]["data"] = (
                                media_reference_string
                            )

    def _enqueue_media_upload(
        self, *, event: dict, body: dict, field: str, base64_data_uri: str
    ):
        parsed_content = self._parse_base64_data_uri(base64_data_uri)
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

        if parsed_content:
            content_length = parsed_content["content_length"]
            content_type = parsed_content["content_type"]
            content_sha256_hash = parsed_content["content_sha256_hash"]
            content_bytes = parsed_content["content_bytes"]

            upload_url_response = self._api_client.media.get_upload_url(
                request=GetMediaUploadUrlRequest(
                    field=field,
                    contentLength=content_length,
                    contentType=content_type,
                    sha256Hash=content_sha256_hash,
                    traceId=trace_id,
                    observationId=observation_id,
                )
            )

            upload_url = upload_url_response.upload_url
            media_id = upload_url_response.media_id

            if upload_url is not None:
                self._queue.put(
                    item={
                        "content_bytes": content_bytes,
                        "content_type": content_type,
                        "content_sha256_hash": content_sha256_hash,
                        "upload_url": upload_url,
                        "media_id": media_id,
                    },
                    block=True,
                )

            return self._format_media_reference_string(
                content_type=content_type,
                media_id=media_id,
                source="base64",
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

    def _format_media_reference_string(
        self, *, content_type: str, media_id: str, source: Literal["base64"]
    ) -> str:
        return f"@@@langfuseMedia:type={content_type}|id={media_id}|source={source}@@@"

    def _parse_base64_data_uri(self, data: str):
        if not data or not isinstance(data, str):
            return None

        if not data.startswith("data:"):
            return None

        try:
            # Split the data into metadata and actual data
            header, _, actual_data = data[5:].partition(",")
            if not header or not actual_data:
                return None

            # Determine if the data is base64 encoded
            is_base64 = header.endswith(";base64")
            if not is_base64:
                return None

            content_type = header[:-7]
            if not content_type:
                return None

            try:
                content_bytes = base64.b64decode(actual_data)
            except Exception:
                return None

            content_length = len(content_bytes)

            sha256_hash_bytes = hashlib.sha256(content_bytes).digest()
            sha256_hash_base64 = base64.b64encode(sha256_hash_bytes).decode("utf-8")

            return {
                "content_type": content_type,
                "content_bytes": content_bytes,
                "content_length": content_length,
                "content_sha256_hash": sha256_hash_base64,
            }
        except Exception:
            return None
