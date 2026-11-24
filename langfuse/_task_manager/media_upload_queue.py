from queue import Queue
from typing import Optional, TypedDict, cast


class UploadMediaJob(TypedDict):
    media_id: str
    content_type: str
    content_length: int
    content_bytes: bytes
    content_sha256_hash: str
    trace_id: Optional[str]
    observation_id: Optional[str]
    dataset_id: Optional[str]
    dataset_item_id: Optional[str]
    field: Optional[str]


class MediaUploadQueue(Queue[object]):
    """Share attachment bytes across pending jobs while preserving each job."""

    def __init__(self, maxsize: int) -> None:
        super().__init__(maxsize)
        self._pending: dict[str, tuple[bytes, int]] = {}

    # Queue calls both hooks with its mutex held.
    def _put(self, item: object) -> None:
        # The shutdown sentinel and non-bytes payloads pass through unchanged.
        if not isinstance(item, dict) or type(item["content_bytes"]) is not bytes:
            super()._put(item)
            return

        job = cast(UploadMediaJob, item)
        key = job["content_sha256_hash"]
        previous = self._pending.get(key)
        payload, count = previous if previous is not None else (job["content_bytes"], 0)
        if payload is not job["content_bytes"]:
            job = job.copy()
            job["content_bytes"] = payload

        self._pending[key] = (payload, count + 1)
        try:
            super()._put(job)
        except Exception:
            if previous is None:
                del self._pending[key]
            else:
                self._pending[key] = previous
            raise

    def _get(self) -> object:
        item = super()._get()
        if isinstance(item, dict) and type(item["content_bytes"]) is bytes:
            key = item["content_sha256_hash"]
            payload, count = self._pending[key]
            if count == 1:
                del self._pending[key]
            else:
                self._pending[key] = (payload, count - 1)
        return item
