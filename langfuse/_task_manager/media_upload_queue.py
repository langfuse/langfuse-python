from queue import Queue
from typing import TypedDict


class UploadMediaJob(TypedDict):
    upload_url: str
    media_id: str
    content_type: str
    content_bytes: bytes
    content_sha256_hash: str


class MediaUploadQueue(Queue[UploadMediaJob]):
    pass
