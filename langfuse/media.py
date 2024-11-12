import base64
import hashlib
import logging
from typing import Optional


class LangfuseMedia:
    """A class for wrapping media objects for upload to Langfuse.

    This class handles the preparation and formatting of media content for Langfuse,
    supporting both base64 data URIs and raw content bytes.

    Args:
        obj: The source object to be wrapped.
        base64_data_uri (Optional[str]): A base64-encoded data URI containing the media content
            and content type (e.g., "data:image/jpeg;base64,/9j/4AAQ...").
        content_type (Optional[str]): The MIME type of the media content when providing raw bytes.
        content_bytes (Optional[bytes]): Raw bytes of the media content.

    Raises:
        ValueError: If neither base64_data_uri or the combination of content_bytes
            and content_type is provided.
    """

    obj: object
    _log = logging.getLogger(__name__)
    _content_bytes: bytes | None
    _content_type: str | None
    _source: str | None
    _media_id: str | None

    def __init__(
        self,
        obj: object,
        *,
        base64_data_uri: Optional[str] = None,
        content_type: Optional[str] = None,
        content_bytes: Optional[bytes] = None,
    ):
        self.obj = obj
        self._media_id = None

        if base64_data_uri is not None:
            parsed_data = self._parse_base64_data_uri(base64_data_uri)
            self._content_bytes, self._content_type = parsed_data
            self._source = "base64_data_uri"

        elif content_bytes is not None and content_type is not None:
            self._content_type = content_type
            self._content_bytes = content_bytes
            self._source = "bytes"
        else:
            raise ValueError(
                "base64_data_uri or content_bytes and content_type must be provided"
            )

    @property
    def _content_length(self) -> int | None:
        return len(self._content_bytes) if self._content_bytes else None

    @property
    def _content_sha256_hash(self) -> str | None:
        if self._content_bytes is None:
            return None

        sha256_hash_bytes = hashlib.sha256(self._content_bytes).digest()

        return base64.b64encode(sha256_hash_bytes).decode("utf-8")

    @property
    def _reference_string(self) -> str | None:
        if self._content_type is None or self._source is None or self._media_id is None:
            return None

        return f"@@@langfuseMedia:type={self._content_type}|id={self._media_id}|source={self._source}@@@"

    def _parse_base64_data_uri(self, data: str):
        try:
            if not data or not isinstance(data, str):
                raise ValueError("Data URI is not a string")

            if not data.startswith("data:"):
                raise ValueError("Data URI does not start with 'data:'")

            header, _, actual_data = data[5:].partition(",")
            if not header or not actual_data:
                raise ValueError("Invalid URI")

            is_base64 = header.endswith(";base64")
            if not is_base64:
                raise ValueError("Data is not base64 encoded")

            content_type = header[:-7]
            if not content_type:
                raise ValueError("Content type is empty")

            return base64.b64decode(actual_data), content_type

        except Exception as e:
            self._log.error("Error parsing base64 data URI", exc_info=e)

            return None, None
