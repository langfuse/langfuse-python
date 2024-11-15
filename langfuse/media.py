"""This module contains the LangfuseMedia class, which is used to wrap media objects for upload to Langfuse."""

import base64
import hashlib
import logging
import os
from typing import Optional, cast, Tuple

from langfuse.api import MediaContentType
from langfuse.types import ParsedMediaReference


class LangfuseMedia:
    """A class for wrapping media objects for upload to Langfuse.

    This class handles the preparation and formatting of media content for Langfuse,
    supporting both base64 data URIs and raw content bytes.

    Args:
        obj (Optional[object]): The source object to be wrapped. Can be accessed via the `obj` attribute.
        base64_data_uri (Optional[str]): A base64-encoded data URI containing the media content
            and content type (e.g., "data:image/jpeg;base64,/9j/4AAQ...").
        content_type (Optional[str]): The MIME type of the media content when providing raw bytes.
        content_bytes (Optional[bytes]): Raw bytes of the media content.
        file_path (Optional[str]): The path to the file containing the media content. For relative paths,
            the current working directory is used.

    Raises:
        ValueError: If neither base64_data_uri or the combination of content_bytes
            and content_type is provided.
    """

    obj: object

    _log = logging.getLogger(__name__)
    _content_bytes: Optional[bytes]
    _content_type: Optional[MediaContentType]
    _source: Optional[str]
    _media_id: Optional[str]

    def __init__(
        self,
        *,
        obj: Optional[object] = None,
        base64_data_uri: Optional[str] = None,
        content_type: Optional[MediaContentType] = None,
        content_bytes: Optional[bytes] = None,
        file_path: Optional[str] = None,
    ):
        """Initialize a LangfuseMedia object.

        Args:
            obj: The object to wrap.

            base64_data_uri: A base64-encoded data URI containing the media content
                and content type (e.g., "data:image/jpeg;base64,/9j/4AAQ...").
            content_type: The MIME type of the media content when providing raw bytes or reading from a file.
            content_bytes: Raw bytes of the media content.
            file_path: The path to the file containing the media content. For relative paths,
                the current working directory is used.
        """
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
        elif (
            file_path is not None
            and content_type is not None
            and os.path.exists(file_path)
        ):
            self._content_bytes = self._read_file(file_path)
            self._content_type = content_type if self._content_bytes else None
            self._source = "file" if self._content_bytes else None
        else:
            self._log.error(
                "base64_data_uri, or content_bytes and content_type, or file_path must be provided to LangfuseMedia"
            )

            self._content_bytes = None
            self._content_type = None
            self._source = None

    def _read_file(self, file_path: str) -> Optional[bytes]:
        try:
            with open(file_path, "rb") as file:
                return file.read()
        except Exception as e:
            self._log.error(f"Error reading file at path {file_path}", exc_info=e)

            return None

    @property
    def _content_length(self) -> Optional[int]:
        return len(self._content_bytes) if self._content_bytes else None

    @property
    def _content_sha256_hash(self) -> Optional[str]:
        if self._content_bytes is None:
            return None

        sha256_hash_bytes = hashlib.sha256(self._content_bytes).digest()

        return base64.b64encode(sha256_hash_bytes).decode("utf-8")

    @property
    def _reference_string(self) -> Optional[str]:
        if self._content_type is None or self._source is None or self._media_id is None:
            return None

        return f"@@@langfuseMedia:type={self._content_type}|id={self._media_id}|source={self._source}@@@"

    @staticmethod
    def parse_reference_string(reference_string: str) -> ParsedMediaReference:
        """Parse a media reference string into a ParsedMediaReference.

        Example reference string:
            "@@@langfuseMedia:type=image/jpeg|id=some-uuid|source=base64_data_uri@@@"

        Args:
            reference_string: The reference string to parse.

        Returns:
            A TypedDict with the media_id, source, and content_type.

        Raises:
            ValueError: If the reference string is empty or not a string.
            ValueError: If the reference string does not start with "@@@langfuseMedia:type=".
            ValueError: If the reference string does not end with "@@@".
            ValueError: If the reference string is missing required fields.
        """
        if not reference_string:
            raise ValueError("Reference string is empty")

        if not isinstance(reference_string, str):
            raise ValueError("Reference string is not a string")

        if not reference_string.startswith("@@@langfuseMedia:type="):
            raise ValueError(
                "Reference string does not start with '@@@langfuseMedia:type='"
            )

        if not reference_string.endswith("@@@"):
            raise ValueError("Reference string does not end with '@@@'")

        content = reference_string[len("@@@langfuseMedia:") :].rstrip("@@@")

        # Split into key-value pairs
        pairs = content.split("|")
        parsed_data = {}

        for pair in pairs:
            key, value = pair.split("=", 1)
            parsed_data[key] = value

        # Verify all required fields are present
        if not all(key in parsed_data for key in ["type", "id", "source"]):
            raise ValueError("Missing required fields in reference string")

        return ParsedMediaReference(
            media_id=parsed_data["id"],
            source=parsed_data["source"],
            content_type=parsed_data["type"],
        )

    def _parse_base64_data_uri(
        self, data: str
    ) -> Tuple[Optional[bytes], Optional[MediaContentType]]:
        # Example data URI: data:image/jpeg;base64,/9j/4AAQ...
        try:
            if not data or not isinstance(data, str):
                raise ValueError("Data URI is not a string")

            if not data.startswith("data:"):
                raise ValueError("Data URI does not start with 'data:'")

            header, actual_data = data[5:].split(",", 1)
            if not header or not actual_data:
                raise ValueError("Invalid URI")

            # Split header into parts and check for base64
            header_parts = header.split(";")
            if "base64" not in header_parts:
                raise ValueError("Data is not base64 encoded")

            # Content type is the first part
            content_type = header_parts[0]
            if not content_type:
                raise ValueError("Content type is empty")

            return base64.b64decode(actual_data), cast(MediaContentType, content_type)

        except Exception as e:
            self._log.error("Error parsing base64 data URI", exc_info=e)

            return None, None
