import json
import logging
from base64 import b64encode
from gzip import GzipFile
from io import BytesIO
from typing import Any, List, Union

import requests

from langfuse.serializer import EventSerializer

_session = requests.sessions.Session()


class LangfuseClient:
    _public_key: str
    _secret_key: str
    _base_url: str
    _version: str
    _timeout: int

    def __init__(
        self,
        public_key: str,
        secret_key: str,
        base_url: str,
        version: str,
        timeout: int,
    ):
        self._public_key = public_key
        self._secret_key = secret_key
        self._base_url = base_url
        self._version = version
        self._timeout = timeout

    def generate_headers(self):
        return {
            "Authorization": "Basic "
            + b64encode(
                f"{self._public_key}:{self._secret_key}".encode("utf-8")
            ).decode("ascii"),
            "Content-Type": "application/json",
            "x_langfuse_sdk_name": "python",
            "x_langfuse_sdk_version": self._version,
            "x_langfuse_public_key": self._public_key,
        }

    def batch_post(self, gzip: bool = False, **kwargs) -> requests.Response:
        """Post the `kwargs` to the batch API endpoint for events"""

        logging.debug("uploading data: %s", kwargs)
        res = self.post(gzip, **kwargs)
        return self._process_response(
            res, success_message="data uploaded successfully", return_json=False
        )

    def post(self, gzip: bool = False, **kwargs) -> requests.Response:
        """Post the `kwargs` to the API"""
        log = logging.getLogger("langfuse")
        body = kwargs

        url = self.remove_trailing_slash(self._base_url) + "/api/public/ingestion"
        data = json.dumps(body, cls=EventSerializer)
        log.debug("making request: %s to %s", data, url)
        headers = self.generate_headers()
        if gzip:
            headers["Content-Encoding"] = "gzip"
            buf = BytesIO()
            with GzipFile(fileobj=buf, mode="w") as gz:
                # 'data' was produced by json.dumps(),
                # whose default encoding is utf-8.
                gz.write(data.encode("utf-8"))
            data = buf.getvalue()

        res = _session.post(url, data=data, headers=headers, timeout=self._timeout)

        if res.status_code == 200:
            log.debug("data uploaded successfully")

        return res

    def remove_trailing_slash(self, url: str) -> str:
        """Removes the trailing slash from a URL"""
        if url.endswith("/"):
            return url[:-1]
        return url

    def _process_response(
        self, res: requests.Response, success_message: str, *, return_json: bool = True
    ) -> Union[requests.Response, Any]:
        log = logging.getLogger("langfuse")
        log.debug("received response: %s", res.text)
        if res.status_code == 200 or res.status_code == 201:
            log.debug(success_message)
            return res.json() if return_json else res
        elif res.status_code == 207:
            payload = res.json()
            errors = payload["errors"]
            if len(errors) > 0:
                raise APIErrors(
                    [
                        APIError(error["status"], error["message"], error["error"])
                        for error in errors
                    ]
                )
            else:
                return res.json() if return_json else res
        try:
            payload = res.json()
            log.error("received error response: %s", payload)
            raise APIError(res.status_code, payload)
        except (KeyError, ValueError):
            raise APIError(res.status_code, res.text)


class APIError(Exception):
    def __init__(self, status: Union[int, str], message: str, details: Any = None):
        self.message = message
        self.status = status
        self.details = details

    def __str__(self):
        msg = "{0} ({1}): {2}"
        return msg.format(self.message, self.status, self.details)


class APIErrors(Exception):
    def __init__(self, errors: List[APIError]):
        self.errors = errors

    def __str__(self):
        errors = ", ".join(str(error) for error in self.errors)

        return f"[Langfuse] {errors}"
