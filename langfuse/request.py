from base64 import b64encode
from gzip import GzipFile
from io import BytesIO
import json
import logging
from typing import Any, Union
import requests

from langfuse.serializer import DatetimeSerializer


_session = requests.sessions.Session()


class LangfuseClient:
    _public_key: str
    _secret_key: str
    _base_url: str
    _version: str

    def __init__(self, public_key: str, secret_key: str, base_url: str, version: str):
        self._public_key = public_key
        self._secret_key = secret_key
        self._base_url = base_url
        self._version = version

    def generate_headers(self):
        return {
            "Authorization": b64encode(f"{self._public_key}:{self._secret_key}".encode("utf-8")).decode("ascii"),
            "Content-Type": "application/json",
            "x_langfuse_sdk_name": "python",
            "x_langfuse_sdk_version": self._version,
            "x_langfuse_public_key": self._public_key,
        }

    def batch_post(self, gzip: bool = False, timeout: int = 15, **kwargs) -> requests.Response:
        """Post the `kwargs` to the batch API endpoint for events"""
        res = self.post(gzip, timeout, **kwargs)
        return self._process_response(res, success_message="data uploaded successfully", return_json=False)

    def post(self, gzip: bool = False, timeout: int = 15, **kwargs) -> requests.Response:
        """Post the `kwargs` to the API"""
        log = logging.getLogger("langfuse")
        body = kwargs

        url = self.remove_trailing_slash(self._base_url) + "/api/public/ingestion"
        data = json.dumps(body, cls=DatetimeSerializer)
        log.debug("making request: %s", data)
        headers = self.generate_headers()
        if gzip:
            headers["Content-Encoding"] = "gzip"
            buf = BytesIO()
            with GzipFile(fileobj=buf, mode="w") as gz:
                # 'data' was produced by json.dumps(),
                # whose default encoding is utf-8.
                gz.write(data.encode("utf-8"))
            data = buf.getvalue()

        res = _session.post(url, data=data, headers=headers, timeout=timeout)

        if res.status_code == 200:
            log.debug("data uploaded successfully")

        return res

    def remove_trailing_slash(self, url: str) -> str:
        """Removes the trailing slash from a URL"""
        if url.endswith("/"):
            return url[:-1]
        return url

    def _process_response(self, res: requests.Response, success_message: str, *, return_json: bool = True) -> Union[requests.Response, Any]:
        log = logging.getLogger("posthog")
        if res.status_code == 200:
            log.debug(success_message)
            return res.json() if return_json else res
        try:
            payload = res.json()
            log.debug("received response: %s", payload)
            raise APIError(res.status_code, payload["detail"])
        except (KeyError, ValueError):
            raise APIError(res.status_code, res.text)


class APIError(Exception):
    def __init__(self, status: Union[int, str], message: str):
        self.message = message
        self.status = status

    def __str__(self):
        msg = "[Langfuse] {0} ({1})"
        return msg.format(self.message, self.status)
