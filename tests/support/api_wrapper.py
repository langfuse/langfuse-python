import os

import httpx

from langfuse.api.commons.errors.not_found_error import NotFoundError
from tests.support.retry import is_not_found_payload, retry_until_ready


class LangfuseAPI:
    def __init__(self, username=None, password=None, base_url=None):
        username = username if username else os.environ["LANGFUSE_PUBLIC_KEY"]
        password = password if password else os.environ["LANGFUSE_SECRET_KEY"]
        self.auth = (username, password)
        self.BASE_URL = base_url if base_url else os.environ["LANGFUSE_BASE_URL"]

    def _get_json(self, url, params=None):
        def _request():
            response = httpx.get(url, params=params, auth=self.auth)
            payload = response.json()

            if response.status_code == 404 and is_not_found_payload(payload):
                raise NotFoundError(body=payload, headers=dict(response.headers))

            return payload

        return retry_until_ready(_request)

    def get_observation(self, observation_id):
        url = f"{self.BASE_URL}/api/public/observations/{observation_id}"
        return self._get_json(url)

    def get_scores(self, page=None, limit=None, user_id=None, name=None):
        params = {"page": page, "limit": limit, "userId": user_id, "name": name}
        url = f"{self.BASE_URL}/api/public/scores"
        return self._get_json(url, params=params)

    def get_traces(self, page=None, limit=None, user_id=None, name=None):
        params = {"page": page, "limit": limit, "userId": user_id, "name": name}
        url = f"{self.BASE_URL}/api/public/traces"
        return self._get_json(url, params=params)

    def get_trace(self, trace_id):
        url = f"{self.BASE_URL}/api/public/traces/{trace_id}"
        return self._get_json(url)
