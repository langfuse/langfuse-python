import os

import httpx


class LangfuseAPI:
    def __init__(self, username=None, password=None, base_url=None):
        username = username if username else os.environ["LANGFUSE_PUBLIC_KEY"]
        password = password if password else os.environ["LANGFUSE_SECRET_KEY"]
        self.auth = (username, password)
        self.BASE_URL = base_url if base_url else os.environ["LANGFUSE_HOST"]

    def get_observation(self, observation_id):
        url = f"{self.BASE_URL}/api/public/observations/{observation_id}"
        response = httpx.get(url, auth=self.auth)
        return response.json()

    def get_scores(self, page=None, limit=None, user_id=None, name=None):
        params = {"page": page, "limit": limit, "userId": user_id, "name": name}
        url = f"{self.BASE_URL}/api/public/scores"
        response = httpx.get(url, params=params, auth=self.auth)
        return response.json()

    def get_traces(self, page=None, limit=None, user_id=None, name=None):
        params = {"page": page, "limit": limit, "userId": user_id, "name": name}
        url = f"{self.BASE_URL}/api/public/traces"
        response = httpx.get(url, params=params, auth=self.auth)
        return response.json()

    def get_trace(self, trace_id):
        url = f"{self.BASE_URL}/api/public/traces/{trace_id}"
        response = httpx.get(url, auth=self.auth)
        return response.json()
