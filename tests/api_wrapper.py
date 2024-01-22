import os

import requests
import logging


class LangfuseAPI:
    def __init__(self, username=None, password=None, base_url=None):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        try:
            username = username or os.environ["LANGFUSE_PUBLIC_KEY"]
            password = password or os.environ["LANGFUSE_SECRET_KEY"]
            self.auth = (username, password)
            self.BASE_URL = base_url or os.environ["LANGFUSE_HOST"]
        except KeyError as e:
            logging.error(f"Environment variable {e} is not set.")

    def get_observation(self, observation_id):
        url = f"{self.BASE_URL}/api/public/observations/{observation_id}"
        response = requests.get(url, auth=self.auth)
        return response.json()

    def get_scores(self, page=None, limit=None, user_id=None, name=None):
        params = {"page": page, "limit": limit, "userId": user_id, "name": name}
        url = f"{self.BASE_URL}/api/public/scores"
        response = requests.get(url, params=params, auth=self.auth)
        return response.json()

    def get_traces(self, page=None, limit=None, user_id=None, name=None):
        params = {"page": page, "limit": limit, "userId": user_id, "name": name}
        url = f"{self.BASE_URL}/api/public/traces"
        response = requests.get(url, params=params, auth=self.auth)
        return response.json()

    def get_trace(self, trace_id):
        url = f"{self.BASE_URL}/api/public/traces/{trace_id}"
        response = requests.get(url, auth=self.auth)
        return response.json()
