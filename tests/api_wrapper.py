import requests


class LangfuseAPI:
    def __init__(self, username=None, password=None, base_url="http://localhost:3000"):
        self.auth = (username, password) if username and password else None
        self.BASE_URL = base_url

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
