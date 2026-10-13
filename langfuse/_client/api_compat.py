from typing import Any

from typing_extensions import deprecated

from langfuse.api import AsyncLangfuseAPI, GetScoresV3Response, LangfuseAPI
from langfuse.api.scores.client import AsyncScoresClient, ScoresClient

_SCORES_V3_MESSAGE = (
    "`api.scores_v3.get_many_v3` is deprecated and will be removed in a future "
    "major release. Use `api.scores.get_many` instead."
)


class _ScoresV3:
    def __init__(self, scores: ScoresClient) -> None:
        self._scores = scores

    @deprecated(_SCORES_V3_MESSAGE)
    def get_many_v3(self, **kwargs: Any) -> GetScoresV3Response:
        return self._scores.get_many(**kwargs)


class _AsyncScoresV3:
    def __init__(self, scores: AsyncScoresClient) -> None:
        self._scores = scores

    @deprecated(_SCORES_V3_MESSAGE)
    async def get_many_v3(self, **kwargs: Any) -> GetScoresV3Response:
        return await self._scores.get_many(**kwargs)


class LangfuseAPIWithCompat(LangfuseAPI):
    @property
    def scores_v3(self) -> _ScoresV3:
        return _ScoresV3(self.scores)


class AsyncLangfuseAPIWithCompat(AsyncLangfuseAPI):
    @property
    def scores_v3(self) -> _AsyncScoresV3:
        return _AsyncScoresV3(self.scores)
