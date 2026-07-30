from tests.support.retry import (
    DEFAULT_RETRY_INTERVAL_SECONDS,
    DEFAULT_RETRY_TIMEOUT_SECONDS,
)
from tests.support.utils import get_api, wait_for_result
from tests.support.v4_api import _score_view


class LangfuseAPI:
    def __init__(self, username=None, password=None, base_url=None):
        self._api = get_api(retry=False)

    @staticmethod
    def _score_json(score):
        return {
            "id": score.id,
            "name": score.name,
            "value": score.value,
            "timestamp": score.timestamp.isoformat(),
            "dataType": score.data_type,
            "stringValue": score.string_value,
            "traceId": score.trace_id,
            "observationId": score.observation_id,
            "sessionId": score.session_id,
            "comment": getattr(score, "comment", None),
            "metadata": getattr(score, "metadata", None),
        }

    @classmethod
    def _observation_json(cls, observation):
        return {
            "id": observation.id,
            "traceId": observation.trace_id,
            "type": observation.type,
            "name": observation.name,
            "startTime": observation.start_time.isoformat(),
            "endTime": (
                observation.end_time.isoformat() if observation.end_time else None
            ),
            "input": observation.input,
            "output": observation.output,
            "metadata": observation.metadata,
            "model": observation.model,
            "usageDetails": observation.usage_details,
            "costDetails": observation.cost_details,
            "promptId": observation.prompt_id,
        }

    @classmethod
    def _trace_json(cls, trace):
        return {
            "id": trace.id,
            "timestamp": trace.timestamp.isoformat(),
            "name": trace.name,
            "input": trace.input,
            "output": trace.output,
            "sessionId": trace.session_id,
            "release": trace.release,
            "version": trace.version,
            "userId": trace.user_id,
            "metadata": trace.metadata,
            "tags": trace.tags,
            "public": trace.public,
            "environment": trace.environment,
            "observations": [
                cls._observation_json(observation) for observation in trace.observations
            ],
            "scores": [cls._score_json(score) for score in trace.scores],
        }

    @staticmethod
    def _read(
        operation,
        *,
        retry,
        is_result_ready,
        timeout_seconds,
        interval_seconds,
    ):
        if not retry:
            return operation()

        return wait_for_result(
            operation,
            is_result_ready=is_result_ready,
            timeout_seconds=timeout_seconds,
            interval_seconds=interval_seconds,
        )

    def get_observation(
        self,
        observation_id,
        *,
        retry=True,
        is_result_ready=None,
        timeout_seconds=DEFAULT_RETRY_TIMEOUT_SECONDS,
        interval_seconds=DEFAULT_RETRY_INTERVAL_SECONDS,
    ):
        return self._read(
            lambda: self._observation_json(self._api.observations.get(observation_id)),
            retry=retry,
            is_result_ready=is_result_ready,
            timeout_seconds=timeout_seconds,
            interval_seconds=interval_seconds,
        )

    def get_scores(
        self,
        page=None,
        limit=None,
        user_id=None,
        name=None,
        *,
        retry=True,
        is_result_ready=None,
        timeout_seconds=DEFAULT_RETRY_TIMEOUT_SECONDS,
        interval_seconds=DEFAULT_RETRY_INTERVAL_SECONDS,
    ):
        def operation():
            response = self._api._client.scores_v3.get_many_v3(
                name=name,
                user_id=user_id,
                limit=limit,
                fields="details,subject,annotation",
            )
            data = [self._score_json(_score_view(score)) for score in response.data]
            return {
                "data": data,
                "meta": {"page": page or 1, "limit": limit or 50},
            }

        return self._read(
            operation,
            retry=retry,
            is_result_ready=is_result_ready,
            timeout_seconds=timeout_seconds,
            interval_seconds=interval_seconds,
        )

    def get_traces(
        self,
        page=None,
        limit=None,
        user_id=None,
        name=None,
        *,
        retry=True,
        is_result_ready=None,
        timeout_seconds=DEFAULT_RETRY_TIMEOUT_SECONDS,
        interval_seconds=DEFAULT_RETRY_INTERVAL_SECONDS,
    ):
        def operation():
            response = self._api.trace.list(
                page=page, limit=limit, user_id=user_id, name=name
            )
            return {
                "data": [self._trace_json(trace) for trace in response.data],
                "meta": vars(response.meta),
            }

        return self._read(
            operation,
            retry=retry,
            is_result_ready=is_result_ready,
            timeout_seconds=timeout_seconds,
            interval_seconds=interval_seconds,
        )

    def get_trace(
        self,
        trace_id,
        *,
        retry=True,
        is_result_ready=None,
        timeout_seconds=DEFAULT_RETRY_TIMEOUT_SECONDS,
        interval_seconds=DEFAULT_RETRY_INTERVAL_SECONDS,
    ):
        return self._read(
            lambda: self._trace_json(self._api.trace.get(trace_id)),
            retry=retry,
            is_result_ready=is_result_ready,
            timeout_seconds=timeout_seconds,
            interval_seconds=interval_seconds,
        )
