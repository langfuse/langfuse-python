from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any, Iterable

from langfuse.api.commons.errors.not_found_error import NotFoundError

OBSERVATION_FIELDS = (
    "core,basic,time,io,metadata,model,usage,prompt,metrics,trace_context"
)
DEFAULT_LOOKBACK = timedelta(days=1)
DEFAULT_LOOKAHEAD = timedelta(minutes=5)


def _time_bounds() -> tuple[datetime, datetime]:
    now = datetime.now(timezone.utc)
    return now - DEFAULT_LOOKBACK, now + DEFAULT_LOOKAHEAD


def _parse_json(value: Any) -> Any:
    if not isinstance(value, str):
        return value

    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return value


def _as_float(value: Any) -> float | None:
    if value is None:
        return None

    return float(value)


def _none_if_empty(value: Any) -> Any:
    return None if value == "" else value


def _score_view(score: Any) -> SimpleNamespace:
    subject = getattr(score, "subject", None)
    kind = getattr(subject, "kind", None)
    value = score.value
    score_type = type(score).__name__.split("_")[-1].removesuffix("ScoreV3").upper()
    string_value = value if score_type in {"CATEGORICAL", "TEXT"} else None
    trace_id = None
    if subject is not None:
        trace_id = getattr(subject, "trace_id", None)
        if trace_id is None and kind == "trace":
            trace_id = subject.id

    data = score.model_dump()
    data.update(
        data_type=score_type,
        string_value=string_value,
        trace_id=trace_id,
        observation_id=(
            subject.id if subject is not None and kind == "observation" else None
        ),
        session_id=subject.id if subject is not None and kind == "session" else None,
    )
    return SimpleNamespace(**data)


class V4ObservationView:
    """Test-friendly view of one row returned by Observations API v2."""

    def __init__(self, observation: Any):
        self._observation = observation
        self.id = observation.id
        self.trace_id = observation.trace_id
        self.type = observation.type
        self.name = observation.name
        self.start_time = observation.start_time
        self.end_time = observation.end_time
        self.completion_start_time = observation.completion_start_time
        self.model = (
            getattr(observation, "model", None) or observation.provided_model_name
        )
        self.model_parameters = _parse_json(observation.model_parameters)
        self.input = _parse_json(observation.input)
        self.output = _parse_json(observation.output)
        self.version = observation.version
        self.metadata = _parse_json(observation.metadata) or {}
        self.level = observation.level
        self.status_message = observation.status_message
        self.parent_observation_id = observation.parent_observation_id
        self.is_root_observation = observation.is_root_observation
        self.prompt_id = _none_if_empty(observation.prompt_id)
        self.prompt_name = _none_if_empty(observation.prompt_name)
        self.prompt_version = observation.prompt_version
        self.usage_details = observation.usage_details or {}
        self.cost_details = observation.cost_details or {}
        self.environment = observation.environment
        self.latency = observation.latency
        self.time_to_first_token = observation.time_to_first_token
        self.model_id = observation.model_id
        self.input_price = _as_float(observation.input_price)
        self.output_price = _as_float(observation.output_price)
        self.total_price = _as_float(observation.total_price)
        self.calculated_input_cost = self.cost_details.get("input")
        self.calculated_output_cost = self.cost_details.get("output")
        self.calculated_total_cost = observation.total_cost or self.cost_details.get(
            "total"
        )
        self.usage = SimpleNamespace(
            input=self.usage_details.get("input", 0),
            output=self.usage_details.get("output", 0),
            total=self.usage_details.get("total", 0),
            unit=self.usage_details.get("unit"),
            input_cost=self.calculated_input_cost,
            output_cost=self.calculated_output_cost,
            total_cost=self.calculated_total_cost,
        )


class V4TraceView:
    """Observation-backed trace view used by SDK behavior tests."""

    def __init__(
        self, trace_id: str, observations: list[V4ObservationView], scores: list
    ):
        if not observations:
            raise NotFoundError(body={"error": "LangfuseNotFoundError"})

        observations = list(
            {observation.id: observation for observation in observations}.values()
        )
        observations.sort(key=lambda observation: observation.start_time)
        root = next(
            (
                observation
                for observation in observations
                if observation.is_root_observation is True
            ),
            None,
        )
        if root is None:
            root = next(
                (
                    observation
                    for observation in observations
                    if observation.parent_observation_id is None
                ),
                observations[0],
            )

        source = root._observation

        def first_populated(attribute: str, fallback: Any = None) -> Any:
            return next(
                (
                    value
                    for observation in observations
                    if (value := getattr(observation._observation, attribute, None))
                ),
                fallback,
            )

        self.id = trace_id
        self.timestamp = root.start_time
        self.name = first_populated("trace_name", root.name)
        self.input = root.input
        self.output = root.output
        self.session_id = first_populated("session_id")
        self.release = first_populated("release")
        self.version = root.version
        self.user_id = first_populated("user_id")
        self.metadata = max(
            (observation.metadata for observation in observations),
            key=len,
            default={},
        )
        self.tags = list(
            dict.fromkeys(
                tag
                for observation in observations
                for tag in observation._observation.tags
            )
        )
        self.public = bool(source.public)
        self.environment = source.environment
        self.observations = observations
        self.scores = scores


class _V4ObservationsClient:
    def __init__(self, client: Any):
        self._client = client

    def get(self, observation_id: str) -> V4ObservationView:
        response = self.get_many(
            filter=json.dumps(
                [
                    {
                        "type": "string",
                        "column": "id",
                        "operator": "=",
                        "value": observation_id,
                    }
                ]
            ),
            limit=1,
        )
        if not response.data:
            raise NotFoundError(body={"error": "LangfuseNotFoundError"})

        return response.data[0]

    def get_many(
        self,
        *,
        fields: str | None = None,
        limit: int | None = None,
        cursor: str | None = None,
        page: int | None = None,
        name: str | None = None,
        user_id: str | None = None,
        type: str | None = None,
        trace_id: str | None = None,
        parent_observation_id: str | None = None,
        environment: str | Iterable[str] | None = None,
        version: str | None = None,
        filter: str | None = None,
        **_: Any,
    ) -> SimpleNamespace:
        from_start_time, to_start_time = _time_bounds()
        requested_page = page or 1
        response = None
        current_cursor = cursor
        fetched_page = 0
        for fetched_page in range(1, requested_page + 1):
            response = self._client.observations.get_many(
                fields=fields or OBSERVATION_FIELDS,
                limit=limit,
                cursor=current_cursor,
                name=name,
                user_id=user_id,
                type=type,
                trace_id=trace_id,
                parent_observation_id=parent_observation_id,
                environment=environment,
                version=version,
                filter=filter,
                from_start_time=from_start_time,
                to_start_time=to_start_time,
            )
            current_cursor = response.meta.cursor
            if current_cursor is None and fetched_page < requested_page:
                break

        if response is None or fetched_page < requested_page:
            return SimpleNamespace(
                data=[],
                meta=SimpleNamespace(
                    cursor=None,
                    total_items=0,
                    total_pages=requested_page - 1,
                    page=requested_page,
                    limit=limit or 50,
                ),
            )

        data = [V4ObservationView(observation) for observation in response.data]
        return SimpleNamespace(
            data=data,
            meta=SimpleNamespace(
                cursor=response.meta.cursor,
                total_items=len(data),
                total_pages=1,
                page=requested_page,
                limit=limit or 50,
            ),
        )


class _V4TraceClient:
    def __init__(self, client: Any, observations: _V4ObservationsClient):
        self._client = client
        self._observations = observations

    def _scores(self, trace_id: str) -> list[Any]:
        response = self._client.scores_v3.get_many_v3(
            trace_id=trace_id,
            fields="details,subject,annotation",
            limit=100,
        )
        return [_score_view(score) for score in response.data]

    def get(self, trace_id: str | None = None, **kwargs: Any) -> V4TraceView:
        resolved_trace_id = trace_id or kwargs["trace_id"]
        response = self._observations.get_many(trace_id=resolved_trace_id, limit=1000)
        return V4TraceView(
            resolved_trace_id,
            response.data,
            self._scores(resolved_trace_id),
        )

    def list(
        self,
        *,
        name: str | None = None,
        session_id: str | None = None,
        user_id: str | None = None,
        filter: str | None = None,
        limit: int | None = None,
        page: int | None = None,
        **_: Any,
    ) -> SimpleNamespace:
        filters: list[dict[str, Any]] = []
        if filter:
            filters.extend(json.loads(filter))
        if name is not None:
            filters.append(
                {
                    "type": "string",
                    "column": "traceName",
                    "operator": "=",
                    "value": name,
                }
            )
        if session_id is not None:
            filters.append(
                {
                    "type": "string",
                    "column": "sessionId",
                    "operator": "=",
                    "value": session_id,
                }
            )

        response = self._observations.get_many(
            user_id=user_id,
            filter=json.dumps(filters) if filters else None,
            limit=1000,
        )
        observations_by_trace: dict[str, list[V4ObservationView]] = {}
        for observation in response.data:
            if observation.trace_id is not None:
                observations_by_trace.setdefault(observation.trace_id, []).append(
                    observation
                )

        traces = [
            V4TraceView(trace_id, observations, self._scores(trace_id))
            for trace_id, observations in observations_by_trace.items()
        ]
        traces.sort(key=lambda trace: trace.timestamp, reverse=True)

        requested_limit = limit or 50
        requested_page = page or 1
        start = (requested_page - 1) * requested_limit
        data = traces[start : start + requested_limit]
        total_items = len(traces)
        total_pages = (total_items + requested_limit - 1) // requested_limit
        return SimpleNamespace(
            data=data,
            meta=SimpleNamespace(
                page=requested_page,
                limit=requested_limit,
                total_items=total_items,
                total_pages=total_pages,
            ),
        )


class _V4SessionsClient:
    def __init__(self, traces: _V4TraceClient):
        self._traces = traces

    def list(
        self, *, limit: int | None = None, page: int | None = None
    ) -> SimpleNamespace:
        traces = self._traces.list(limit=1000).data
        sessions: dict[str, list[V4TraceView]] = {}
        for trace in traces:
            if trace.session_id is not None:
                sessions.setdefault(trace.session_id, []).append(trace)

        data = [
            SimpleNamespace(id=session_id, traces=session_traces)
            for session_id, session_traces in sessions.items()
        ]
        requested_limit = limit or 50
        requested_page = page or 1
        start = (requested_page - 1) * requested_limit
        page_data = data[start : start + requested_limit]
        total_items = len(data)
        return SimpleNamespace(
            data=page_data,
            meta=SimpleNamespace(
                page=requested_page,
                limit=requested_limit,
                total_items=total_items,
                total_pages=(total_items + requested_limit - 1) // requested_limit,
            ),
        )


class _V4ScoresClient:
    def __init__(self, client: Any):
        self._client = client

    def get_by_id(self, score_id: str) -> Any:
        response = self._client.scores_v3.get_many_v3(
            id=score_id,
            fields="details,subject,annotation",
            limit=1,
        )
        if not response.data:
            raise NotFoundError(body={"error": "LangfuseNotFoundError"})

        return _score_view(response.data[0])


class _V4DatasetsClient:
    def __init__(self, client: Any):
        self._client = client
        self._datasets = client.datasets

    def __getattr__(self, name: str) -> Any:
        return getattr(self._datasets, name)

    def get_run(self, dataset_name: str, run_name: str) -> Any:
        dataset = self._datasets.get(dataset_name)
        from_start_time, to_start_time = _time_bounds()
        response = self._client.experiments.list(
            from_start_time=from_start_time,
            to_start_time=to_start_time,
            dataset_id=dataset.id,
            name=run_name,
            fields="core,metadata,scores",
            limit=1,
        )
        if not response.data:
            raise NotFoundError(body={"error": "LangfuseNotFoundError"})

        return response.data[0]

    def get_runs(self, dataset_name: str) -> SimpleNamespace:
        dataset = self._datasets.get(dataset_name)
        from_start_time, to_start_time = _time_bounds()
        return self._client.experiments.list(
            from_start_time=from_start_time,
            to_start_time=to_start_time,
            dataset_id=dataset.id,
            fields="core,metadata,scores",
            limit=100,
        )


class _V4ExperimentItemsClient:
    def __init__(self, client: Any):
        self._client = client

    def list(self, *, dataset_id: str, run_name: str) -> Any:
        from_start_time, to_start_time = _time_bounds()
        experiments = self._client.experiments.list(
            from_start_time=from_start_time,
            to_start_time=to_start_time,
            dataset_id=dataset_id,
            name=run_name,
            limit=1,
        )
        if not experiments.data:
            raise NotFoundError(body={"error": "LangfuseNotFoundError"})

        return self._client.experiments.list_items(
            from_start_time=from_start_time,
            to_start_time=to_start_time,
            experiment_id=experiments.data[0].id,
            fields=("core,dataset,io,metadata,itemMetadata,experimentMetadata,scores"),
            limit=100,
        )


class V4TestAPI:
    """Expose v4-native read paths behind the test suite's existing API helper."""

    def __init__(self, client: Any):
        self._client = client
        self.observations = _V4ObservationsClient(client)
        self.trace = _V4TraceClient(client, self.observations)
        self.legacy = SimpleNamespace(observations_v1=self.observations)
        self.sessions = _V4SessionsClient(self.trace)
        self.scores = _V4ScoresClient(client)
        self.datasets = _V4DatasetsClient(client)
        self.dataset_run_items = _V4ExperimentItemsClient(client)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)
