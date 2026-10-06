import base64
import json
import os
from dataclasses import dataclass
from typing import Any, Callable, Sequence, TypeVar
from uuid import uuid4

from langfuse.api import LangfuseAPI, ObservationV2, ScoreV3
from tests.support.retry import (
    DEFAULT_RETRY_INTERVAL_SECONDS,
    DEFAULT_RETRY_TIMEOUT_SECONDS,
    retry_until_ready,
)

READ_METHOD_NAMES = {"get", "get_by_id", "get_many", "get_many_v3", "get_run", "list"}
PAGINATION_ARGUMENTS = {"limit", "page", "cursor", "fields", "expand_metadata"}
T = TypeVar("T")

ALL_OBSERVATION_FIELDS = (
    "core,basic,time,io,metadata,model,usage,prompt,metrics,trace_context"
)
SCORE_FIELDS = "details,subject"

# The v2 observations API returns "" instead of null for unset string fields.
_EMPTY_AS_NONE_FIELDS = (
    "name",
    "status_message",
    "version",
    "user_id",
    "session_id",
    "model",
    "internal_model_id",
    "prompt_id",
    "prompt_name",
    "trace_name",
    "release",
)
_SDK_METADATA_KEY_PREFIXES = ("scope.", "resourceAttributes.")


def _has_filters(kwargs: dict[str, Any]) -> bool:
    return any(
        key not in PAGINATION_ARGUMENTS and value is not None
        for key, value in kwargs.items()
    )


class _RetryingApiProxy:
    def __init__(self, target: Any):
        self._target = target

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._target, name)

        if callable(attr):
            if name not in READ_METHOD_NAMES:
                return attr

            def _call(*args: Any, **kwargs: Any) -> Any:
                return retry_until_ready(
                    lambda: attr(*args, **kwargs),
                    is_result_ready=_result_ready(name, kwargs),
                )

            return _call

        if isinstance(attr, (str, bytes, int, float, bool, list, dict, tuple, set)):
            return attr

        if attr is None:
            return None

        return _RetryingApiProxy(attr)


def _result_ready(method_name: str, kwargs: dict[str, Any]):
    if method_name not in {"get_many", "get_many_v3", "list"} or not _has_filters(
        kwargs
    ):
        return None

    def _has_data(result: Any) -> bool:
        data = getattr(result, "data", None)
        return data is None or len(data) > 0

    return _has_data


def create_uuid():
    return str(uuid4())


def get_api(*, retry: bool = True):
    client = LangfuseAPI(
        username=os.environ.get("LANGFUSE_PUBLIC_KEY"),
        password=os.environ.get("LANGFUSE_SECRET_KEY"),
        base_url=os.environ.get("LANGFUSE_BASE_URL"),
    )
    return _RetryingApiProxy(client) if retry else client


def wait_for_result(
    operation: Callable[[], T],
    *,
    is_result_ready: Callable[[T], bool] | None = None,
    timeout_seconds: float = DEFAULT_RETRY_TIMEOUT_SECONDS,
    interval_seconds: float = DEFAULT_RETRY_INTERVAL_SECONDS,
) -> T:
    return retry_until_ready(
        operation,
        is_result_ready=is_result_ready,
        timeout_seconds=timeout_seconds,
        interval_seconds=interval_seconds,
    )


def wait_for_trace(
    trace_id: str,
    *,
    is_result_ready: Callable[[Any], bool] | None = None,
    timeout_seconds: float = DEFAULT_RETRY_TIMEOUT_SECONDS,
    interval_seconds: float = DEFAULT_RETRY_INTERVAL_SECONDS,
):
    """Read a trace through the v3 trace API.

    Unavailable on servers running the v4 `events_only` write mode; use
    `wait_for_observations` / `wait_for_root_observation` instead.
    """
    api = get_api(retry=False)
    return wait_for_result(
        lambda: api.trace.get(trace_id),
        is_result_ready=is_result_ready,
        timeout_seconds=timeout_seconds,
        interval_seconds=interval_seconds,
    )


def _parse_json_string(value: Any) -> Any:
    if not isinstance(value, str):
        return value

    try:
        return json.loads(value)
    except ValueError:
        return value


def normalize_observation(observation: ObservationV2) -> ObservationV2:
    """Map a v2 observation to the shape the SDK sent.

    The v2 API returns input/output as raw JSON strings and unset string
    fields as "". This parses JSON input/output and maps "" to None.
    """
    update: dict[str, Any] = {
        "input": _parse_json_string(observation.input),
        "output": _parse_json_string(observation.output),
    }
    for field in _EMPTY_AS_NONE_FIELDS:
        if getattr(observation, field, None) == "":
            update[field] = None

    return observation.model_copy(update=update)


def user_metadata(observation: ObservationV2) -> dict[str, Any]:
    """Observation metadata without the scope/resource keys the server adds."""
    metadata = observation.metadata or {}
    assert isinstance(metadata, dict), metadata

    return {
        key: value
        for key, value in metadata.items()
        if not key.startswith(_SDK_METADATA_KEY_PREFIXES)
    }


def get_observations(
    *,
    fields: str = ALL_OBSERVATION_FIELDS,
    api: Any = None,
    **filters: Any,
) -> list[ObservationV2]:
    """Fetch all observations matching `filters` (all pages), oldest first."""
    api = api or get_api(retry=False)
    observations: list[ObservationV2] = []
    cursor = None

    while True:
        response = api.observations.get_many(
            fields=fields, limit=1000, cursor=cursor, **filters
        )
        observations.extend(response.data)
        cursor = response.meta.cursor
        if not cursor or not response.data:
            break

    return sorted(
        (normalize_observation(observation) for observation in observations),
        key=lambda observation: observation.start_time,
    )


def wait_for_observations(
    trace_id: str | None = None,
    *,
    min_count: int = 1,
    is_result_ready: Callable[[list[ObservationV2]], bool] | None = None,
    fields: str = ALL_OBSERVATION_FIELDS,
    timeout_seconds: float = DEFAULT_RETRY_TIMEOUT_SECONDS,
    interval_seconds: float = DEFAULT_RETRY_INTERVAL_SECONDS,
    **filters: Any,
) -> list[ObservationV2]:
    """Poll the v2 observations API until at least `min_count` observations
    match (and `is_result_ready` holds), then return them oldest first."""
    if trace_id is not None:
        filters["trace_id"] = trace_id
    assert filters, "wait_for_observations needs at least one filter"

    def _ready(observations: list[ObservationV2]) -> bool:
        return len(observations) >= min_count and (
            is_result_ready is None or is_result_ready(observations)
        )

    return wait_for_result(
        lambda: get_observations(fields=fields, **filters),
        is_result_ready=_ready,
        timeout_seconds=timeout_seconds,
        interval_seconds=interval_seconds,
    )


def get_root_observation(observations: Sequence[ObservationV2]) -> ObservationV2:
    """Return the single root observation, which carries the trace-level
    attributes (trace name, user, session, tags, public) and trace IO."""
    roots = [
        observation for observation in observations if observation.is_root_observation
    ]
    assert len(roots) == 1, (
        f"expected exactly one root observation, got {[r.name for r in roots]}"
    )

    return roots[0]


def wait_for_root_observation(
    trace_id: str,
    *,
    min_count: int = 1,
    is_result_ready: Callable[[ObservationV2], bool] | None = None,
    timeout_seconds: float = DEFAULT_RETRY_TIMEOUT_SECONDS,
    interval_seconds: float = DEFAULT_RETRY_INTERVAL_SECONDS,
) -> ObservationV2:
    def _ready(observations: list[ObservationV2]) -> bool:
        roots = [o for o in observations if o.is_root_observation]
        return len(roots) == 1 and (
            is_result_ready is None or is_result_ready(roots[0])
        )

    return get_root_observation(
        wait_for_observations(
            trace_id,
            min_count=min_count,
            is_result_ready=_ready,
            timeout_seconds=timeout_seconds,
            interval_seconds=interval_seconds,
        )
    )


@dataclass(frozen=True)
class TraceSnapshot:
    """A trace as v4 exposes it: its observations plus (optionally) scores.

    Mirrors how the platform aggregates events into a trace: input, output
    and metadata come from the root observation; name, user, session,
    version, release and environment are the latest non-empty value across
    all observations; tags are the union; public is true if any observation
    is public.
    """

    id: str
    observations: list[ObservationV2]
    scores: list[ScoreV3] | None = None

    @property
    def root(self) -> ObservationV2:
        return get_root_observation(self.observations)

    def _latest(self, field: str) -> Any:
        values = [
            getattr(observation, field)
            for observation in self.observations
            if getattr(observation, field)
        ]
        return values[-1] if values else None

    @property
    def name(self) -> str | None:
        # The API falls back to the root's own name when no trace name was
        # set, so a root whose trace_name equals its name carries no signal.
        explicit_names = [
            observation.trace_name
            for observation in self.observations
            if observation.trace_name
            and not (
                observation.is_root_observation
                and observation.trace_name == observation.name
            )
        ]
        return explicit_names[-1] if explicit_names else self.root.trace_name

    @property
    def user_id(self) -> str | None:
        return self._latest("user_id")

    @property
    def session_id(self) -> str | None:
        return self._latest("session_id")

    @property
    def tags(self) -> list[str]:
        return sorted(
            {tag for observation in self.observations for tag in observation.tags or []}
        )

    @property
    def public(self) -> bool:
        return any(observation.public for observation in self.observations)

    @property
    def version(self) -> str | None:
        return self._latest("version")

    @property
    def release(self) -> str | None:
        return self._latest("release")

    @property
    def environment(self) -> str | None:
        return self._latest("environment")

    @property
    def input(self) -> Any:
        return self.root.input

    @property
    def output(self) -> Any:
        return self.root.output

    @property
    def metadata(self) -> dict[str, Any]:
        return user_metadata(self.root)


def wait_for_trace_snapshot(
    trace_id: str,
    *,
    min_observations: int = 1,
    min_scores: int | None = None,
    is_result_ready: Callable[[TraceSnapshot], bool] | None = None,
    timeout_seconds: float = DEFAULT_RETRY_TIMEOUT_SECONDS,
    interval_seconds: float = DEFAULT_RETRY_INTERVAL_SECONDS,
) -> TraceSnapshot:
    """Poll until the trace has `min_observations` observations (and
    `min_scores` scores, which are only fetched when given)."""

    def _fetch() -> TraceSnapshot:
        return TraceSnapshot(
            id=trace_id,
            observations=get_observations(trace_id=trace_id),
            scores=None if min_scores is None else get_scores(trace_id=trace_id),
        )

    def _ready(snapshot: TraceSnapshot) -> bool:
        if len(snapshot.observations) < min_observations:
            return False
        if min_scores is not None and len(snapshot.scores or []) < min_scores:
            return False
        if is_result_ready is None:
            return True
        try:
            return is_result_ready(snapshot)
        except AssertionError:
            # Root-derived attributes are unavailable until the root arrives.
            return False

    return wait_for_result(
        _fetch,
        is_result_ready=_ready,
        timeout_seconds=timeout_seconds,
        interval_seconds=interval_seconds,
    )


def get_scores(
    *, fields: str = SCORE_FIELDS, api: Any = None, **filters: Any
) -> list[ScoreV3]:
    api = api or get_api(retry=False)
    scores: list[ScoreV3] = []
    cursor = None

    while True:
        response = api.scores_v3.get_many_v3(
            fields=fields, limit=100, cursor=cursor, **filters
        )
        scores.extend(response.data)
        cursor = response.meta.cursor
        if not cursor or not response.data:
            break

    return scores


def wait_for_scores(
    *,
    min_count: int = 1,
    is_result_ready: Callable[[list[ScoreV3]], bool] | None = None,
    fields: str = SCORE_FIELDS,
    timeout_seconds: float = DEFAULT_RETRY_TIMEOUT_SECONDS,
    interval_seconds: float = DEFAULT_RETRY_INTERVAL_SECONDS,
    **filters: Any,
) -> list[ScoreV3]:
    """Poll the v3 scores API (filters: trace_id, session_id, observation_id,
    name, ...) until at least `min_count` scores match."""
    assert filters, "wait_for_scores needs at least one filter"

    def _ready(scores: list[ScoreV3]) -> bool:
        return len(scores) >= min_count and (
            is_result_ready is None or is_result_ready(scores)
        )

    return wait_for_result(
        lambda: get_scores(fields=fields, **filters),
        is_result_ready=_ready,
        timeout_seconds=timeout_seconds,
        interval_seconds=interval_seconds,
    )


def encode_file_to_base64(image_path) -> str:
    with open(image_path, "rb") as file:
        return base64.b64encode(file.read()).decode("utf-8")
