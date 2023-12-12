# This file was auto-generated by Fern from our API Definition.

import typing

import httpx

from .core.client_wrapper import AsyncClientWrapper, SyncClientWrapper
from .resources.dataset_items.client import AsyncDatasetItemsClient, DatasetItemsClient
from .resources.dataset_run_items.client import AsyncDatasetRunItemsClient, DatasetRunItemsClient
from .resources.datasets.client import AsyncDatasetsClient, DatasetsClient
from .resources.event.client import AsyncEventClient, EventClient
from .resources.generations.client import AsyncGenerationsClient, GenerationsClient
from .resources.health.client import AsyncHealthClient, HealthClient
from .resources.ingestion.client import AsyncIngestionClient, IngestionClient
from .resources.observations.client import AsyncObservationsClient, ObservationsClient
from .resources.projects.client import AsyncProjectsClient, ProjectsClient
from .resources.score.client import AsyncScoreClient, ScoreClient
from .resources.sessions.client import AsyncSessionsClient, SessionsClient
from .resources.span.client import AsyncSpanClient, SpanClient
from .resources.trace.client import AsyncTraceClient, TraceClient


class FernLangfuse:
    def __init__(
        self,
        *,
        base_url: str,
        x_langfuse_sdk_name: typing.Optional[str] = None,
        x_langfuse_sdk_version: typing.Optional[str] = None,
        x_langfuse_public_key: typing.Optional[str] = None,
        username: typing.Optional[typing.Union[str, typing.Callable[[], str]]] = None,
        password: typing.Optional[typing.Union[str, typing.Callable[[], str]]] = None,
        timeout: typing.Optional[float] = 60,
        httpx_client: typing.Optional[httpx.Client] = None
    ):
        self._client_wrapper = SyncClientWrapper(
            base_url=base_url,
            x_langfuse_sdk_name=x_langfuse_sdk_name,
            x_langfuse_sdk_version=x_langfuse_sdk_version,
            x_langfuse_public_key=x_langfuse_public_key,
            username=username,
            password=password,
            httpx_client=httpx.Client(timeout=timeout) if httpx_client is None else httpx_client,
        )
        self.dataset_items = DatasetItemsClient(client_wrapper=self._client_wrapper)
        self.dataset_run_items = DatasetRunItemsClient(client_wrapper=self._client_wrapper)
        self.datasets = DatasetsClient(client_wrapper=self._client_wrapper)
        self.event = EventClient(client_wrapper=self._client_wrapper)
        self.generations = GenerationsClient(client_wrapper=self._client_wrapper)
        self.health = HealthClient(client_wrapper=self._client_wrapper)
        self.ingestion = IngestionClient(client_wrapper=self._client_wrapper)
        self.observations = ObservationsClient(client_wrapper=self._client_wrapper)
        self.projects = ProjectsClient(client_wrapper=self._client_wrapper)
        self.score = ScoreClient(client_wrapper=self._client_wrapper)
        self.sessions = SessionsClient(client_wrapper=self._client_wrapper)
        self.span = SpanClient(client_wrapper=self._client_wrapper)
        self.trace = TraceClient(client_wrapper=self._client_wrapper)


class AsyncFernLangfuse:
    def __init__(
        self,
        *,
        base_url: str,
        x_langfuse_sdk_name: typing.Optional[str] = None,
        x_langfuse_sdk_version: typing.Optional[str] = None,
        x_langfuse_public_key: typing.Optional[str] = None,
        username: typing.Optional[typing.Union[str, typing.Callable[[], str]]] = None,
        password: typing.Optional[typing.Union[str, typing.Callable[[], str]]] = None,
        timeout: typing.Optional[float] = 60,
        httpx_client: typing.Optional[httpx.AsyncClient] = None
    ):
        self._client_wrapper = AsyncClientWrapper(
            base_url=base_url,
            x_langfuse_sdk_name=x_langfuse_sdk_name,
            x_langfuse_sdk_version=x_langfuse_sdk_version,
            x_langfuse_public_key=x_langfuse_public_key,
            username=username,
            password=password,
            httpx_client=httpx.AsyncClient(timeout=timeout) if httpx_client is None else httpx_client,
        )
        self.dataset_items = AsyncDatasetItemsClient(client_wrapper=self._client_wrapper)
        self.dataset_run_items = AsyncDatasetRunItemsClient(client_wrapper=self._client_wrapper)
        self.datasets = AsyncDatasetsClient(client_wrapper=self._client_wrapper)
        self.event = AsyncEventClient(client_wrapper=self._client_wrapper)
        self.generations = AsyncGenerationsClient(client_wrapper=self._client_wrapper)
        self.health = AsyncHealthClient(client_wrapper=self._client_wrapper)
        self.ingestion = AsyncIngestionClient(client_wrapper=self._client_wrapper)
        self.observations = AsyncObservationsClient(client_wrapper=self._client_wrapper)
        self.projects = AsyncProjectsClient(client_wrapper=self._client_wrapper)
        self.score = AsyncScoreClient(client_wrapper=self._client_wrapper)
        self.sessions = AsyncSessionsClient(client_wrapper=self._client_wrapper)
        self.span = AsyncSpanClient(client_wrapper=self._client_wrapper)
        self.trace = AsyncTraceClient(client_wrapper=self._client_wrapper)
