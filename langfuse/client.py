from enum import Enum
import logging
import os
from typing import Optional
import typing
import uuid


import datetime as dt

from langfuse.api.client import FernLangfuse
from datetime import datetime
from langfuse.api.resources.commons.types.create_event_request import CreateEventRequest
from langfuse.api.resources.commons.types.create_generation_request import CreateGenerationRequest
from langfuse.api.resources.commons.types.create_span_request import CreateSpanRequest
from langfuse.api.resources.commons.types.dataset import Dataset
from langfuse.api.resources.commons.types.dataset_item import DatasetItem
from langfuse.api.resources.commons.types.observation import Observation
from langfuse.api.resources.commons.types.dataset_status import DatasetStatus
from langfuse.api.resources.score.types.create_score_request import CreateScoreRequest
from langfuse.api.resources.trace.types.create_trace_request import CreateTraceRequest
from langfuse.environment import get_common_release_envs
from langfuse.logging import clean_logger
from langfuse.model import (
    DatasetItem,
    CreateDatasetRunItemRequest,
    CreateDatasetRequest,
    CreateEvent,
    CreateGeneration,
    CreateScore,
    CreateSpan,
    CreateTrace,
    InitialGeneration,
    InitialScore,
    InitialSpan,
    UpdateGeneration,
    UpdateSpan,
    CreateDatasetItemRequest,
    DatasetRun,
)
from langfuse.api.resources.commons.types.trace_with_full_details import TraceWithFullDetails
from langfuse.api.resources.generations.types.update_generation_request import UpdateGenerationRequest
from langfuse.api.resources.span.types.update_span_request import UpdateSpanRequest
from langfuse.request import LangfuseClient
from langfuse.task_manager import TaskManager
from langfuse.utils import convert_observation_to_event
from .version import __version__ as version


class Langfuse(object):
    log = logging.getLogger("langfuse")

    def __init__(
        self,
        public_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        host: Optional[str] = None,
        release: Optional[str] = None,
        debug: bool = False,
        threads: int = 1,
        flush_at: int = 50,
        flush_interval: int = 0.5,
        max_retries=3,
        timeout=15,
    ):
        set_debug = debug if debug else (os.getenv("LANGFUSE_DEBUG", "False") == "True")

        if set_debug is True:
            # Ensures that debug level messages are logged when debug mode is on.
            # Otherwise, defaults to WARNING level.
            # See https://docs.python.org/3/howto/logging.html#what-happens-if-no-configuration-is-provided
            logging.basicConfig()
            self.log.setLevel(logging.DEBUG)

            clean_logger()
        else:
            self.log.setLevel(logging.WARNING)
            clean_logger()

        public_key = public_key if public_key else os.environ.get("LANGFUSE_PUBLIC_KEY")
        secret_key = secret_key if secret_key else os.environ.get("LANGFUSE_SECRET_KEY")
        self.base_url = host if host else os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com")

        if not public_key:
            self.log.warning("public_key is not set.")
            raise ValueError("public_key is required, set as parameter or environment variable 'LANGFUSE_PUBLIC_KEY'")

        if not secret_key:
            self.log.warning("secret_key is not set.")
            raise ValueError("secret_key is required, set as parameter or environment variable 'LANGFUSE_SECRET_KEY'")

        self.client = FernLangfuse(
            base_url=self.base_url,
            username=public_key,
            password=secret_key,
            x_langfuse_sdk_name="python",
            x_langfuse_sdk_version=version,
            x_langfuse_public_key=public_key,
        )

        langfuse_client = LangfuseClient(
            public_key=public_key,
            secret_key=secret_key,
            base_url=self.base_url,
            version=version,
            timeout=timeout,
        )

        args = {"threads": threads, "flush_at": flush_at, "flush_interval": flush_interval, "max_retries": max_retries, "client": langfuse_client}

        if threads is not None:
            args["threads"] = threads

        self.task_manager = TaskManager(**args)

        self.trace_id = None

        self.release = self.get_release_value(release)

    def get_release_value(self, release: Optional[str] = None) -> Optional[str]:
        if release:
            return release
        elif "LANGFUSE_RELEASE" in os.environ:
            return os.environ["LANGFUSE_RELEASE"]
        else:
            return get_common_release_envs()

    def get_trace_id(self):
        return self.trace_id

    def get_trace_url(self):
        return f"{self.base_url}/trace/{self.trace_id}"

    def get_dataset(self, name: str):
        try:
            self.log.debug(f"Getting datasets {name}")
            dataset = self.client.datasets.get(dataset_name=name)

            items = [DatasetItemClient(i, langfuse=self) for i in dataset.items]

            return DatasetClient(dataset, items=items)
        except Exception as e:
            self.log.exception(e)
            raise e

    def get_dataset_item(self, id: str):
        try:
            self.log.debug(f"Getting dataset item {id}")
            dataset_item = self.client.dataset_items.get(id=id)
            return DatasetItemClient(dataset_item, langfuse=self)
        except Exception as e:
            self.log.exception(e)
            raise e

    def auth_check(self) -> bool:
        try:
            projects = self.client.projects.get()
            self.log.debug(f"Auth check successful, found {len(projects.data)} projects")
            if len(projects.data) == 0:
                raise Exception("Auth check failed, no project found for the keys provided.")
            return True

        except Exception as e:
            self.log.exception(e)
            raise e

    def get_dataset_run(
        self,
        dataset_name: str,
        dataset_run_name: str,
    ) -> DatasetRun:
        try:
            self.log.debug(f"Getting dataset runs for dataset {dataset_name} and run {dataset_run_name}")
            return self.client.datasets.get_runs(dataset_name=dataset_name, run_name=dataset_run_name)
        except Exception as e:
            self.log.exception(e)
            raise e

    def create_dataset(self, body: CreateDatasetRequest) -> Dataset:
        try:
            self.log.debug(f"Creating datasets {body}")
            return self.client.datasets.create(request=body)
        except Exception as e:
            self.log.exception(e)
            raise e

    def create_dataset_item(self, body: CreateDatasetItemRequest) -> DatasetItem:
        """
        Creates a dataset item. Upserts if an item with id already exists.
        """
        try:
            self.log.debug(f"Creating dataset item {body}")
            return self.client.dataset_items.create(request=body)
        except Exception as e:
            self.log.exception(e)
            raise e

    def get_trace(
        self,
        id: str,
    ) -> TraceWithFullDetails:
        try:
            self.log.debug(f"Getting trace {id}")
            return self.client.trace.get(id)
        except Exception as e:
            self.log.exception(e)
            raise e

    def get_generations(
        self,
        *,
        page: typing.Optional[int] = None,
        limit: typing.Optional[int] = None,
        name: typing.Optional[str] = None,
        user_id: typing.Optional[str] = None,
    ):
        try:
            self.log.debug(f"Getting generations... {page}, {limit}, {name}, {user_id}")
            return self.client.observations.get_many(page=page, limit=limit, name=name, user_id=user_id, type="GENERATION")
        except Exception as e:
            self.log.exception(e)
            raise e

    def get_observation(
        self,
        id: str,
    ) -> Observation:
        try:
            self.log.debug(f"Getting observation {id}")
            return self.client.observations.get(id)
        except Exception as e:
            self.log.exception(e)
            raise e

    def trace(self, body: CreateTrace):
        try:
            new_id = str(uuid.uuid4()) if body.id is None else body.id
            self.trace_id = new_id

            new_body = body.copy(update={"id": new_id})

            if self.release is not None:
                new_body = new_body.copy(update={"release": self.release})

            self.log.debug(f"Creating trace {new_body}")
            event = {
                "id": str(uuid.uuid4()),
                "type": "trace-create",
                "body": new_body.dict(),
            }

            self.task_manager.add_task(
                event,
            )

            return StatefulTraceClient(self.client, new_id, StateType.TRACE, new_id, self.task_manager)
        except Exception as e:
            self.log.exception(e)

    def score(self, body: InitialScore):
        try:
            new_id = str(uuid.uuid4()) if body.id is None else body.id

            new_body = body.copy(update={"id": new_id})
            self.log.debug(f"Creating score {new_body}...")
            event = {
                "id": str(uuid.uuid4()),
                "type": "score-create",
                "body": new_body.dict(),
            }

            self.task_manager.add_task(event)

            if body.observation_id is not None:
                return StatefulClient(self.client, body.observation_id, StateType.OBSERVATION, body.trace_id, self.task_manager)
            else:
                return StatefulClient(self.client, new_id, StateType.TRACE, new_id, self.task_manager)

        except Exception as e:
            self.log.exception(e)

    def span(self, body: InitialSpan):
        try:
            new_span_id = str(uuid.uuid4()) if body.id is None else body.id
            new_trace_id = str(uuid.uuid4()) if body.trace_id is None else body.trace_id
            self.trace_id = new_trace_id

            if body.trace_id is None:
                new_body = {
                    "id": new_trace_id,
                    "release": self.release,
                    "name": body.name,
                }

                request = CreateTraceRequest(**new_body)

                event = {
                    "id": str(uuid.uuid4()),
                    "type": "trace-create",
                    "body": request.dict(),
                }

                self.log.debug(f"Creating trace {event}...")
                self.task_manager.add_task(event)

            new_body = body.copy(update={"id": new_span_id, "trace_id": new_trace_id})
            if self.release is not None:
                new_body = new_body.copy(update={"trace": {"release": self.release}})

            if new_body.start_time is None:
                new_body = new_body.copy(update={"startTime": datetime.now()})

            request = CreateSpanRequest(**new_body.dict())

            event = convert_observation_to_event(request, "SPAN")

            self.log.debug(f"Creating span {event}...")
            self.task_manager.add_task(event)

            return StatefulSpanClient(self.client, new_span_id, StateType.OBSERVATION, new_trace_id, self.task_manager)
        except Exception as e:
            self.log.exception(e)

    def generation(self, body: InitialGeneration):
        try:
            new_trace_id = str(uuid.uuid4()) if body.trace_id is None else body.trace_id
            new_generation_id = str(uuid.uuid4()) if body.id is None else body.id
            self.trace_id = new_trace_id

            if body.trace_id is None:
                trace = {
                    "id": new_trace_id,
                    "release": self.release,
                    "name": body.name,
                }
                request = CreateTraceRequest(**trace)

                event = {
                    "id": str(uuid.uuid4()),
                    "type": "trace-create",
                    "body": request.dict(),
                }

                self.log.debug(f"Creating trace {event}...")

                self.task_manager.add_task(event)

            new_body = body.copy(update={"id": new_generation_id, "trace_id": new_trace_id})
            if self.release is not None:
                new_body = new_body.copy(update={"trace": {"release": self.release}})

            if new_body.start_time is None:
                new_body = new_body.copy(update={"startTime": datetime.now()})

            request = CreateGenerationRequest(**new_body.dict())

            event = convert_observation_to_event(request, "GENERATION")

            self.log.debug(f"Creating top-level generation {event} ...")
            self.task_manager.add_task(event)

            return StatefulGenerationClient(self.client, new_generation_id, StateType.OBSERVATION, new_trace_id, self.task_manager)
        except Exception as e:
            self.log.exception(e)

    # On program exit, allow the consumer thread to exit cleanly.
    # This prevents exceptions and a messy shutdown when the
    # interpreter is destroyed before the daemon thread finishes
    # execution. However, it is *not* the same as flushing the queue!
    # To guarantee all messages have been delivered, you'll still need to call flush().
    def join(self):
        try:
            return self.task_manager.join()
        except Exception as e:
            self.log.exception(e)

    def flush(self):
        try:
            return self.task_manager.flush()
        except Exception as e:
            self.log.exception(e)

    def shutdown(self):
        try:
            return self.task_manager.shutdown()
        except Exception as e:
            self.log.exception(e)


class StateType(Enum):
    OBSERVATION = 1
    TRACE = 0


class StatefulClient(object):
    log = logging.getLogger("langfuse")

    def __init__(self, client: FernLangfuse, id: str, state_type: StateType, trace_id: str, task_manager: TaskManager):
        self.client = client
        self.trace_id = trace_id
        self.id = id
        self.state_type = state_type
        self.task_manager = task_manager

    def _add_state_to_event(self, body: dict):
        if self.state_type == StateType.OBSERVATION:
            body["parent_observation_id"] = self.id
            body["trace_id"] = self.trace_id
        else:
            body["trace_id"] = self.id
        return body

    def _add_default_values(self, body: dict):
        if body.get("startTime") is None:
            body["start_time"] = datetime.now()
        return body

    def generation(self, body: CreateGeneration):
        try:
            generation_id = str(uuid.uuid4()) if body.id is None else body.id

            new_body = body.copy(update={"id": generation_id})
            new_body = self._add_state_to_event(new_body.dict())
            new_body = self._add_default_values(new_body)

            new_body = CreateGenerationRequest(**new_body)

            self.log.debug(f"Creating generation {new_body}...")
            self.task_manager.add_task(
                convert_observation_to_event(new_body, "GENERATION"),
            )
            return StatefulGenerationClient(self.client, generation_id, StateType.OBSERVATION, self.trace_id, task_manager=self.task_manager)
        except Exception as e:
            self.log.exception(e)

    def span(self, body: CreateSpan):
        try:
            span_id = str(uuid.uuid4()) if body.id is None else body.id

            new_body = body.copy(update={"id": span_id})
            self.log.debug(f"Creating span {new_body}...")

            new_dict = self._add_state_to_event(new_body.dict())
            new_body = self._add_default_values(new_dict)

            request = CreateSpanRequest(**new_dict)
            event = convert_observation_to_event(request, "SPAN")

            self.task_manager.add_task(event)
            return StatefulSpanClient(self.client, span_id, StateType.OBSERVATION, self.trace_id, task_manager=self.task_manager)
        except Exception as e:
            self.log.exception(e)

    def score(self, body: CreateScore):
        try:
            score_id = str(uuid.uuid4()) if body.id is None else body.id

            new_body = body.copy(update={"id": score_id})
            self.log.debug(f"Creating score {new_body}...")

            new_dict = self._add_state_to_event(new_body.dict())

            if self.state_type == StateType.OBSERVATION:
                new_dict["observationId"] = self.id

            request = CreateScoreRequest(**new_dict)

            event = {
                "id": str(uuid.uuid4()),
                "type": "score-create",
                "body": request.dict(),
            }

            self.task_manager.add_task(event)
            return StatefulClient(self.client, self.id, StateType.OBSERVATION, self.trace_id, task_manager=self.task_manager)
        except Exception as e:
            self.log.exception(e)

    def event(self, body: CreateEvent):
        try:
            event_id = str(uuid.uuid4()) if body.id is None else body.id

            new_body = body.copy(update={"id": event_id})

            new_dict = self._add_state_to_event(new_body.dict())
            new_body = self._add_default_values(new_dict)

            request = CreateEventRequest(**new_dict)

            event = convert_observation_to_event(request, "EVENT")
            self.log.debug(f"Creating event {event}...")
            self.task_manager.add_task(event)
            return StatefulClient(self.client, event_id, self.state_type, self.trace_id, self.task_manager)
        except Exception as e:
            self.log.exception(e)

    def get_trace_url(self):
        return f"{self.client._client_wrapper._base_url}/trace/{self.trace_id}"


class StatefulGenerationClient(StatefulClient):
    log = logging.getLogger("langfuse")

    def __init__(self, client: FernLangfuse, id: str, state_type: StateType, trace_id: str, task_manager: TaskManager):
        super().__init__(client, id, state_type, trace_id, task_manager)

    def update(self, body: UpdateGeneration):
        try:
            new_body = body.copy(update={"generation_id": self.id, "trace_id": self.trace_id})

            request = UpdateGenerationRequest(**new_body.dict())

            event = convert_observation_to_event(request, "GENERATION", True)
            self.log.debug(f"Update generation {event}...")
            self.task_manager.add_task(event)
            return StatefulGenerationClient(self.client, self.id, StateType.OBSERVATION, self.trace_id, task_manager=self.task_manager)
        except Exception as e:
            self.log.exception(e)

    def end(self, body: Optional[UpdateGeneration] = None):
        try:
            if body is None:
                end_time = datetime.now()
                return self.update(UpdateGeneration(endTime=end_time))

            if body.end_time is None:
                end_time = datetime.now()
                body = body.copy(update={"endTime": end_time})

            return self.update(body)

        except Exception as e:
            self.log.warning(e)


class StatefulSpanClient(StatefulClient):
    log = logging.getLogger("langfuse")

    def __init__(self, client: FernLangfuse, id: str, state_type: StateType, trace_id: str, task_manager: TaskManager):
        super().__init__(client, id, state_type, trace_id, task_manager)

    def update(self, body: UpdateSpan):
        try:
            new_body = body.copy(update={"span_id": self.id, "trace_id": self.trace_id})
            self.log.debug(f"Update span {new_body}...")
            request = UpdateSpanRequest(**new_body.dict())

            event = convert_observation_to_event(request, "SPAN", True)

            self.task_manager.add_task(event)
            return StatefulSpanClient(self.client, self.id, StateType.OBSERVATION, self.trace_id, task_manager=self.task_manager)
        except Exception as e:
            self.log.exception(e)

    def end(self, body: Optional[UpdateSpan] = None):
        try:
            if body is None:
                end_time = datetime.now()
                return self.update(UpdateSpan(endTime=end_time))

            if body.end_time is None:
                end_time = datetime.now()
                body = body.copy(update={"endTime": end_time})

            return self.update(body)

        except Exception as e:
            self.log.warning(e)

    def get_langchain_handler(self):
        from langfuse.callback import CallbackHandler

        return CallbackHandler(statefulClient=self)


class StatefulTraceClient(StatefulClient):
    log = logging.getLogger("langfuse")

    def __init__(self, client: FernLangfuse, id: str, state_type: StateType, trace_id: str, task_manager: TaskManager):
        super().__init__(client, id, state_type, trace_id, task_manager)
        self.task_manager = task_manager

    def get_langchain_handler(self):
        try:
            # adding this to ensure our users installed langchain
            import langchain  # noqa
            from langfuse.callback import CallbackHandler

            self.log.debug(f"Creating new handler for trace {self.id}")

            return CallbackHandler(statefulClient=self, debug=self.log.level == logging.DEBUG)
        except ImportError as e:
            self.log.exception(f"Could not import langchain. Some functionality may be missing. {e.message}")

        except Exception as e:
            self.log.exception(e)

    def getNewHandler(self):
        return self.get_langchain_handler()


class DatasetItemClient:
    id: str
    status: DatasetStatus
    input: typing.Any
    expected_output: typing.Optional[typing.Any]
    source_observation_id: typing.Optional[str]
    dataset_id: str
    created_at: dt.datetime
    updated_at: dt.datetime

    langfuse: Langfuse

    def __init__(self, dataset_item: DatasetItem, langfuse: Langfuse):
        self.id = dataset_item.id
        self.status = dataset_item.status
        self.input = dataset_item.input
        self.expected_output = dataset_item.expected_output
        self.source_observation_id = dataset_item.source_observation_id
        self.dataset_id = dataset_item.dataset_id
        self.created_at = dataset_item.created_at
        self.updated_at = dataset_item.updated_at

        self.langfuse = langfuse

    def flush(self, observation: StatefulClient, run_name: str):
        # flush the queue before creating the dataset run item
        # to ensure that all events are persistet.
        observation.task_manager.flush()

    def link(self, observation: typing.Union[StatefulClient, str], run_name: str):
        observation_id = None

        if isinstance(observation, StatefulClient):
            # flush the queue before creating the dataset run item
            # to ensure that all events are persisted.
            observation.task_manager.flush()
            observation_id = observation.id
        elif isinstance(observation, str):
            self.langfuse.flush()
            observation_id = observation
        else:
            raise ValueError("observation parameter must be either a StatefulClient or a string")

        logging.debug(f"Creating dataset run item: {run_name} {self.id} {observation_id}")
        self.langfuse.client.dataset_run_items.create(request=CreateDatasetRunItemRequest(runName=run_name, datasetItemId=self.id, observationId=observation_id))

    def get_langchain_handler(self, *, run_name: str):
        from langfuse.callback import CallbackHandler

        metadata = {"dataset_item_id": self.id, "run_name": run_name, "dataset_id": self.dataset_id}
        trace = self.langfuse.trace(CreateTrace(name="dataset-run", metadata=metadata))
        span = trace.span(CreateSpan(name="dataset-run", metadata=metadata))

        self.langfuse.flush()

        self.link(span, run_name)

        return CallbackHandler(statefulClient=span)


class DatasetClient:
    id: str
    name: str
    status: DatasetStatus
    project_id: str
    dataset_name: str
    created_at: dt.datetime
    updated_at: dt.datetime
    items: typing.List[DatasetItemClient]
    runs: typing.List[str]

    def __init__(self, dataset: Dataset, items: typing.List[DatasetItemClient]):
        self.id = dataset.id
        self.name = dataset.name
        self.status = dataset.status
        self.project_id = dataset.project_id
        self.dataset_name = dataset.name
        self.created_at = dataset.created_at
        self.updated_at = dataset.updated_at
        self.items = items
        self.runs = dataset.runs
