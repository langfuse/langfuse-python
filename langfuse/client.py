from enum import Enum
import logging
import os
from typing import Optional
import uuid
from langfuse.api.client import FintoLangfuse
from langfuse.api.resources.commons.types.create_event_request import CreateEventRequest
from langfuse.api.resources.commons.types.create_generation_request import CreateGenerationRequest
from langfuse.api.resources.commons.types.create_span_request import CreateSpanRequest
from langfuse.api.resources.score.types.create_score_request import CreateScoreRequest
from langfuse.api.resources.trace.types.create_trace_request import CreateTraceRequest
from langfuse.model import (
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
)
from langfuse.api.resources.generations.types.update_generation_request import UpdateGenerationRequest
from langfuse.api.resources.span.types.update_span_request import UpdateSpanRequest
from langfuse.task_manager import TaskManager
from .version import __version__ as version


class Langfuse(object):
    log = logging.getLogger("langfuse")

    def __init__(
        self,
        public_key: str,
        secret_key: str,
        host: Optional[str] = None,
        release: Optional[str] = None,
        debug: bool = False,
    ):
        self.task_manager = TaskManager()

        self.base_url = host if host else "https://cloud.langfuse.com"

        self.client = FintoLangfuse(
            environment=self.base_url,
            username=public_key,
            password=secret_key,
            x_langfuse_sdk_name="python",
            x_langfuse_sdk_version=version,
        )

        self.trace_id = None

        self.release = os.environ.get("LANGFUSE_RELEASE", release)

        if debug:
            # Ensures that debug level messages are logged when debug mode is on.
            # Otherwise, defaults to WARNING level.
            # See https://docs.python.org/3/howto/logging.html#what-happens-if-no-configuration-is-provided
            logging.basicConfig()
            self.log.setLevel(logging.DEBUG)
        else:
            self.log.setLevel(logging.WARNING)

    def get_trace_id(self):
        return self.trace_id

    def trace(self, body: CreateTrace):
        try:
            new_id = str(uuid.uuid4()) if body.id is None else body.id
            self.trace_id = new_id

            def task():
                try:
                    new_body = body.copy(update={"id": new_id})

                    if self.release is not None:
                        new_body = new_body.copy(update={"release": self.release})

                    self.log.debug(f"Creating trace {new_body}")
                    return self.client.trace.create(request=new_body)
                except Exception as e:
                    self.log.exception(e)
                    raise e

            self.task_manager.add_task(new_id, task)

            return StatefulTraceClient(self.client, new_id, StateType.TRACE, new_id, self.task_manager)
        except Exception as e:
            self.log.exception(e)

    def score(self, body: InitialScore):
        try:
            new_id = str(uuid.uuid4()) if body.id is None else body.id

            def task():
                try:
                    new_body = body.copy(update={"id": new_id})
                    self.log.debug(f"Creating score {new_body}...")
                    return self.client.score.create(request=new_body)
                except Exception as e:
                    self.log.exception(e)
                    raise e

            self.task_manager.add_task(new_id, task)

            if body.observation_id is not None:
                return StatefulClient(
                    self.client, body.observation_id, StateType.OBSERVATION, body.trace_id, self.task_manager
                )
            else:
                return StatefulClient(self.client, new_id, StateType.TRACE, new_id, self.task_manager)

        except Exception as e:
            self.log.exception(e)

    def span(self, body: InitialSpan):
        try:
            new_trace_id = str(uuid.uuid4()) if body.trace_id is None else body.trace_id
            self.trace_id = new_trace_id
            new_span_id = str(uuid.uuid4()) if body.id is None else body.id

            def create_trace():
                try:
                    new_body = {
                        "id": new_trace_id,
                        "release": self.release,
                        "name": body.name,
                    }

                    self.log.debug(f"Creating trace {new_body}...")
                    request = CreateTraceRequest(**new_body)
                    return self.client.trace.create(request=request)
                except Exception as e:
                    self.log.exception(e)
                    raise e

            def create_span():
                try:
                    new_body = body.copy(update={"id": new_span_id, "trace_id": new_trace_id})

                    if self.release is not None:
                        new_body = new_body.copy(update={"trace": {"release": self.release}})
                    self.log.debug(f"Creating span {new_body}...")
                    request = CreateSpanRequest(**new_body.dict())
                    return self.client.span.create(request=request)
                except Exception as e:
                    self.log.exception(e)
                    raise e

            self.task_manager.add_task(new_trace_id, create_trace)
            self.task_manager.add_task(new_span_id, create_span)

            return StatefulSpanClient(self.client, new_span_id, StateType.OBSERVATION, new_trace_id, self.task_manager)
        except Exception as e:
            self.log.exception(e)

    def generation(self, body: InitialGeneration):
        try:
            new_trace_id = str(uuid.uuid4()) if body.trace_id is None else body.trace_id
            new_generation_id = str(uuid.uuid4()) if body.id is None else body.id
            self.trace_id = new_trace_id

            def create_trace():
                try:
                    new_body = {
                        "id": new_trace_id,
                        "release": self.release,
                        "name": body.name,
                    }

                    self.log.debug(f"Creating trace {new_body}...")
                    request = CreateTraceRequest(**new_body)
                    return self.client.trace.create(request=request)
                except Exception as e:
                    self.log.exception(e)
                    raise e

            def create_generation():
                try:
                    new_body = body.copy(update={"id": new_generation_id, "trace_id": new_trace_id})

                    if self.release is not None:
                        new_body = new_body.copy(update={"trace": {"release": self.release}})
                    self.log.debug(f"Creating top-level generation {new_body}...")
                    request = CreateGenerationRequest(**new_body.dict())
                    return self.client.generations.log(request=request)
                except Exception as e:
                    self.log.exception(e)
                    raise e

            self.task_manager.add_task(new_generation_id, create_generation)
            self.task_manager.add_task(new_trace_id, create_trace)

            return StatefulGenerationClient(
                self.client, new_generation_id, StateType.OBSERVATION, new_trace_id, self.task_manager
            )
        except Exception as e:
            self.log.exception(e)

    # On program exit, allow the consumer thread to exit cleanly.
    # This prevents exceptions and a messy shutdown when the
    # interpreter is destroyed before the daemon thread finishes
    # execution. However, it is *not* the same as flushing the queue!
    # To guarantee all messages have been delivered, you'll still need
    # to call flush().
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

    def __init__(self, client: FintoLangfuse, id: str, state_type: StateType, trace_id: str, task_manager: TaskManager):
        self.client = client
        self.trace_id = trace_id
        self.id = id
        self.state_type = state_type
        self.task_manager = task_manager

    def _add_state_to_observation(self, body: dict):
        if self.state_type == StateType.OBSERVATION:
            body["parent_observation_id"] = self.id
            body["trace_id"] = self.trace_id
        else:
            body["trace_id"] = self.id
        return body

    def generation(self, body: CreateGeneration):
        try:
            generation_id = str(uuid.uuid4()) if body.id is None else body.id

            def task():
                try:
                    new_body = body.copy(update={"id": generation_id})

                    new_dict = self._add_state_to_observation(new_body.dict())

                    self.log.debug(f"Creating generation {new_dict}...")

                    request = CreateGenerationRequest(**new_dict)
                    return self.client.generations.log(request=request)
                except Exception as e:
                    self.log.exception(e)
                    raise e

            self.task_manager.add_task(generation_id, task)
            return StatefulGenerationClient(
                self.client, generation_id, StateType.OBSERVATION, self.trace_id, task_manager=self.task_manager
            )
        except Exception as e:
            self.log.exception(e)

    def span(self, body: CreateSpan):
        try:
            span_id = str(uuid.uuid4()) if body.id is None else body.id

            def task():
                try:
                    new_body = body.copy(update={"id": span_id})
                    self.log.debug(f"Creating span {new_body}...")

                    new_dict = self._add_state_to_observation(new_body.dict())

                    request = CreateSpanRequest(**new_dict)
                    return self.client.span.create(request=request)
                except Exception as e:
                    self.log.exception(e)
                    raise e

            self.task_manager.add_task(span_id, task)
            return StatefulSpanClient(
                self.client, span_id, StateType.OBSERVATION, self.trace_id, task_manager=self.task_manager
            )
        except Exception as e:
            self.log.exception(e)

    def score(self, body: CreateScore):
        try:
            score_id = str(uuid.uuid4()) if body.id is None else body.id

            def task():
                try:
                    new_body = body.copy(update={"id": score_id})
                    self.log.debug(f"Creating score {new_body}...")

                    new_dict = self._add_state_to_observation(new_body.dict())

                    if self.state_type == StateType.OBSERVATION:
                        new_dict["observationId"] = self.id

                    request = CreateScoreRequest(**new_dict)
                    return self.client.score.create(request=request)
                except Exception as e:
                    self.log.exception(e)
                    raise e

            self.task_manager.add_task(score_id, task)
            return StatefulClient(
                self.client, self.id, StateType.OBSERVATION, self.trace_id, task_manager=self.task_manager
            )
        except Exception as e:
            self.log.exception(e)

    def event(self, body: CreateEvent):
        try:
            event_id = str(uuid.uuid4()) if body.id is None else body.id

            def task():
                try:
                    new_body = body.copy(update={"id": event_id})
                    self.log.debug(f"Creating event {new_body}...")

                    new_dict = self._add_state_to_observation(new_body.dict())

                    request = CreateEventRequest(**new_dict)
                    return self.client.event.create(request=request)
                except Exception as e:
                    self.log.exception(e)
                    raise e

            self.task_manager.add_task(body.id, task)
            return StatefulClient(self.client, event_id, self.state_type, self.trace_id, self.task_manager)
        except Exception as e:
            self.log.exception(e)


class StatefulGenerationClient(StatefulClient):
    log = logging.getLogger("langfuse")

    def __init__(self, client: FintoLangfuse, id: str, state_type: StateType, trace_id: str, task_manager: TaskManager):
        super().__init__(client, id, state_type, trace_id, task_manager)

    def update(self, body: UpdateGeneration):
        try:
            update_id = str(uuid.uuid4())

            def task():
                try:
                    new_body = body.copy(update={"generation_id": self.id})
                    self.log.debug(f"Update generation {new_body}...")
                    request = UpdateGenerationRequest(**new_body.dict())
                    return self.client.generations.update(request=request)
                except Exception as e:
                    self.log.exception(e)
                    raise e

            self.task_manager.add_task(update_id, task)
            return StatefulGenerationClient(
                self.client, self.id, StateType.OBSERVATION, self.trace_id, task_manager=self.task_manager
            )
        except Exception as e:
            self.log.exception(e)


class StatefulSpanClient(StatefulClient):
    log = logging.getLogger("langfuse")

    def __init__(self, client: FintoLangfuse, id: str, state_type: StateType, trace_id: str, task_manager: TaskManager):
        super().__init__(client, id, state_type, trace_id, task_manager)

    def update(self, body: UpdateSpan):
        try:
            update_id = str(uuid.uuid4())

            def task():
                try:
                    new_body = body.copy(update={"span_id": self.id})
                    self.log.debug(f"Update span {new_body}...")
                    request = UpdateSpanRequest(**new_body.dict())
                    return self.client.span.update(request=request)
                except Exception as e:
                    self.log.exception(e)
                    raise e

            self.task_manager.add_task(update_id, task)
            return StatefulSpanClient(
                self.client, self.id, StateType.OBSERVATION, self.trace_id, task_manager=self.task_manager
            )
        except Exception as e:
            self.log.exception(e)


class StatefulTraceClient(StatefulClient):
    log = logging.getLogger("langfuse")

    def __init__(self, client: FintoLangfuse, id: str, state_type: StateType, trace_id: str, task_manager: TaskManager):
        super().__init__(client, id, state_type, trace_id, task_manager)
        self.task_manager = task_manager

    def getNewHandler(self):
        from langfuse.callback import CallbackHandler

        return CallbackHandler(statefulTraceClient=self)
