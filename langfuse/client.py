import asyncio
from enum import Enum
import logging
import traceback
from typing import Awaitable, Optional
import uuid
from langfuse.api.model import (
    CreateEvent,
    CreateGeneration,
    CreateScore,
    CreateSpan,
    CreateTrace,
    Generation,
    Span,
    UpdateGeneration,
    UpdateSpan,
)
from langfuse.api.resources.event.types.create_event_request import CreateEventRequest
from langfuse.api.resources.generations.types.create_log import CreateLog
from langfuse.api.resources.generations.types.update_generation_request import UpdateGenerationRequest
from langfuse.api.resources.score.types.create_score_request import CreateScoreRequest
from langfuse.api.resources.span.types.create_span_request import CreateSpanRequest
from langfuse.api.resources.span.types.update_span_request import UpdateSpanRequest
from langfuse.api.client import FintoLangfuse
from langfuse.task_manager import Task, TaskManager
from .version import __version__ as version


class Langfuse:
    def __init__(self, public_key: str, secret_key: str, host: Optional[str] = None):
        self.task_manager = TaskManager()

        self.base_url = host if host else "https://cloud.langfuse.com"

        self.client = FintoLangfuse(
            environment=self.base_url,
            username=public_key,
            password=secret_key,
            x_langfuse_sdk_name="python",
            x_langfuse_sdk_version=version,
        )

        self.future_id = None

    def trace(self, body: CreateTrace):
        try:
            new_id = str(uuid.uuid4())

            def task(*args):
                try:
                    logging.info(f"Creating span {body}...")
                    return self.client.trace.create(request=body)
                except Exception as e:
                    traceback.print_exception(e)
                    raise e

            self.task_manager.add_task(Task(new_id, task, self.future_id))
            self.future_id = new_id

            return StatefulClient(self.client, None, StateType.TRACE, new_id, task_manager=self.task_manager)
        except Exception as e:
            traceback.print_exception(e)

    def span(self, body: Span):
        try:
            new_id = str(uuid.uuid4())

            def task(*args):
                try:
                    new_body = body.copy(update={"id": new_id})
                    logging.info(f"Creating span {new_body}...")
                    request = CreateSpanRequest(**new_body.dict())
                    return self.client.span.create(request=request)
                except Exception as e:
                    traceback.print_exception(e)
                    raise e

            self.task_manager.add_task(Task(new_id, task, self.future_id))
            self.future_id = new_id

            return StatefulSpanClient(self.client, new_id, StateType.TRACE, new_id, task_manager=self.task_manager)
        except Exception as e:
            traceback.print_exception(e)

    def generation(self, body: Generation):
        try:
            new_id = str(uuid.uuid4()) if body.id is None else body.id
            new_body = body.copy(update={"id": new_id})

            def task(*args):
                try:
                    logging.info(f"Creating generation {new_body}...")
                    request = CreateLog(**new_body.dict())
                    return self.client.generations.log(request=request)
                except Exception as e:
                    traceback.print_exception(e)
                    raise e

            self.task_manager.add_task(Task(new_id, task, self.future_id))
            self.future_id = new_id

            return StatefulGenerationClient(self.client, new_id, StateType.OBSERVATION, new_id, task_manager=self.task_manager)
        except Exception as e:
            traceback.print_exception(e)

    def flush(self):
        try:
            return self.task_manager.wait_for_completion()
        except Exception as e:
            traceback.print_exception(e)


class StateType(Enum):
    OBSERVATION = 1
    TRACE = 0
    SCORE = 2


class StatefulClient:
    def __init__(self, client: Langfuse, id: Optional[str], state_type: StateType, future_id: str, task_manager: TaskManager):
        self.client = client
        self.id = id
        self.future_id = future_id
        self.state_type = state_type
        self.task_manager = task_manager

    def generation(self, body: CreateGeneration):
        try:
            generation_id = str(uuid.uuid4()) if body.id is None else body.id

            def task(future_result):
                try:
                    new_body = body.copy(update={"id": generation_id})
                    parent = future_result

                    if self.state_type == StateType.OBSERVATION:
                        new_body = new_body.copy(update={"parent_observation_id": parent.id})
                        new_body = new_body.copy(update={"trace_id": parent.trace_id})
                    elif self.state_type == StateType.SCORE:
                        new_body = new_body.copy(update={"parent_observation_id": parent.observation_id})
                        new_body = new_body.copy(update={"trace_id": parent.trace_id})
                    else:
                        new_body = new_body.copy(update={"trace_id": parent.id})
                    logging.info(f"Creating generation {new_body}...")

                    request = CreateLog(**new_body.dict())
                    return self.client.generations.log(request=request)
                except Exception as e:
                    traceback.print_exception(e)
                    raise e

            logging.info(f"HUHU task {generation_id} with dependency {self.future_id}...")
            self.task_manager.add_task(Task(generation_id, task, self.future_id))
        except Exception as e:
            traceback.print_exception(e)

        return StatefulGenerationClient(self.client, generation_id, StateType.OBSERVATION, generation_id, task_manager=self.task_manager)

    def span(self, body: CreateSpan):
        try:
            span_id = str(uuid.uuid4()) if body.id is None else body.id

            def task(future_result):
                try:
                    new_body = body.copy(update={"id": span_id})
                    logging.info(f"Creating span {new_body}...")

                    parent = future_result
                    if self.state_type == StateType.OBSERVATION:
                        new_body = new_body.copy(update={"parent_observation_id": parent.id})
                        new_body = new_body.copy(update={"trace_id": parent.trace_id})
                    elif self.state_type == StateType.SCORE:
                        new_body = new_body.copy(update={"parent_observation_id": parent.observation_id})
                        new_body = new_body.copy(update={"trace_id": parent.trace_id})
                    else:
                        new_body = new_body.copy(update={"trace_id": parent.id})

                    request = CreateSpanRequest(**new_body.dict())
                    return self.client.span.create(request=request)
                except Exception as e:
                    traceback.print_exception(e)
                    raise e

            self.task_manager.add_task(Task(span_id, task, self.future_id))

            return StatefulSpanClient(self.client, span_id, StateType.OBSERVATION, span_id, task_manager=self.task_manager)
        except Exception as e:
            traceback.print_exception(e)

    def score(self, body: CreateScore):
        try:
            score_id = str(uuid.uuid4()) if body.id is None else body.id

            def task(future_result):
                try:
                    new_body = body.copy(update={"id": score_id})
                    logging.info(f"Creating score {new_body}...")
                    parent = future_result

                    new_body = body
                    if self.state_type == StateType.OBSERVATION:
                        new_body = new_body.copy(update={"observation_id": parent.id})
                        new_body = new_body.copy(update={"trace_id": parent.trace_id})
                    elif self.state_type == StateType.SCORE:
                        new_body = new_body.copy(update={"parent_observation_id": parent.observation_id})
                        new_body = new_body.copy(update={"trace_id": parent.trace_id})
                    else:
                        new_body = new_body.copy(update={"trace_id": parent.id})

                    request = CreateScoreRequest(**new_body.dict())
                    return self.client.score.create(request=request)
                except Exception as e:
                    traceback.print_exception(e)
                    raise e

            self.task_manager.add_task(Task(score_id, task, self.future_id))

            return StatefulClient(self.client, self.id, StateType.SCORE, score_id, task_manager=self.task_manager)
        except Exception as e:
            traceback.print_exception(e)

    def event(self, body: CreateEvent):
        try:
            event_id = str(uuid.uuid4()) if body.id is None else body.id

            def task(future_result):
                try:
                    new_body = body.copy(update={"id": event_id})
                    logging.info(f"Creating event {new_body}...")
                    parent = future_result

                    if self.state_type == StateType.OBSERVATION:
                        new_body = new_body.copy(update={"parent_observation_id": parent.id})
                        new_body = new_body.copy(update={"trace_id": parent.trace_id})
                    elif self.state_type == StateType.SCORE:
                        new_body = new_body.copy(update={"parent_observation_id": parent.observation_id})
                        new_body = new_body.copy(update={"trace_id": parent.trace_id})
                    else:
                        new_body = new_body.copy(update={"trace_id": parent.id})

                    request = CreateEventRequest(**new_body.dict())
                    return self.client.event.create(request=request)
                except Exception as e:
                    traceback.print_exception(e)
                    raise e

            self.task_manager.add_task(Task(body.id, task, self.future_id))

            return StatefulClient(self.client, event_id, self.state_type, event_id, task_manager=self.task_manager)
        except Exception as e:
            traceback.print_exception(e)


class StatefulGenerationClient(StatefulClient):
    def __init__(self, client: Langfuse, id: Optional[str], state_type: StateType, future_id: str, task_manager: TaskManager):
        super().__init__(client, id, state_type, future_id, task_manager)

    def update(self, body: UpdateGeneration):
        try:
            future_id = str(uuid.uuid4())
            generation_id = self.future_id

            def task(future_result):
                try:
                    parent = future_result

                    new_body = body.copy(update={"generation_id": parent.id})
                    logging.info(f"Update generation {new_body}...")
                    request = UpdateGenerationRequest(**new_body.dict())
                    return self.client.generations.update(request=request)
                except Exception as e:
                    traceback.print_exception(e)
                    raise e

            self.task_manager.add_task(Task(future_id, task, self.future_id))

            return StatefulGenerationClient(self.client, generation_id, StateType.OBSERVATION, future_id, task_manager=self.task_manager)
        except Exception as e:
            traceback.print_exception(e)


class StatefulSpanClient(StatefulClient):
    def __init__(self, client: Langfuse, id: Optional[str], state_type: StateType, future_id: str, task_manager: TaskManager):
        super().__init__(client, id, state_type, future_id, task_manager)

    def update(self, body: UpdateSpan):
        try:
            future_id = str(uuid.uuid4())
            span_id = self.future_id

            def task(future_result):
                try:
                    parent = future_result

                    new_body = body.copy(update={"span_id": parent.id})
                    logging.info(f"Update span {new_body}...")
                    request = UpdateSpanRequest(**new_body.dict())
                    return self.client.span.update(request=request)
                except Exception as e:
                    traceback.print_exception(e)
                    raise e

            self.task_manager.add_task(Task(future_id, task, self.future_id))

            return StatefulSpanClient(self.client, span_id, StateType.OBSERVATION, future_id, task_manager=self.task_manager)
        except Exception as e:
            traceback.print_exception(e)


class LangfuseAsync:
    def __init__(self, public_key: str, secret_key: str, host: Optional[str] = None):
        self.langfuse = Langfuse(public_key, secret_key, host)

    async def _run_in_executor(self, func, *args, **kwargs) -> Awaitable[StatefulClient]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, func, *args, **kwargs)

    async def trace(self, body: CreateTrace):
        client = await self._run_in_executor(self.langfuse.trace, body)
        return StatefulClientAsync(client.client, client.id, client.state_type, client.future_id, self.langfuse.task_manager)

    async def span(self, body: Span):
        client = await self._run_in_executor(self.langfuse.span, body)
        return StatefulClientAsync(client.client, client.id, client.state_type, client.future_id, self.langfuse.task_manager)

    async def generation(self, body: Generation):
        client = await self._run_in_executor(self.langfuse.generation, body)
        return StatefulClientAsync(client.client, client.id, client.state_type, client.future_id, self.langfuse.task_manager)

    async def flush(self):
        try:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self.langfuse.task_manager.wait_for_completion)
        except Exception as e:
            traceback.print_exception(e)


class StatefulClientAsync(StatefulClient):
    def __init__(self, client: Langfuse, id: Optional[str], state_type: StateType, future_id: str, task_manager: TaskManager):
        self.stateful_client = StatefulClient(client, id, state_type, future_id, task_manager)

    async def _run_in_executor(self, func, *args, **kwargs) -> Awaitable[StatefulClient]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, func, *args, **kwargs)

    async def generation(self, body: CreateGeneration):
        client = await self._run_in_executor(self.stateful_client.generation, body)
        return StatefulGenerationClientAsync(client.client, client.id, client.state_type, client.future_id, self.stateful_client.task_manager)

    async def span(self, body: CreateSpan):
        client = await self._run_in_executor(self.stateful_client.span, body)
        return StatefulSpanClientAsync(client.client, client.id, client.state_type, client.future_id, self.stateful_client.task_manager)

    async def score(self, body: CreateScore):
        client = await self._run_in_executor(self.stateful_client.score, body)
        return StatefulClientAsync(client.client, client.id, client.state_type, client.future_id, self.stateful_client.task_manager)

    async def event(self, body: CreateEvent):
        client = await self._run_in_executor(self.stateful_client.event, body)
        return StatefulClientAsync(client.client, client.id, client.state_type, client.future_id, self.stateful_client.task_manager)


class StatefulGenerationClientAsync(StatefulClientAsync):
    def __init__(self, client: Langfuse, id: Optional[str], state_type: StateType, future_id: str, task_manager: TaskManager):
        self.stateful_client = StatefulGenerationClient(client, id, state_type, future_id, task_manager)

    async def update(self, body: UpdateGeneration):
        client = await self._run_in_executor(self.stateful_client.update, body)
        return StatefulGenerationClientAsync(client.client, client.id, client.state_type, client.future_id, self.stateful_client.task_manager)


class StatefulSpanClientAsync(StatefulClientAsync):
    def __init__(self, client: Langfuse, id: Optional[str], state_type: StateType, future_id: str, task_manager: TaskManager):
        self.stateful_client = StatefulSpanClient(client, id, state_type, future_id, task_manager)

    async def _run_in_executor(self, func, *args, **kwargs) -> Awaitable[StatefulClient]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, func, *args, **kwargs)

    async def update(self, body: UpdateSpan):
        client = await self._run_in_executor(self.stateful_client.update, body)
        return StatefulSpanClientAsync(client.client, client.id, client.state_type, client.future_id, self.stateful_client.task_manager)
