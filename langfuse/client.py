import asyncio
from enum import Enum
from typing import Optional
import uuid
from langfuse import version
from langfuse.api.model import CreateEvent, CreateGeneration, CreateScore, CreateSpan, CreateTrace, UpdateGeneration, UpdateSpan
from langfuse.api.resources.event.types.create_event_request import CreateEventRequest
from langfuse.api.resources.generations.types.create_log import CreateLog
from langfuse.api.resources.generations.types.update_generation_request import UpdateGenerationRequest
from langfuse.api.resources.score.types.create_score_request import CreateScoreRequest
from langfuse.api.resources.span.types.create_span_request import CreateSpanRequest
from langfuse.api.resources.span.types.update_span_request import UpdateSpanRequest
from langfuse.futures import FuturesStore
from langfuse.api.client import AsyncFintoLangfuse
from .version import __version__ as version

class Langfuse:
    
    def __init__(self, public_key: str, secret_key: str, base_url: Optional[str] = None):
        
        self.future_store = FuturesStore()

        self.base_url = base_url if base_url else 'https://cloud.langfuse.com'

        self.client = AsyncFintoLangfuse(
            environment=self.base_url,
            username=public_key,
            password=secret_key,
            x_langfuse_sdk_name='python',
            x_langfuse_sdk_version=version,
        )
    
    def trace(self, body: CreateTrace):

        new_id = str(uuid.uuid4())

        trace_promise = lambda: self.client.trace.create(request=body)
        self.future_store.append(new_id, trace_promise)

        return StatefulClient(self.client, None, StateType.TRACE, new_id, future_store=self.future_store)

    def generation(self, body: CreateLog):

        new_id = str(uuid.uuid4()) if body.id is None else body.id
        body = body.copy(update={'id': new_id})
        print('generation: ', body.dict())  
        request = CreateLog(**body.dict())
        print('request: ', body.dict())  
        generation_promise = lambda: self.client.generations.log(request=request)
        self.future_store.append(new_id, generation_promise)

        return StatefulGenerationClient(self.client, new_id, StateType.OBSERVATION, new_id, future_store=self.future_store)
        
    async def async_flush(self):
        return await self.future_store.flush()
    
    def flush(self):
        return asyncio.run(self.future_store.flush())  # Make sure to call self.async_flush() here


class StateType(Enum):
    OBSERVATION=1
    TRACE=0
        

class StatefulClient:

    def __init__(self, client: Langfuse, id:  Optional[str], state_type: StateType, future_id: str, future_store: FuturesStore):
        self.client = client
        self.id =id
        self.future_id = future_id
        self.state_type=state_type
        self.future_store = future_store


    def generation(self, body: CreateGeneration):
        print('generation: ', body, self.future_id)
        
        generation_id = str(uuid.uuid4()) if body.id is None else body.id

        async def task(future_result):
            new_body = body.copy(update={'id': generation_id})

            parent = future_result
            
            if self.state_type == StateType.OBSERVATION:
                new_body = new_body.copy(update={'parent_observation_id': parent.id})
                new_body = new_body.copy(update={'trace_id': parent.trace_id})
            else:   
                new_body = new_body.copy(update={'trace_id': parent.id})
            
            request = CreateLog(**new_body.dict())
            print('submitting generation: ', request)
            return await self.client.generations.log(request=request)

        # Add the task to the future store with trace_future_id as a dependency
        self.future_store.append(generation_id, task, future_id=self.future_id)

        return StatefulGenerationClient(self.client, generation_id, StateType.OBSERVATION, generation_id, future_store=self.future_store)

    def span(self, body: CreateSpan):
        print("span body", body, type(body), body.input, body.output)
        span_id = str(uuid.uuid4()) if body.id is None else body.id

        async def task(future_result):
            new_body = body.copy(update={'id': span_id})

            parent = future_result
            print("parent", parent)
            if self.state_type == StateType.OBSERVATION:
                new_body = new_body.copy(update={'parent_observation_id': parent.id})
                new_body = new_body.copy(update={'trace_id': parent.trace_id})
            else:   
                new_body = new_body.copy(update={'trace_id': parent.id})
            print("new_body", new_body)
            request = CreateSpanRequest(**new_body.dict())
            print('submitting span: ', request)
            return await self.client.span.create(request=request)

        # Add the task to the future store with trace_future_id as a dependency
        self.future_store.append(span_id, task, future_id=self.future_id)

        return StatefulSpanClient(self.client, span_id, StateType.OBSERVATION, span_id, future_store=self.future_store)
    
    def score(self, body: CreateScore):

        score_id = str(uuid.uuid4()) if body.id is None else body.id

        async def task(future_result):

            new_body = body.copy(update={'id': score_id})

            parent = future_result
            
            new_body = body
            if self.state_type == StateType.OBSERVATION:
                new_body = new_body.copy(update={'observation_id': parent.id})
                new_body = new_body.copy(update={'trace_id': parent.trace_id})
            else:
                new_body = new_body.copy(update={'trace_id': parent.id})
            
            request = CreateScoreRequest(**new_body.dict())
            print('submitting score: ', request)
            return await self.client.score.create(request=request)

        self.future_store.append(score_id, task, future_id=self.future_id)

        return StatefulClient(self.client, self.id, self.state_type, self.future_id, future_store=self.future_store)

    def event(self, body: CreateEvent):
            
        event_id = str(uuid.uuid4()) if body.id is None else body.id

        async def task(future_result):
            new_body = body.copy(update={'id': event_id})

            parent = future_result
            
            if self.state_type == StateType.OBSERVATION:
                new_body = new_body.copy(update={'parent_observation_id': parent.id})
                new_body = new_body.copy(update={'trace_id': parent.trace_id})
            else:   
                print('parent', parent)
                new_body = new_body.copy(update={'trace_id': parent.id})
    
            request = CreateEventRequest(**new_body.dict())
            print('submitting event: ', request)
            return await self.client.event.create(request=request)
    
        self.future_store.append(body.id, task, future_id=self.future_id)

        return StatefulClient(self.client, event_id, self.state_type, event_id, future_store=self.future_store)


class StatefulGenerationClient(StatefulClient):
    
    def __init__(self, client: Langfuse, id:  Optional[str], state_type: StateType, future_id: str, future_store: FuturesStore):
        super().__init__(client, id, state_type, future_id, future_store)

    def update(self, body: UpdateGeneration):
        
        future_id = str(uuid.uuid4())
        generation_id = self.future_id

        async def task(future_result):

            parent = future_result
            
            new_body = body.copy(update={'generation_id': parent.id})

            request = UpdateGenerationRequest(**new_body.dict())
            print('updating generation: ', request)
            return await self.client.generations.update(request=request)

        # Add the task to the future store with trace_future_id as a dependency
        self.future_store.append(future_id, task, future_id=self.future_id)

        return StatefulGenerationClient(self.client, generation_id, StateType.OBSERVATION, future_id, future_store=self.future_store)


class StatefulSpanClient(StatefulClient):
    
    def __init__(self, client: Langfuse, id:  Optional[str], state_type: StateType, future_id: str, future_store: FuturesStore):
        super().__init__(client, id, state_type, future_id, future_store)

    def update(self, body: UpdateSpan):
        print('span: ', body, self.future_id)
        
        future_id = str(uuid.uuid4())
        span_id = self.future_id

        async def task(future_result):

            parent = future_result
            
            new_body = body.copy(update={'span_id': parent.id})

            request = UpdateSpanRequest(**new_body.dict())
            print('updating span: ', request)
            return await self.client.span.update(request=request)

        # Add the task to the future store with trace_future_id as a dependency
        self.future_store.append(future_id, task, future_id=self.future_id)

        return StatefulGenerationClient(self.client, span_id, StateType.OBSERVATION, future_id, future_store=self.future_store)