from enum import Enum
from typing import Optional
import uuid
from langfuse import version
from langfuse.api.resources.event.types.create_event_request import CreateEventRequest
from langfuse.api.resources.generations.types.create_log import CreateLog
from langfuse.api.resources.score.types.create_score_request import CreateScoreRequest
from langfuse.api.resources.span.types.create_span_request import CreateSpanRequest
from langfuse.futures import FuturesStore
from langfuse.api.client import AsyncFintoLangfuse
from langfuse.api.resources.trace.types.create_trace_request import CreateTraceRequest
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
    
    def trace(self, body: CreateTraceRequest):

        new_id = str(uuid.uuid4()) if body.id is None else body.id
        body = body.copy(update={'id': new_id})
        trace_promise = lambda: self.client.trace.create(request=body)
        self.future_store.append(new_id, trace_promise)

        return StatefulClient(self.client, new_id,StateType.TRACE, future_store=self.future_store)

    def generation(self, body: CreateLog):

        new_id = str(uuid.uuid4()) if body.id is None else body.id
        body = body.copy(update={'id': new_id})    
        generation_promise = lambda: self.client.generations.log(request=body)
        self.future_store.append(id, generation_promise)

        return StatefulClient(self.client,id, StateType.OBSERVATION, future_store=self.future_store)
        
    def flush(self):
        # Flush the future store instead of executing promises directly
        self.future_store.flush()


class StateType(Enum):
    OBSERVATION=1
    TRACE=0
        

class StatefulClient:

    def __init__(self, client: Langfuse, id: str, state_type: StateType, future_store: FuturesStore):
        self.client = client
        self.id =id
        self.state_type=state_type
        self.future_store = future_store


    def generation(self, body: CreateLog):
        
        id = str(uuid.uuid4()) if body.id is None else body.id

        async def task(future_result):
            new_body = body.copy(update={'id': id})

            parent = future_result
            
            if self.state_type == StateType.OBSERVATION:
                new_body = new_body.copy(update={'parent_observation_id': body.parent_observation_id if body.parent_observation_id is not None else parent.id})
                new_body = new_body.copy(update={'trace_id': body.trace_id if body.trace_id is not None else parent.trace_id})
            else:   
                new_body = new_body.copy(update={'trace_id': body.trace_id if body.trace_id is not None else parent.id})
            
            return await self.client.generations.log(request=new_body)

        # Add the task to the future store with trace_future_id as a dependency
        self.future_store.append(id, task, future_id=self.id)

        return StatefulClient(self.client, id, StateType.OBSERVATION, future_store=self.future_store)

    def span(self, body: CreateSpanRequest):
        
        id = str(uuid.uuid4()) if body.id is None else body.id

        async def task(future_result):
            new_body = body.copy(update={'id': id})

            parent = future_result
            
            if self.state_type == StateType.OBSERVATION:
                new_body = new_body.copy(update={'parent_observation_id': body.parent_observation_id if body.parent_observation_id is not None else parent.id})
                new_body = new_body.copy(update={'trace_id': body.trace_id if body.trace_id is not None else parent.trace_id})
            else:   
                new_body = new_body.copy(update={'trace_id': body.trace_id if body.trace_id is not None else parent.id})
            
            return await self.client.span.create(request=new_body)

        # Add the task to the future store with trace_future_id as a dependency
        self.future_store.append(id, task, future_id=self.id)

        return StatefulClient(self.client,self.id,StateType.OBSERVATION, future_store=self.future_store)
    
    def score(self, body: CreateScoreRequest):

        async def task(future_result):

            parent = future_result
            
            new_body = body
            if self.state_type == StateType.OBSERVATION:
                new_body = new_body.copy(update={'observation_id': body.observation_id if body.observation_id is not None else parent.id})

            return await self.client.score.create(request=new_body)

        # Add the task to the future store with trace_future_id as a dependency
        self.future_store.append(body.id, task, future_id=self.id)

        return StatefulClient(self.client, self.id, self.state_type, future_store=self.future_store)

    def event(self, body: CreateEventRequest):
            
        id = str(uuid.uuid4()) if body.id is None else body.id

        async def task(future_result):
            new_body = body.copy(update={'id': id})

            parent = future_result
            
            if self.state_type == StateType.OBSERVATION:
                new_body = new_body.copy(update={'parent_observation_id': body.parent_observation_id if body.parent_observation_id is not None else parent.id})
                new_body = new_body.copy(update={'trace_id': body.trace_id if body.trace_id is not None else parent.trace_id})
            else:   
                new_body = new_body.copy(update={'trace_id': body.trace_id if body.trace_id is not None else parent.id})
    
            return await self.client.event.create(request=new_body)
    
        self.future_store.append(body.id, task, future_id=self.id)

        return StatefulClient(self.client, self.id, self.state_type, future_store=self.future_store)
