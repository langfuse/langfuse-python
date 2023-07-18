from enum import Enum
import json
from typing import Optional
import uuid
from langfuse import version
from langfuse.api.resources.generations.types.create_log import CreateLog
from langfuse.api.resources.span.types.create_span_request import CreateSpanRequest
from langfuse.concurrency import FuturesStore
from langfuse.api.client import AsyncFintoLangfuse
from langfuse.api.resources.trace.types.create_trace_request import CreateTraceRequest
from .version import __version__ as version

class Langfuse:
    
    def __init__(self, public_key: str, secret_key: str, base_url: Optional[str]):
        
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

        trace_promise = lambda: self.client.trace.create(request=body)
        future_id = uuid.uuid4()
        self.future_store.append(future_id, trace_promise)

        return StatefulClient(self.client, State(StateType.TRACE, future_id), future_store=self.future_store)

    def generation(self, body: CreateLog):
            
        generation_promise = lambda: self.client.generations.log(request=body)
        future_id = uuid.uuid4()
        self.future_store.append(future_id, generation_promise)

        return StatefulClient(self.client, State(StateType.OBSERVATION, future_id), future_store=self.future_store)
        

    def flush(self):
        # Flush the future store instead of executing promises directly
        self.future_store.flush()


class StateType(Enum):
    OBSERVATION=1
    TRACE=0

class State:
    def __init__(self, type: StateType, id: str):
        self.type = type
        self.id = id

class StatefulClient:

    def __init__(self, client: Langfuse, state: State, future_store: FuturesStore):
        self.client = client
        self.state = state
        self.future_store = future_store


    def generation(self, body: CreateLog):
        print('state', self.state.type, self.state.id)
        print('generation', list(self.future_store.futures.keys()))
        
        id = uuid.uuid4() if body.id is None else body.id
        print(self.future_store)
        async def task(future_result):
            new_body = body.copy(update={'id': id})

            parent = future_result
            
            if self.state.type == StateType.OBSERVATION:
                new_body = new_body.copy(update={'parent_observation_id': body.parent_observation_id if body.parent_observation_id is not None else parent.id})
                new_body = new_body.copy(update={'trace_id': body.trace_id if body.trace_id is not None else parent.trace_id})
            else:   
                new_body = new_body.copy(update={'trace_id': body.trace_id if body.trace_id is not None else parent.id})
            
            print('parent', parent.dict())
            print('new',new_body.dict())
            return await self.client.generations.log(request=new_body)

        # Add the task to the future store with trace_future_id as a dependency
        self.future_store.append(id, task, future_id=self.state.id)

        return StatefulClient(self.client, State(StateType.OBSERVATION, id), future_store=self.future_store)

    def span(self, body: CreateSpanRequest):
        print('state', self.state.type, self.state.id)
        print('generation', list(self.future_store.futures.keys()))
        
        id = uuid.uuid4() if body.id is None else body.id
        print(self.future_store)
        async def task(future_result):
            new_body = body.copy(update={'id': id})

            parent = future_result
            
            if self.state.type == StateType.OBSERVATION:
                new_body = new_body.copy(update={'parent_observation_id': body.parent_observation_id if body.parent_observation_id is not None else parent.id})
                new_body = new_body.copy(update={'trace_id': body.trace_id if body.trace_id is not None else parent.trace_id})
            else:   
                new_body = new_body.copy(update={'trace_id': body.trace_id if body.trace_id is not None else parent.id})
            
            print('parent', parent.dict())
            print('new',new_body.dict())
            return await self.client.span.create(request=new_body)

        # Add the task to the future store with trace_future_id as a dependency
        self.future_store.append(id, task, future_id=self.state.id)

        return StatefulClient(self.client, State(StateType.OBSERVATION, id), future_store=self.future_store)
