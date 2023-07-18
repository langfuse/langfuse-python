from typing import Optional
import uuid
from langfuse.api.resources.generations.types.create_log import CreateLog
from langfuse.concurrency import FuturesStore
from langfuse.api.client import AsyncFintoLangfuse
from langfuse.api.resources.trace.types.create_trace_request import CreateTraceRequest


class Langfuse:
    
    def __init__(self, public_key: str, secret_key: str, base_url: Optional[str]):
        
        self.future_store = FuturesStore()

        self.base_url = base_url if base_url else 'https://cloud.langfuse.com'

        # auth = base64.b64encode(f"{public_key}:{secret_key}".encode()).decode()
        # headers = {
        #     'Authorization': 'Basic ' + auth,
        #     'X-Langfuse-Sdk-Name': 'langfuse-python',
        #     'X-Langfuse-Sdk-Version': 'version',
        #     'X-Langfuse-Sdk-Variant': 'Server',
        # }
        # self.client = Client(
        #     base_url=self.base_url,
        #     headers=headers,
        #     verify_ssl=True,
        #     raise_on_unexpected_status=False,
        #     follow_redirects=False,
        # )

        self.client = AsyncFintoLangfuse(
            environment=self.base_url,
            username = public_key,
            password = secret_key,
        )
        
    
    def trace(self, body: CreateTraceRequest):

        trace_promise = lambda: self.client.trace.create(request=body)
        trace_future_id = self.future_store.append(trace_promise)

        return TraceClient(self.future_store, self.client, trace_future_id)
        

    def flush(self):
        # Flush the future store instead of executing promises directly
        self.future_store.flush()
  

class TraceClient:

    def __init__(self, future_store: FuturesStore, client: AsyncFintoLangfuse, trace_future_id: int) -> None:
        self.client = client
        self.trace_future_id = trace_future_id
        self.future_store = future_store

    def generation(self, body: CreateLog):

        id = uuid.uuid4() if body.id is None else body.id

        async def task(future_result):
            new_body = body.copy(update={'id': id})

            trace = future_result  # use the future_result directly
            #trace = await self.future_store.futures[self.trace_future_id].result()  # get the result from the trace
            new_body = new_body.copy(update={'trace_id': body.trace_id if body.trace_id is not None else trace.id})

            return await self.client.generations.log(request=new_body)

        # Add the task to the future store with trace_future_id as a dependency
        self.future_store.append(task, future_id=self.trace_future_id)

        return id

