import asyncio
import base64
from typing import Coroutine, Optional
from wrapper.api.trace import trace_create
from wrapper.client import Client

from wrapper.models.create_trace_request import CreateTraceRequest


class ApiClient:
    """A Client which has been authenticated for use on secured endpoints"""

    
    def __init__(self, public_key: str, secret_key: str, base_url: Optional[str]):
        
        self.promises: list[Coroutine] = []#attr.ib(factory=list)

        self.base_url = base_url if base_url else 'https://cloud.langfuse.com'

        print("__init__", self.promises)
        auth = base64.b64encode(f"{public_key}:{secret_key}".encode()).decode()
        headers = {
            'Authorization': 'Basic ' + auth,
            'X-Langfuse-Sdk-Name': 'langfuse-python',
            'X-Langfuse-Sdk-Version': 'version',
            'X-Langfuse-Sdk-Variant': 'Server',
        }
        self.client = Client(
            base_url=self.base_url,
            headers=headers,
            verify_ssl=True,
            raise_on_unexpected_status=False,
            follow_redirects=False,
        )
        
        print("self.promises")
    
    def trace(self, body: CreateTraceRequest):
        trace_promise = trace_create.asyncio(client=self.client, json_body=body)
        print("trace_promise", self.promises)
        self.promises.append(trace_promise)

    async def flush(self):
        return await asyncio.gather(*self.promises)
        
