import asyncio
from langfuse.models.create_trace_request import CreateTraceRequest
from langfuse.wrapper import ApiClient


def test_create_trace():
    req = CreateTraceRequest(
        name="test",
        user_id="test",
        external_id="test",
        metadata="test",
    )
    print(req)
    client = ApiClient("pk-lf-1234567890","sk-lf-1234567890", 'http://localhost:3000')
    print(client)
    print('before trace call')
    client.trace(req)
    print('post trace call')
    asyncio.run(client.flush())
