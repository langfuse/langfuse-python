import asyncio
from langfuse.wrapper import ApiClient


from wrapper.models.create_trace_request import CreateTraceRequest


def test_create_trace():
    req = CreateTraceRequest(
        name="this is great",
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