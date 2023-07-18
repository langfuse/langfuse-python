import pytest
#from langfuse.openapi.models.create_trace_request import CreateTraceRequest

from langfuse import Langfuse
from langfuse.api.resources.generations.types.create_log import CreateLog
from langfuse.api.resources.trace.types.create_trace_request import CreateTraceRequest

@pytest.mark.timeout(5)
def test_create_trace():
    client = Langfuse("pk-lf-1234567890","sk-lf-1234567890", 'http://localhost:3000')

    trace = client.trace(CreateTraceRequest(
        name="this-is-so-great-new",
        user_id="test",
        metadata="test",
    ))

    generation = trace.generation(CreateLog(name="his-is-so-great-new", metadata="test"))

    generation.generation(CreateLog(name="yet another child", metadata="test"))

    client.flush()




