from langfuse import Langfuse
from langfuse.api.resources.generations.types.create_log import CreateLog
from langfuse.api.resources.score.types.create_score_request import CreateScoreRequest
from langfuse.api.resources.span.types.create_span_request import CreateSpanRequest
from langfuse.api.resources.trace.types.create_trace_request import CreateTraceRequest

def test_create_trace():
    print("hello world")
    
    client = Langfuse("pk-lf-1234567890","sk-lf-1234567890", 'http://localhost:3000')

    trace = client.trace(CreateTraceRequest(
        name="this-is-so-great-new",
        user_id="test",
        metadata="test",
    ))
    
    print('add score', trace.id)
    
    trace = trace.score(CreateScoreRequest(
        traceId=trace.id,                  # trace the score is related to
        name="user-explicit-feedback",
        value=1,
        comment="I like how personalized the response is"
    ))

    generation = trace.generation(CreateLog(name="his-is-so-great-new", metadata="test"))

    sub_generation = generation.generation(CreateLog(name="yet another child", metadata="test"))

    sub_sub_span = sub_generation.span(CreateSpanRequest(name="sub-sub-span", metadata="test"))

    sub_sub_span = sub_sub_span.score(CreateScoreRequest(
        traceId=trace.id,                  # trace the score is related to
        name="user-explicit-feedback",
        value=1,
        comment="I like how personalized the response is"
    ))

    client.flush()


def test_create_generation():
    client = Langfuse("pk-lf-1234567890","sk-lf-1234567890", 'http://localhost:3000')

    generation = client.generation(CreateLog(name="top-level-generation", metadata="test"))
    sub_generation = generation.generation(CreateLog(name="su-child", metadata="test"))

    client.flush()




