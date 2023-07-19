import datetime

import pytest
from langfuse import Langfuse
from langfuse.api.model import CreateEvent, CreateGeneration, CreateScore, CreateSpan, CreateTrace

def test_create_trace():
    
    client = Langfuse("pk-lf-1234567890","sk-lf-1234567890", 'http://localhost:3000')

    trace = client.trace(CreateTrace(
        name="this-is-so-great-new",
        user_id="test",
        metadata="test",
    ))
    
    trace = trace.score(CreateScore(
        name="user-explicit-feedback",
        value=1,
        comment="I like how personalized the response is"
    ))

    generation = trace.generation(CreateGeneration(name="his-is-so-great-new", metadata="test"))

    sub_generation = generation.generation(CreateGeneration(name="yet another child", metadata="test"))

    sub_sub_span = sub_generation.span(CreateSpan(name="sub-sub-span", metadata="test"))

    sub_sub_span = sub_sub_span.score(CreateScore(
        name="user-explicit-feedback",
        value=1,
        comment="I like how personalized the response is"
    ))

    result = client.flush()
    print("result", result)

    assert result['status'] == 'success'

@pytest.mark.asyncio
async def test_create_generation():
    client = Langfuse("pk-lf-1234567890","sk-lf-1234567890", 'http://localhost:3000')

    generation = client.generation(CreateGeneration(name="top-level-generation", metadata="test"))
    sub_generation = generation.generation(CreateGeneration(name="su-child", metadata="test"))

    sub_generation = sub_generation.event(CreateEvent(name="sub-sub-event", metadata="test"))


    result = await client.async_flush()    

    print("result", result)

    assert result['status'] == 'success'


@pytest.mark.asyncio
async def test_notebook():
    langfuse = Langfuse("pk-lf-1234567890","sk-lf-1234567890", 'http://localhost:3000')

    trace = langfuse.trace(CreateTrace(
        name = "chat-completion",
        userId = "user__935d7d1d-8625-4ef4-8651-544613e7bd22",
        metadata = {
            "env": "production",
            "email": "user@langfuse.com",
        }
    ))

    retrievalStartTime = datetime.datetime.now()

    # retrieveDocs = retrieveDoc()
    # ...

    span = trace.span(CreateSpan(
            name="chat-completion",
            startTime=retrievalStartTime,
            endTime=datetime.datetime.now(),
            metadata={
                'key': 'value'
            },
            input = {
                'key': 'value'
            },
            output = {
                'key': 'value'
            },
        )
    )

    event = span.event(CreateEvent(
        name="chat-docs-retrieval",
        startTime=datetime.datetime.now(),
        metadata={
            "key": "value"
        },
        input = {
            "key": "value"
        },
        output = {
            "key": "value"
        }
    ))

    result = await langfuse.async_flush()    

    print("result", result)

    assert result['status'] == 'success'

    assert result['status'] == 'success'

