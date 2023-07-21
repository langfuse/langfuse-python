import asyncio
import datetime

import pytest
from langfuse import Langfuse
from langfuse.api.model import (
    CreateEvent,
    CreateGeneration,
    CreateScore,
    CreateSpan,
    CreateTrace,
    UpdateGeneration,
    UpdateSpan,
    Usage,
)
from langfuse.api.resources.span.types.observation_level_span import ObservationLevelSpan


def test_create_trace():
    langfuse = Langfuse("pk-lf-1234567890", "sk-lf-1234567890", "http://localhost:3000")

    trace = langfuse.trace(
        CreateTrace(
            name="this-is-so-great-new",
            user_id="test",
            metadata="test",
        )
    )

    trace = trace.score(CreateScore(name="user-explicit-feedback", value=1, comment="I like how personalized the response is"))

    generation = trace.generation(CreateGeneration(name="his-is-so-great-new", metadata="test"))

    sub_generation = generation.generation(CreateGeneration(name="yet another child", metadata="test"))

    sub_sub_span = sub_generation.span(CreateSpan(name="sub-sub-span", metadata="test"))

    sub_sub_span = sub_sub_span.score(CreateScore(name="user-explicit-feedback", value=1, comment="I like how personalized the response is"))

    result = langfuse.flush()

    assert result["status"] == "success"


@pytest.mark.asyncio
async def test_update_generation():
    client = Langfuse("pk-lf-1234567890", "sk-lf-1234567890", "http://localhost:3000")

    generation = client.generation(CreateGeneration(name="to-be-renames", metadata="test", prompt={"key": "value"}, completion="very long completion"))
    updated_generation = generation.update(UpdateGeneration(name="new-name", metadata="something-else"))
    span = updated_generation.span(CreateSpan(name="sub-span", metadata="test", input={"key": "value"}, output={"key": "value"}))

    span.update(UpdateSpan(level=ObservationLevelSpan.WARNING, metadata="something-else"))

    result = await client.async_flush()

    assert result["status"] == "success"


@pytest.mark.asyncio
async def test_create_generation():
    client = Langfuse("pk-lf-1234567890", "sk-lf-1234567890", "http://localhost:3000")

    generation = client.generation(CreateGeneration(name="top-level-generation", metadata="test"))
    sub_generation = generation.generation(CreateGeneration(name="su-child", metadata="test"))

    sub_generation = sub_generation.event(CreateEvent(name="sub-sub-event", metadata="test"))

    result = await client.async_flush()

    assert result["status"] == "success"


@pytest.mark.asyncio
async def test_notebook():
    langfuse = Langfuse("pk-lf-1234567890", "sk-lf-1234567890", "http://localhost:3000")

    trace = langfuse.trace(
        CreateTrace(
            name="chat-completion",
            userId="user__935d7d1d-8625-4ef4-8651-544613e7bd22",
            metadata={
                "env": "production",
                "email": "user@langfuse.com",
            },
        )
    )

    retrievalStartTime = datetime.datetime.now()

    # retrieveDocs = retrieveDoc()
    # ...

    span = trace.span(
        CreateSpan(
            name="chat-completion",
            startTime=retrievalStartTime,
            endTime=datetime.datetime.now(),
            metadata={"key": "value"},
            input={"key": "value"},
            output={"key": "value"},
        )
    )

    span.event(
        CreateEvent(
            name="chat-docs-retrieval",
            startTime=datetime.datetime.now(),
            metadata={"key": "value"},
            input={"key": "value"},
            output={"key": "value"},
        )
    )

    result = await langfuse.async_flush()

    assert result["status"] == "success"


@pytest.mark.asyncio
async def test_full_nested_example():
    langfuse = Langfuse("pk-lf-1234567890", "sk-lf-1234567890", "http://localhost:3000")

    trace = langfuse.trace(
        CreateTrace(
            name="docs-retrieval",
            userId="user__935d7d1d-8625-4ef4-8651-544613e7bd22",
            metadata={
                "env": "production",
                "email": "user@langfuse.com",
            },
        )
    )

    start = datetime.datetime.now()

    await asyncio.sleep(1)

    trace.generation(
        CreateGeneration(
            name="query-generation",
            startTime=start,
            endTime=datetime.datetime.now(),
            model="gpt-3.5-turbo",
            modelParameters={"maxTokens": "1000", "temperature": "0.9"},
            prompt=[
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": "Please generate the start of a company documentation that contains the answer to the questinon: Write a summary of the Q3 OKR goals",
                },
            ],
            completion="This document entails the OKR goals for ACME",
            usage=Usage(promptTokens=50, completionTokens=49),
            metadata={"interface": "whatsapp"},
        )
    )

    retrievalStartTime = datetime.datetime.now()

    await asyncio.sleep(1)

    dbSearchStart = datetime.datetime.now()
    await asyncio.sleep(1)
    # retrieveDocs = retrieveDoc()
    # ...
    dbSearchEnd = datetime.datetime.now()

    span = trace.span(
        CreateSpan(
            name="embedding-search",
            startTime=retrievalStartTime,
            endTime=datetime.datetime.now(),
            metadata={"database": "pinecone"},
            input={"query": "This document entails the OKR goals for ACME"},
            output={"response": "[{'name': 'OKR Engineering', 'content': 'The engineering department defined the following OKR goals...'},{'name': 'OKR Marketing', 'content': 'The marketing department defined the following OKR goals...'}]"},
        )
    )

    span = span.span(
        CreateSpan(
            name="chat-completion",
            startTime=dbSearchStart,
            endTime=dbSearchEnd,
            metadata={"database": "postgres"},
            input={"email": "user@langfuse.com"},
            output={"firstName": "User", "lastName": "Langfuse", "email": "user@langfuse.com"},
        )
    )

    finalStart = datetime.datetime.now()

    await asyncio.sleep(1)

    trace.generation(
        CreateGeneration(
            name="summary-generation",
            startTime=finalStart,
            endTime=datetime.datetime.now(),
            model="gpt-3.5-turbo",
            modelParameters={"maxTokens": "1000", "temperature": "0.9"},
            prompt=[
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": "Please generate a summary of the following documents \nThe engineering department defined the following OKR goals...\nThe marketing department defined the following OKR goals...",
                },
            ],
            completion="The Q3 OKRs contain goals for multiple teams...",
            usage=Usage(promptTokens=50, completionTokens=49),
            metadata={"interface": "whatsapp"},
        )
    )

    trace.score(CreateScore(name="user-explicit-feedback", value=1, comment="I like how personalized the response is"))

    result = await langfuse.async_flush()

    assert result["status"] == "success"
