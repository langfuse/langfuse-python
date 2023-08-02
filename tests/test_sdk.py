import datetime
import time

import pytest

from langfuse import Langfuse
from langfuse.api.resources.commons.types.observation_level import ObservationLevel
from langfuse.api.resources.commons.types.trace_id_type_enum import TraceIdTypeEnum
from langfuse.model import (
    CreateEvent,
    CreateGeneration,
    CreateScore,
    CreateSpan,
    CreateTrace,
    Generation,
    Span,
    UpdateGeneration,
    UpdateSpan,
    Usage,
)

from langfuse.client import LangfuseAsync

host = "http://localhost:3000/"


@pytest.mark.asyncio
async def test_create_trace_async():
    langfuse = LangfuseAsync("pk-lf-1234567890", "sk-lf-1234567890", host)

    trace = await langfuse.trace(
        CreateTrace(
            name="this-is-so-great-new",
            user_id="test",
            metadata="test",
        )
    )

    trace = await trace.score(CreateScore(name="user-explicit-feedback", value=1, comment="I like how personalized the response is"))

    generation = await trace.generation(CreateGeneration(name="his-is-so-great-new", metadata="test"))

    sub_generation = await generation.generation(CreateGeneration(name="yet another child", metadata="test"))
    # result = asyncio.gather(langfuse.async_flush(), langfuse.async_flush())
    sub_sub_span = await sub_generation.span(CreateSpan(name="sub-sub-span", metadata="test"))

    sub_sub_span = await sub_sub_span.score(CreateScore(name="user-explicit-feedback", value=1, comment="I like how personalized the response is"))

    await langfuse.flush()


def test_create_trace():
    langfuse = Langfuse("pk-lf-1234567890", "sk-lf-1234567890", host)

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
    # result = asyncio.gather(langfuse.async_flush(), langfuse.async_flush())
    sub_sub_span = sub_generation.span(CreateSpan(name="sub-sub-span", metadata="test"))

    sub_sub_span = sub_sub_span.score(CreateScore(name="user-explicit-feedback", value=1, comment="I like how personalized the response is"))

    langfuse.flush()


def test_update_generation():
    langfuse = Langfuse("pk-lf-1234567890", "sk-lf-1234567890", host)

    generation = langfuse.generation(CreateGeneration(name="to-be-renames", metadata="test", prompt={"key": "value"}, completion="very long completion"))
    updated_generation = generation.update(UpdateGeneration(name="new-name", metadata="something-else"))
    span = updated_generation.span(CreateSpan(name="sub-span", metadata="test", input={"key": "value"}, output={"key": "value"}))

    span.update(UpdateSpan(level=ObservationLevel.WARNING, metadata="something-else"))

    langfuse.flush()


def test_create_generation():
    langfuse = Langfuse("pk-lf-1234567890", "sk-lf-1234567890", host)

    langfuse.generation(CreateGeneration(name="max-top-level-generation", traceId="some-id", traceIdType=TraceIdTypeEnum.EXTERNAL, metadata="test"))

    langfuse.flush()


def test_notebook():
    langfuse = Langfuse("pk-lf-1234567890", "sk-lf-1234567890", host)

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

    langfuse.flush()


def test_full_nested_example():
    langfuse = Langfuse("pk-lf-1234567890", "sk-lf-1234567890", host)

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

    time.sleep(1)

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

    time.sleep(1)

    dbSearchStart = datetime.datetime.now()
    time.sleep(1)
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

    time.sleep(1)

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

    langfuse.flush()


def test_customer_nested():
    langfuse = Langfuse("pk-lf-1234567890", "sk-lf-1234567890", host)

    span = langfuse.span(
        Span(
            name="chat-completion-top",
            userId="user__935d7d1d-8625-4ef4-8651-544613e7bd22",
            metadata={
                "env": "production",
            },
            input={"key": "value"},
            output={"key": "value"},
            traceId="this-is-an-external-id-1",
            traceIdType=TraceIdTypeEnum.EXTERNAL,
        )
    )

    langfuse.span(
        Span(
            name="retrieval",
            userId="user__935d7d1d-8625-4ef4-8651-544613e7bd22",
            metadata={
                "env": "production",
            },
            input={"key": "value"},
            output={"key": "value"},
            parentObservationId=span.id,
            traceId="this-is-an-external-id-1",
            traceIdType=TraceIdTypeEnum.EXTERNAL,
        )
    )

    langfuse.generation(
        Generation(
            name="retrieval",
            userId="user__935d7d1d-8625-4ef4-8651-544613e7bd22",
            metadata={
                "env": "production",
            },
            prompt={"role": "client", "message": "some message"},
            completion="completion string",
            parentObservationId=span.id,
            traceId="this-is-an-external-id-1",
            traceIdType=TraceIdTypeEnum.EXTERNAL,
        )
    )
    langfuse.flush()


@pytest.mark.asyncio
async def test_customer_nested_async():
    langfuse = LangfuseAsync("pk-lf-1234567890", "sk-lf-1234567890", host)

    span = await langfuse.span(
        Span(
            name="chat-completion-top",
            userId="user__935d7d1d-8625-4ef4-8651-544613e7bd22",
            metadata={
                "env": "production",
            },
            input={"key": "value"},
            output={"key": "value"},
            traceId="this-is-an-external-id-1",
            traceIdType=TraceIdTypeEnum.EXTERNAL,
        )
    )

    await langfuse.span(
        Span(
            name="retrieval",
            userId="user__935d7d1d-8625-4ef4-8651-544613e7bd22",
            metadata={
                "env": "production",
            },
            input={"key": "value"},
            output={"key": "value"},
            parentObservationId=span.id,
            traceId="this-is-an-external-id-1",
            traceIdType=TraceIdTypeEnum.EXTERNAL,
        )
    )

    await langfuse.generation(
        Generation(
            name="retrieval",
            userId="user__935d7d1d-8625-4ef4-8651-544613e7bd22",
            metadata={
                "env": "production",
            },
            prompt={"role": "client", "message": "some message"},
            completion="completion string",
            parentObservationId=span.id,
            traceId="this-is-an-external-id-1",
            traceIdType=TraceIdTypeEnum.EXTERNAL,
        )
    )
    await langfuse.flush()


@pytest.mark.asyncio
async def test_customer_root_async():
    langfuse = LangfuseAsync("pk-lf-1234567890", "sk-lf-1234567890", host)

    await langfuse.span(
        Span(
            name="retrieval",
            userId="user__935d7d1d-8625-4ef4-8651-544613e7bd22",
            metadata={
                "env": "production",
            },
            input={"key": "value"},
            output={"key": "value"},
            traceId="this-is-an-external-id",
            traceIdType=TraceIdTypeEnum.EXTERNAL,
        )
    )

    await langfuse.generation(
        Generation(
            name="compeletion",
            userId="user__935d7d1d-8625-4ef4-8651-544613e7bd22",
            metadata={
                "env": "production",
            },
            prompt={"role": "client", "message": "some message"},
            completion="completion string",
            traceId="this-is-an-external-id",
            traceIdType=TraceIdTypeEnum.EXTERNAL,
        )
    )

    await langfuse.flush()


def test_customer_root():
    langfuse = Langfuse("pk-lf-1234567890", "sk-lf-1234567890", host)

    langfuse.span(
        Span(
            name="retrieval",
            userId="user__935d7d1d-8625-4ef4-8651-544613e7bd22",
            metadata={
                "env": "production",
            },
            input={"key": "value"},
            output={"key": "value"},
            traceId="this-is-an-external-id",
            traceIdType=TraceIdTypeEnum.EXTERNAL,
        )
    )

    langfuse.generation(
        Generation(
            name="compeletion",
            userId="user__935d7d1d-8625-4ef4-8651-544613e7bd22",
            metadata={
                "env": "production",
            },
            prompt={"role": "client", "message": "some message"},
            completion="completion string",
            traceId="this-is-an-external-id",
            traceIdType=TraceIdTypeEnum.EXTERNAL,
        )
    )

    langfuse.flush()


def test_customer_blub():
    langfuse = Langfuse("pk-lf-1234567890", "sk-lf-1234567890", host)

    span = langfuse.span(
        Span(
            name="retrieval",
            userId="user__935d7d1d-8625-4ef4-8651-544613e7bd22",
            metadata={
                "env": "production",
            },
            input={"key": "value"},
            output={"key": "value"},
            traceId="this-is-an-external-id",
            traceIdType=TraceIdTypeEnum.EXTERNAL,
        )
    )

    span.generation(
        Generation(
            name="compeletion",
            userId="user__935d7d1d-8625-4ef4-8651-544613e7bd22",
            metadata={
                "env": "production",
            },
            prompt={"role": "client", "message": "some message"},
            completion="completion string",
            traceId="this-is-an-external-id",
            traceIdType=TraceIdTypeEnum.EXTERNAL,
        )
    )

    langfuse.flush()


@pytest.mark.asyncio
async def test_customer_blub_async():
    langfuse = LangfuseAsync("pk-lf-1234567890", "sk-lf-1234567890", host)

    span = await langfuse.span(
        Span(
            name="retrieval",
            userId="user__935d7d1d-8625-4ef4-8651-544613e7bd22",
            metadata={
                "env": "production",
            },
            input={"key": "value"},
            output={"key": "value"},
            traceId="this-is-an-external-id",
            traceIdType=TraceIdTypeEnum.EXTERNAL,
        )
    )

    await span.generation(
        Generation(
            name="compeletion",
            userId="user__935d7d1d-8625-4ef4-8651-544613e7bd22",
            metadata={
                "env": "production",
            },
            prompt={"role": "client", "message": "some message"},
            completion="completion string",
            traceId="this-is-an-external-id",
            traceIdType=TraceIdTypeEnum.EXTERNAL,
        )
    )

    await langfuse.flush()
