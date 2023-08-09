import datetime
import time


from langfuse import Langfuse
from langfuse.model import CreateEvent, CreateGeneration, CreateScore, CreateSpan, CreateTrace, InitialGeneration, InitialScore, InitialSpan, UpdateGeneration, UpdateSpan, Usage, TraceIdTypeEnum, ObservationLevel

from langfuse.task_manager import TaskStatus

host = "http://localhost:3000/"


def test_flush():
    # set up the consumer with more requests than a single batch will allow
    langfuse = Langfuse("pk-lf-1234567890", "sk-lf-1234567890", host)

    for i in range(2):
        langfuse.trace(
            CreateTrace(
                name=str(i),
            )
        )

    langfuse.flush()
    # Make sure that the client queue is empty after flushing
    assert langfuse.task_manager.queue.empty()


def test_shutdown():
    langfuse = Langfuse("pk-lf-1234567890", "sk-lf-1234567890", host)

    for i in range(2):
        langfuse.trace(
            CreateTrace(
                name=str(i),
            )
        )

    langfuse.shutdown()
    # we expect two things after shutdown:
    # 1. client queue is empty
    # 2. consumer thread has stopped
    assert langfuse.task_manager.queue.empty()
    assert not langfuse.task_manager.consumer_thread.is_alive()


def test_create_score():
    langfuse = Langfuse("pk-lf-1234567890", "sk-lf-1234567890", host)
    trace = langfuse.trace(
        CreateTrace(
            name="this-is-so-great-new",
            user_id="test",
            metadata="test",
        )
    )
    langfuse.flush()
    assert langfuse.task_manager.queue.qsize() == 0
    assert all(v.status == TaskStatus.SUCCESS for v in langfuse.task_manager.result_mapping.values()), "Not all tasks succeeded"

    trace = langfuse.score(
        InitialScore(
            traceId=trace.id,
            name="this-is-so-great-new",
            value=1,
            user_id="test",
            metadata="test",
        )
    )

    trace.generation(CreateGeneration(name="yet another child", metadata="test"))

    langfuse.flush()
    assert langfuse.task_manager.queue.qsize() == 0
    assert all(v.status == TaskStatus.SUCCESS for v in langfuse.task_manager.result_mapping.values()), "Not all tasks succeeded"


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

    generation = trace.generation(InitialGeneration(name="new test", metadata="test"))

    # sub_generation = generation.generation(CreateGeneration(name="yet another child", metadata="test"))
    # # result = asyncio.gather(langfuse.async_flush(), langfuse.async_flush())
    # sub_sub_span = sub_generation.span(CreateSpan(name="sub-sub-span", metadata="test"))

    # sub_sub_span = sub_sub_span.score(CreateScore(name="user-explicit-feedback", value=1, comment="I like how personalized the response is"))

    langfuse.flush()
    assert langfuse.task_manager.queue.qsize() == 0
    assert all(v.status == TaskStatus.SUCCESS for v in langfuse.task_manager.result_mapping.values()), "Not all tasks succeeded"


def test_update_generation():
    langfuse = Langfuse("pk-lf-1234567890", "sk-lf-1234567890", host)

    generation = langfuse.generation(CreateGeneration(name="to-be-renames", metadata="test", prompt={"key": "value"}, completion="very long completion"))
    updated_generation = generation.update(UpdateGeneration(name="new-name", metadata="something-else"))
    span = updated_generation.span(CreateSpan(name="sub-span", metadata="test", input={"key": "value"}, output={"key": "value"}))

    span.update(UpdateSpan(level=ObservationLevel.WARNING, metadata="something-else"))

    langfuse.flush()
    assert langfuse.task_manager.queue.qsize() == 0
    assert all(v.status == TaskStatus.SUCCESS for v in langfuse.task_manager.result_mapping.values()), "Not all tasks succeeded"


def test_create_generation():
    langfuse = Langfuse("pk-lf-1234567890", "sk-lf-1234567890", host, release="1.0.0")

    langfuse.generation(CreateGeneration(name="max-top-level-generation", metadata="test"))

    langfuse.flush()
    assert langfuse.task_manager.queue.qsize() == 0
    assert all(v.status == TaskStatus.SUCCESS for v in langfuse.task_manager.result_mapping.values()), "Not all tasks succeeded"


def test_create_span():
    langfuse = Langfuse("pk-lf-1234567890", "sk-lf-1234567890", host, release="1.0.0")

    langfuse.span(InitialSpan(name="max-top-level-span", metadata="test"))

    langfuse.flush()
    assert langfuse.task_manager.queue.qsize() == 0
    assert all(v.status == TaskStatus.SUCCESS for v in langfuse.task_manager.result_mapping.values()), "Not all tasks succeeded"


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
    assert langfuse.task_manager.queue.qsize() == 0
    assert all(v.status == TaskStatus.SUCCESS for v in langfuse.task_manager.result_mapping.values()), "Not all tasks succeeded"


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
    assert langfuse.task_manager.queue.qsize() == 0
    assert all(v.status == TaskStatus.SUCCESS for v in langfuse.task_manager.result_mapping.values()), "Not all tasks succeeded"


def test_customer_nested():
    langfuse = Langfuse("pk-lf-1234567890", "sk-lf-1234567890", host)

    span = langfuse.span(
        InitialSpan(
            name="chat-completion-top",
            userId="user__935d7d1d-8625-4ef4-8651-544613e7bd22",
            metadata={
                "env": "production",
            },
            input={"key": "value"},
            output={"key": "value"},
            traceId="this-is-an-external-id-1",
        )
    )

    langfuse.span(
        InitialSpan(
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
        InitialGeneration(
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
    assert langfuse.task_manager.queue.qsize() == 0
    assert all(v.status == TaskStatus.SUCCESS for v in langfuse.task_manager.result_mapping.values()), "Not all tasks succeeded"


def test_customer_root():
    langfuse = Langfuse("pk-lf-1234567890", "sk-lf-1234567890", host)

    langfuse.span(
        InitialSpan(
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
        InitialGeneration(
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
    assert langfuse.task_manager.queue.qsize() == 0
    assert all(v.status == TaskStatus.SUCCESS for v in langfuse.task_manager.result_mapping.values()), "Not all tasks succeeded"


def test_customer_blub():
    langfuse = Langfuse("pk-lf-1234567890", "sk-lf-1234567890", host)

    span = langfuse.span(
        InitialSpan(
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
        InitialGeneration(
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
    assert langfuse.task_manager.queue.qsize() == 0
    assert all(v.status == TaskStatus.SUCCESS for v in langfuse.task_manager.result_mapping.values()), "Not all tasks succeeded"
