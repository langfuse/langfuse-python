import logging
from queue import Queue
from unittest.mock import Mock

from pydantic import BaseModel

from langfuse._task_manager import score_ingestion_consumer as sic
from langfuse._task_manager.score_ingestion_consumer import ScoreIngestionConsumer


def _consumer(queue: Queue, client: Mock, flush_at: int = 15):
    return ScoreIngestionConsumer(
        ingestion_queue=queue,
        identifier=0,
        client=client,
        public_key="pk",
        flush_at=flush_at,
        flush_interval=0.1,
    )


def _event(event_id: str, comment: str) -> dict:
    return {"id": event_id, "type": "score-create", "body": {"comment": comment}}


def test_oversized_event_is_dropped_and_others_are_sent(monkeypatch, caplog):
    monkeypatch.setattr(sic, "MAX_EVENT_SIZE_BYTES", 1_000)
    queue = Queue()
    client = Mock()
    # flush_at=2 ends the batch once both valid events are read, independent of timing
    consumer = _consumer(queue, client, flush_at=2)

    queue.put(_event("a", "small"))
    queue.put(_event("big", "x" * 5_000))
    queue.put(_event("c", "small"))

    with caplog.at_level(logging.ERROR):
        consumer.upload()

    batch = client.batch_post.call_args.kwargs["batch"]
    assert [e["id"] for e in batch] == ["a", "c"]
    assert queue.unfinished_tasks == 0
    assert len(caplog.records) == 1
    assert "Score event big is" in caplog.text


def test_only_oversized_event_posts_nothing(monkeypatch):
    monkeypatch.setattr(sic, "MAX_EVENT_SIZE_BYTES", 1_000)
    queue = Queue()
    client = Mock()
    consumer = _consumer(queue, client)

    queue.put(_event("big", "x" * 5_000))

    consumer.upload()

    client.batch_post.assert_not_called()
    assert queue.unfinished_tasks == 0


def test_event_exactly_at_limit_is_sent(monkeypatch):
    queue = Queue()
    client = Mock()
    consumer = _consumer(queue, client, flush_at=1)
    event = _event("edge", "x" * 100)
    size = consumer._get_item_size(event)

    monkeypatch.setattr(sic, "MAX_EVENT_SIZE_BYTES", size)
    queue.put(event)
    consumer.upload()
    assert client.batch_post.call_count == 1

    monkeypatch.setattr(sic, "MAX_EVENT_SIZE_BYTES", size - 1)
    queue.put(_event("edge", "x" * 100))
    consumer.upload()
    assert client.batch_post.call_count == 1


def test_pydantic_body_is_measured_after_dump(monkeypatch):
    class Body(BaseModel):
        comment: str

    monkeypatch.setattr(sic, "MAX_EVENT_SIZE_BYTES", 1_000)
    queue = Queue()
    client = Mock()
    consumer = _consumer(queue, client)

    queue.put({"id": "big", "type": "score-create", "body": Body(comment="x" * 5_000)})

    consumer.upload()

    client.batch_post.assert_not_called()
    assert queue.unfinished_tasks == 0
