"""@private"""

import atexit
import json
import logging
import queue
import threading
from queue import Empty, Queue
import time
from typing import List, Any, Optional
import typing

from langfuse.Sampler import Sampler
from langfuse.parse_error import handle_exception
from langfuse.request import APIError
from langfuse.utils import _get_timestamp
from langfuse.types import MaskFunction

try:
    import pydantic.v1 as pydantic  # type: ignore
except ImportError:
    import pydantic  # type: ignore


import backoff

from langfuse.request import LangfuseClient
from langfuse.serializer import EventSerializer

# largest message size in db is 331_000 bytes right now
MAX_MSG_SIZE = 1_000_000

# https://vercel.com/docs/functions/serverless-functions/runtimes#request-body-size
# The maximum payload size for the request body or the response body of a Serverless Function is 4.5 MB
# 4_500_000 Bytes = 4.5 MB
# configured to be 3 MB to be safe

BATCH_SIZE_LIMIT = 2_500_000


class LangfuseMetadata(pydantic.BaseModel):
    batch_size: int
    sdk_integration: typing.Optional[str] = None
    sdk_name: str = None
    sdk_version: str = None
    public_key: str = None


class Consumer(threading.Thread):
    _log = logging.getLogger("langfuse")
    _queue: Queue
    _identifier: int
    _client: LangfuseClient
    _flush_at: int
    _flush_interval: float
    _max_retries: int
    _public_key: str
    _sdk_name: str
    _sdk_version: str
    _sdk_integration: str

    def __init__(
        self,
        queue: Queue,
        identifier: int,
        client: LangfuseClient,
        flush_at: int,
        flush_interval: float,
        max_retries: int,
        public_key: str,
        sdk_name: str,
        sdk_version: str,
        sdk_integration: str,
    ):
        """Create a consumer thread."""
        threading.Thread.__init__(self)
        # Make consumer a daemon thread so that it doesn't block program exit
        self.daemon = True
        self._queue = queue
        # It's important to set running in the constructor: if we are asked to
        # pause immediately after construction, we might set running to True in
        # run() *after* we set it to False in pause... and keep running
        # forever.
        self.running = True
        self._identifier = identifier
        self._client = client
        self._flush_at = flush_at
        self._flush_interval = flush_interval
        self._max_retries = max_retries
        self._public_key = public_key
        self._sdk_name = sdk_name
        self._sdk_version = sdk_version
        self._sdk_integration = sdk_integration

    def _next(self):
        """Return the next batch of items to upload."""
        queue = self._queue
        items = []

        start_time = time.monotonic()
        total_size = 0

        while len(items) < self._flush_at:
            elapsed = time.monotonic() - start_time
            if elapsed >= self._flush_interval:
                break
            try:
                item = queue.get(block=True, timeout=self._flush_interval - elapsed)

                item_size = self._truncate_item_in_place(
                    item=item,
                    max_size=MAX_MSG_SIZE,
                    log_message="<truncated due to size exceeding limit>",
                )

                items.append(item)
                total_size += item_size
                if total_size >= BATCH_SIZE_LIMIT:
                    self._log.debug("hit batch size limit (size: %d)", total_size)
                    break

            except Empty:
                break
        self._log.debug("~%d items in the Langfuse queue", self._queue.qsize())

        return items

    def _truncate_item_in_place(
        self,
        *,
        item: typing.Any,
        max_size: int,
        log_message: typing.Optional[str] = None,
    ) -> int:
        """Truncate the item in place to fit within the size limit."""
        item_size = self._get_item_size(item)
        self._log.debug(f"item size {item_size}")

        if item_size > max_size:
            self._log.warning(
                "Item exceeds size limit (size: %s), dropping input / output / metadata of item until it fits.",
                item_size,
            )

            if "body" in item:
                drop_candidates = ["input", "output", "metadata"]
                sorted_field_sizes = sorted(
                    [
                        (
                            field,
                            self._get_item_size((item["body"][field]))
                            if field in item["body"]
                            else 0,
                        )
                        for field in drop_candidates
                    ],
                    key=lambda x: x[1],
                )

                # drop the largest field until the item size is within the limit
                for _ in range(len(sorted_field_sizes)):
                    field_to_drop, size_to_drop = sorted_field_sizes.pop()

                    if field_to_drop not in item["body"]:
                        continue

                    item["body"][field_to_drop] = log_message
                    item_size -= size_to_drop

                    self._log.debug(
                        f"Dropped field {field_to_drop}, new item size {item_size}"
                    )

                    if item_size <= max_size:
                        break

            # if item does not have body or input/output fields, drop the event
            if "body" not in item or (
                "input" not in item["body"] and "output" not in item["body"]
            ):
                self._log.warning(
                    "Item does not have body or input/output fields, dropping item."
                )
                self._queue.task_done()
                return 0

        return self._get_item_size(item)

    def _get_item_size(self, item: typing.Any) -> int:
        """Return the size of the item in bytes."""
        return len(json.dumps(item, cls=EventSerializer).encode())

    def run(self):
        """Runs the consumer."""
        self._log.debug("consumer is running...")
        while self.running:
            self.upload()

    def upload(self):
        """Upload the next batch of items, return whether successful."""
        batch = self._next()
        if len(batch) == 0:
            return

        try:
            self._upload_batch(batch)
        except Exception as e:
            handle_exception(e)
        finally:
            # mark items as acknowledged from queue
            for _ in batch:
                self._queue.task_done()

    def pause(self):
        """Pause the consumer."""
        self.running = False

    def _upload_batch(self, batch: List[Any]):
        self._log.debug("uploading batch of %d items", len(batch))

        metadata = LangfuseMetadata(
            batch_size=len(batch),
            sdk_integration=self._sdk_integration,
            sdk_name=self._sdk_name,
            sdk_version=self._sdk_version,
            public_key=self._public_key,
        ).dict()

        @backoff.on_exception(
            backoff.expo, Exception, max_tries=self._max_retries, logger=None
        )
        def execute_task_with_backoff(batch: List[Any]):
            try:
                self._client.batch_post(batch=batch, metadata=metadata)
            except Exception as e:
                if (
                    isinstance(e, APIError)
                    and 400 <= int(e.status) < 500
                    and int(e.status) != 429  # retry if rate-limited
                ):
                    return

                raise e

        execute_task_with_backoff(batch)
        self._log.debug("successfully uploaded batch of %d items", len(batch))


class TaskManager(object):
    _log = logging.getLogger("langfuse")
    _consumers: List[Consumer]
    _enabled: bool
    _threads: int
    _max_task_queue_size: int
    _queue: Queue
    _client: LangfuseClient
    _flush_at: int
    _flush_interval: float
    _max_retries: int
    _public_key: str
    _sdk_name: str
    _sdk_version: str
    _sdk_integration: str
    _sampler: Sampler
    _mask: Optional[MaskFunction]

    def __init__(
        self,
        client: LangfuseClient,
        flush_at: int,
        flush_interval: float,
        max_retries: int,
        threads: int,
        public_key: str,
        sdk_name: str,
        sdk_version: str,
        sdk_integration: str,
        enabled: bool = True,
        max_task_queue_size: int = 100_000,
        sample_rate: float = 1,
        mask: Optional[MaskFunction] = None,
    ):
        self._max_task_queue_size = max_task_queue_size
        self._threads = threads
        self._queue = queue.Queue(self._max_task_queue_size)
        self._consumers = []
        self._client = client
        self._flush_at = flush_at
        self._flush_interval = flush_interval
        self._max_retries = max_retries
        self._public_key = public_key
        self._sdk_name = sdk_name
        self._sdk_version = sdk_version
        self._sdk_integration = sdk_integration
        self._enabled = enabled
        self._sampler = Sampler(sample_rate)
        self._mask = mask

        self.init_resources()

        # cleans up when the python interpreter closes
        atexit.register(self.join)

    def init_resources(self):
        for i in range(self._threads):
            consumer = Consumer(
                queue=self._queue,
                identifier=i,
                client=self._client,
                flush_at=self._flush_at,
                flush_interval=self._flush_interval,
                max_retries=self._max_retries,
                public_key=self._public_key,
                sdk_name=self._sdk_name,
                sdk_version=self._sdk_version,
                sdk_integration=self._sdk_integration,
            )
            consumer.start()
            self._consumers.append(consumer)

    def add_task(self, event: dict):
        if not self._enabled:
            return

        try:
            if not self._sampler.sample_event(event):
                return  # event was sampled out

            self._apply_mask_in_place(event)

            json.dumps(event, cls=EventSerializer)
            event["timestamp"] = _get_timestamp()

            self._queue.put(event, block=False)
        except queue.Full:
            self._log.warning("analytics-python queue is full")
            return False
        except Exception as e:
            self._log.exception(f"Exception in adding task {e}")

            return False

    def _apply_mask_in_place(self, event: dict):
        """Apply the mask function to the event. This is done in place."""
        if not self._mask:
            return

        body = event["body"] if "body" in event else {}
        for key in ("input", "output"):
            if key in body:
                try:
                    body[key] = self._mask(data=body[key])
                except Exception as e:
                    self._log.error(f"Mask function failed with error: {e}")
                    body[key] = "<fully masked due to failed mask function>"

    def flush(self):
        """Force a flush from the internal queue to the server."""
        self._log.debug("flushing queue")
        queue = self._queue
        size = queue.qsize()
        queue.join()
        # Note that this message may not be precise, because of threading.
        self._log.debug("successfully flushed about %s items.", size)

    def join(self):
        """End the consumer threads once the queue is empty.

        Blocks execution until finished
        """
        self._log.debug(f"joining {len(self._consumers)} consumer threads")

        # pause all consumers before joining them so we don't have to wait for multiple
        # flush intervals to join them all.
        for consumer in self._consumers:
            consumer.pause()

        for consumer in self._consumers:
            try:
                consumer.join()
            except RuntimeError:
                # consumer thread has not started
                pass

            self._log.debug(f"consumer thread {consumer._identifier} joined")

    def shutdown(self):
        """Flush all messages and cleanly shutdown the client."""
        self._log.debug("shutdown initiated")

        self.flush()
        self.join()

        self._log.debug("shutdown completed")
