import atexit
from datetime import datetime
import json
import logging
import queue
from queue import Empty, Queue
import threading
from typing import List
import monotonic
from dateutil.tz import tzutc


import backoff

from langfuse.request import LangfuseClient
from langfuse.serializer import EventSerializer

# largest message size in db is 331_000 bytes right now
MAX_MSG_SIZE = 650_000

# https://vercel.com/docs/functions/serverless-functions/runtimes#request-body-size
# The maximum payload size for the request body or the response body of a Serverless Function is 4.5 MB
# 4_500_000 Bytes = 4.5 MB
# https://nextjs.org/docs/pages/building-your-application/routing/api-routes#custom-config
# The default nextjs body parser takes a max body size of 1mb. Hence, our BATCH_SIZE_LIMIT should be less to accomodate the final event.
BATCH_SIZE_LIMIT = 650_000


class Consumer(threading.Thread):
    _log = logging.getLogger("langfuse")
    _queue: Queue
    _identifier: int
    _client: LangfuseClient
    _flush_at: int
    _flush_interval: float
    _max_retries: int

    def __init__(self, queue: Queue, identifier: int, client: LangfuseClient, flush_at: int, flush_interval: float, max_retries: int):
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

    def _next(self):
        """Return the next batch of items to upload."""

        queue = self._queue
        items = []

        start_time = monotonic.monotonic()
        total_size = 0

        while len(items) < self._flush_at:
            elapsed = monotonic.monotonic() - start_time
            if elapsed >= self._flush_interval:
                break
            try:
                item = queue.get(block=True, timeout=self._flush_interval - elapsed)
                item_size = len(json.dumps(item, cls=EventSerializer).encode())
                self._log.debug(f"item size {item_size}")
                if item_size > MAX_MSG_SIZE:
                    self._log.warning("Item exceeds size limit (size: %s), dropping item. (%s)", item_size, item)
                    continue
                items.append(item)
                total_size += item_size
                if total_size >= BATCH_SIZE_LIMIT:
                    self._log.debug("hit batch size limit (size: %d)", total_size)
                    break

            except Empty:
                break

        return items

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
            self._log.exception("error uploading: %s", e)
        finally:
            # mark items as acknowledged from queue
            for _ in batch:
                self._queue.task_done()

    def pause(self):
        """Pause the consumer."""
        self.running = False

    def _upload_batch(self, batch: List[any]):
        self._log.debug("uploading batch of %d items", len(batch))

        @backoff.on_exception(backoff.expo, Exception, max_tries=self._max_retries)
        def execute_task_with_backoff(batch: [any]):
            self._log.debug("uploading batch of %d items", len(batch))
            return self._client.batch_post(gzip=False, batch=batch)

        execute_task_with_backoff(batch)
        self._log.debug("successfully uploaded batch of %d items", len(batch))


class TaskManager(object):
    _log = logging.getLogger("langfuse")
    _consumers: List[Consumer]
    _threads: int
    _max_task_queue_size: int
    _queue: Queue
    _client: LangfuseClient
    _flush_at: int
    _flush_interval: float
    _max_retries: int

    def __init__(self, client: LangfuseClient, flush_at: int, flush_interval: float, max_retries: int, threads: int, max_task_queue_size: int = 100_000):
        self._max_task_queue_size = max_task_queue_size
        self._threads = threads
        self._queue = queue.Queue(self._max_task_queue_size)
        self._consumers = []
        self._client = client
        self._flush_at = flush_at
        self._flush_interval = flush_interval
        self._max_retries = max_retries

        self.init_resources()

        # cleans up when the python interpreter closes
        atexit.register(self.join)

    def init_resources(self):
        for i in range(self._threads):
            consumer = Consumer(self._queue, i, self._client, self._flush_at, self._flush_interval, self._max_retries)
            consumer.start()
            self._consumers.append(consumer)

    def add_task(self, event):
        try:
            self._log.debug("Adding task")
            event["timestamp"] = datetime.utcnow().replace(tzinfo=tzutc())
            self._queue.put(event, block=False)
        except queue.Full:
            self._log.warning("analytics-python queue is full")
            return False
        except Exception as e:
            self._log.warning(f"Exception in adding task {e}")
            return False

    def flush(self):
        """Forces a flush from the internal queue to the server"""
        self._log.debug("flushing queue")
        queue = self._queue
        size = queue.qsize()
        queue.join()
        # Note that this message may not be precise, because of threading.
        self._log.debug("successfully flushed about %s items.", size)

    def join(self):
        """Ends the consumer threads once the queue is empty.
        Blocks execution until finished
        """
        self._log.debug(f"joining {len(self._consumers)} consumer threads")
        for consumer in self._consumers:
            consumer.pause()
            try:
                consumer.join()
            except RuntimeError:
                # consumer thread has not started
                pass

            self._log.debug(f"consumer thread {consumer._identifier} joined")

    def shutdown(self):
        """Flush all messages and cleanly shutdown the client"""

        self._log.debug("shutdown initiated")

        self.flush()
        self.join()

        self._log.debug("shutdown completed")
