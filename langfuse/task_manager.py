import atexit
import json
import logging
import queue
from queue import Empty, Queue
import threading
from typing import List
import monotonic

import backoff

from langfuse.logging import clean_logger
from langfuse.request import LangfuseClient
from langfuse.serializer import DatetimeSerializer


MAX_MSG_SIZE = 32 << 10

# Our servers only accept batches less than 500KB. Here limit is set slightly
# lower to leave space for extra data that will be added later, eg. "sentAt".
BATCH_SIZE_LIMIT = 475000


class Task:
    data: any
    task_id: int

    def __init__(self, task_id, data):
        self.task_id = task_id
        self.data = data


class Consumer(threading.Thread):
    _log = logging.getLogger("langfuse")
    _queue: Queue
    _identifier: int
    _client: LangfuseClient
    _flush_at: int
    _flush_interval: float

    def __init__(self, queue: Queue, identifier: int, client: LangfuseClient, flush_at=100, flush_interval=0.5):
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

    def _next(self):
        """Return the next batch of items to upload."""

        self._log.debug("consumer thread %s is getting next batch", self._identifier)
        queue = self._queue
        items = []

        start_time = monotonic.monotonic()
        total_size = 0

        while len(items) < self._flush_at:
            self._log.debug("consumer thread %s is waiting for next item", self._identifier)
            elapsed = monotonic.monotonic() - start_time
            if elapsed >= self._flush_interval:
                break
            try:
                self._log.debug("getting from q")
                item = queue.get(block=True, timeout=self._flush_interval - elapsed)

                item_size = len(json.dumps(item.data, cls=DatetimeSerializer).encode())

                items.append(item.data)
                total_size += item_size
                if total_size >= BATCH_SIZE_LIMIT:
                    self._log.debug("hit batch size limit (size: %d)", total_size)
                    break
            except Empty:
                break

        return items

    def run(self):
        """Runs the consumer."""
        while self.running:
            self._log.debug("consumer thread %s is running", self._identifier)
            success = False
            batch = self._next()
            if len(batch) == 0:
                return False

            try:
                self._upload_batch(batch)
                success = True
            except Exception as e:
                self._log.error("error uploading: %s", e)
                success = False
                if self.on_error:
                    self.on_error(e, batch)
            finally:
                # mark items as acknowledged from queue
                for item in batch:
                    self._queue.task_done()
                return success

    def pause(self):
        """Pause the consumer."""
        self.running = False

    def _upload_batch(self, batch: List[Task]):
        @backoff.on_exception(backoff.expo, Exception, max_tries=3)
        def execute_task_with_backoff(batch: [Task]):
            return self._client.batch_post(gzip=False, batch=batch)

        execute_task_with_backoff(batch)


class TaskManager(object):
    log = logging.getLogger("langfuse")
    consumers: List[Consumer]
    number_of_consumers: int
    max_task_queue_size: int
    queue: Queue
    _client: LangfuseClient

    def __init__(self, client: LangfuseClient, debug=False, max_task_queue_size=10_000, number_of_consumers=1):
        self.max_task_queue_size = max_task_queue_size
        self.number_of_consumers = number_of_consumers
        self.queue = queue.Queue(max_task_queue_size)
        self.consumers = []
        self._client = client

        if debug:
            # Ensures that debug level messages are logged when debug mode is on.
            # Otherwise, defaults to WARNING level.
            # See https://docs.python.org/3/howto/logging.html#what-happens-if-no-configuration-is-provided
            logging.basicConfig()
            self.log.setLevel(logging.DEBUG)
            clean_logger()
        else:
            self.log.setLevel(logging.WARNING)
            clean_logger()

        self.init_resources()

        # cleans up when the python interpreter closes
        atexit.register(self.join)

    def init_resources(self):
        for i in range(self.number_of_consumers):
            consumer = Consumer(self.queue, i, self._client)
            consumer.start()
            self.consumers.append(consumer)

    def add_task(self, task_id, function):
        try:
            self.log.debug(f"Adding task {task_id}")
            task = Task(task_id, function)
            self.queue.put(task, block=False)
            self.log.debug(f"Task {task_id} added to queue")
        except queue.Full:
            self.log.warning("analytics-python queue is full")
            return False
        except Exception as e:
            self.log.warning(f"Exception in adding task {task_id} {e}")
            return False

    def flush(self):
        """Forces a flush from the internal queue to the server"""
        self.log.debug("flushing queue")
        queue = self.queue
        size = queue.qsize()
        queue.join()
        # Note that this message may not be precise, because of threading.
        self.log.debug("successfully flushed about %s items.", size)

    def join(self):
        """Ends the consumer threads once the queue is empty.
        Blocks execution until finished
        """
        self.log.debug(f"joining {len(self.consumers)} consumer threads")
        for consumer in self.consumers:
            consumer.pause()
            try:
                consumer.join()
            except RuntimeError:
                # consumer thread has not started
                pass

            self.log.debug(f"consumer thread {consumer._identifier} joined")

    def shutdown(self):
        """Flush all messages and cleanly shutdown the client"""
        self.log.debug("shutdown initiated")
        self.flush()
        self.join()
        self.log.debug("shutdown completed")
