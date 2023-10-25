import atexit
import logging
import queue
from queue import Queue
import threading
from typing import List

import backoff

from langfuse.logging import clean_logger


class Task:
    def __init__(self, task_id, function):
        self.task_id = task_id
        self.function = function


class Consumer(threading.Thread):
    log = logging.getLogger("langfuse")
    queue: Queue
    identifier: int

    def __init__(self, queue, identifier):
        """Create a consumer thread."""

        threading.Thread.__init__(self)
        # Make consumer a daemon thread so that it doesn't block program exit
        self.daemon = True
        self.queue = queue
        # It's important to set running in the constructor: if we are asked to
        # pause immediately after construction, we might set running to True in
        # run() *after* we set it to False in pause... and keep running
        # forever.
        self.running = True
        self.identifier = identifier

    def run(self):
        """Runs the consumer."""
        while self.running:
            try:
                task = self.queue.get(block=True, timeout=1)

                self.log.debug(f"Task {task.task_id} received from the queue")

                self._execute_task(task)
                self.log.debug(f"Task {task.task_id} done")
                self.queue.task_done()

            except queue.Empty:
                pass

    def pause(self):
        """Pause the consumer."""
        self.running = False

    def _execute_task(self, task: Task):
        try:
            self.log.debug(f"Task {task.task_id} executing")
            result = self._execute_task_with_backoff(task)
            self.log.debug(f"Task {task.task_id} done with result {result}")
        except Exception as e:
            self.log.warning(f"Task {task.task_id} failed with exception {e} ")

    @backoff.on_exception(backoff.expo, Exception, max_tries=3)
    def _execute_task_with_backoff(self, task: Task):
        return task.function()


class TaskManager(object):
    log = logging.getLogger("langfuse")
    consumers: List[Consumer]
    number_of_consumers: int
    max_task_queue_size: int
    queue: Queue

    def __init__(self, debug=False, max_task_queue_size=10_000, number_of_consumers=1):
        self.max_task_queue_size = max_task_queue_size
        self.number_of_consumers = number_of_consumers
        self.queue = queue.Queue(max_task_queue_size)
        self.consumers = []
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
            consumer = Consumer(self.queue, i)
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

            self.log.debug(f"consumer thread {consumer.identifier} joined")

    def shutdown(self):
        """Flush all messages and cleanly shutdown the client"""
        self.log.debug("shutdown initiated")
        self.flush()
        self.join()
        self.log.debug("shutdown completed")
