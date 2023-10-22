import asyncio
import asyncio_atexit
import logging

import backoff


class Task:
    def __init__(self, task_id, function):
        self.task_id = task_id
        self.function = function


class Consumer:
    log = logging.getLogger("langfuse")
    queue: asyncio.Queue
    identifier: int
    running: bool

    def __init__(self, queue, identifier):
        self.queue = queue
        self.identifier = identifier
        self.running = True

    async def run(self):
        while self.running:
            self.log.warning(f"Consumer {self.identifier} waiting for task {self.queue.qsize()}")
            task = await self.queue.get()

            if task is None:  # sentinel check
                self.queue.task_done()
                break

            self.log.debug(f"Consumer {self.identifier} got task {task.task_id}")
            try:
                self.log.debug(f"Task {task.task_id} received from the queue")
                await self._execute_task(task)
                self.log.debug(f"Task {task.task_id} done")
            except Exception as e:
                self.log.warning(f"Task {task.task_id} failed with exception {e}")
            finally:
                self.queue.task_done()

    def pause(self):
        """Pause the consumer."""
        self.running = False
        self.queue.put_nowait(None)  # sentinel value

    async def _execute_task(self, task: Task):
        try:
            self.log.debug(f"Task {task.task_id} executing")
            await self._execute_task_with_backoff(task)
        except Exception as e:
            self.log.warning(f"Task {task.task_id} failed with exception {e}")

    @backoff.on_exception(backoff.expo, Exception, max_tries=3)
    async def _execute_task_with_backoff(self, task: Task):
        self.log.debug(f"Task {task.task_id} executing with backoff")
        result = await asyncio.ensure_future(task.function())
        self.log.debug(f"Task {task.task_id} done with result {result}")


class TaskManager:
    log = logging.getLogger("langfuse")
    consumers: list[Consumer]

    def __init__(self, debug=False, max_task_queue_size=10_000):
        self.max_task_queue_size = max_task_queue_size
        self.queue = asyncio.Queue(max_task_queue_size)
        self.consumer_tasks = []
        self.consumers = []
        if debug:
            logging.basicConfig()
            self.log.setLevel(logging.DEBUG)
        else:
            self.log.setLevel(logging.WARNING)

        asyncio_atexit.register(self.join)

    @classmethod
    async def create(cls, debug=False, max_task_queue_size=10_000):
        instance = cls(debug=debug, max_task_queue_size=max_task_queue_size)
        await instance.init_resources()
        return instance

    async def init_resources(self):
        for i in range(20):
            self.log.debug(f"Creating consumer {i}")
            consumer = Consumer(self.queue, i)
            task = asyncio.create_task(consumer.run())
            self.consumers.append(consumer)
            self.consumer_tasks.append(task)
            self.log.debug(f"Consumer {i} created")

    def add_task(self, task_id, function):
        try:
            self.log.debug(f"Adding task {task_id}")
            task = Task(task_id, function)
            self.queue.put_nowait(task)
            self.log.debug(f"Task {task_id} added to queue")
        except asyncio.QueueFull:
            self.log.warning("analytics-python queue is full")
            return False
        except Exception as e:
            self.log.warning(f"Exception in adding task {task_id} {e}")
            return False

    async def flush(self):
        self.log.debug("flushing queue")
        await self.queue.join()
        self.log.debug("successfully flushed the queue")

    async def join(self):
        for consumer in self.consumers:
            consumer.pause()
        await asyncio.gather(*self.consumer_tasks, return_exceptions=True)
        self.log.debug("successfully joined all consumers")

    async def shutdown(self):
        self.log.info("shutdown initiated")
        await self.flush()
        await self.join()
        self.log.info("shutdown completed")
