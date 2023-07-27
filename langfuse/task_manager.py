import asyncio
import logging


class Task:
    def __init__(self, id, coro, prev_id=None):
        self.id = id
        self.coro = coro
        self.prev_id = prev_id


class TaskManager:
    def __init__(self):
        self.tasks = asyncio.Queue()
        self.result_mapping = {}
        self.processing = True
        self.loop = asyncio.get_running_loop()
        self.consumer = asyncio.create_task(self.process_tasks())
        self.lock = asyncio.Lock()  # Lock for synchronizing add_task and wait_for_completion
        self.result_mapping_lock = asyncio.Lock()  # Lock to sync  # Lock to synchronize result mapping

    async def process_tasks(self):
        logging.info("Processing tasks...")
        while self.processing:
            task = await self.tasks.get()
            logging.info(f"Processing task {task.id} with dependency {task.prev_id}...")

            # Wait for the task's dependency to complete
            if task.prev_id is not None:
                prev_task_future = self.result_mapping.get(task.prev_id)
                if prev_task_future is not None:
                    await prev_task_future  # Suspend until the dependency is completed
                prev_result = prev_task_future.result() if prev_task_future is not None else None
            else:
                prev_result = None

            logging.info(f"Task {task.id} dependency completed with result {prev_result}")
            try:
                result = await task.coro(prev_result)
                task_future = self.loop.create_future()
                task_future.set_result(result)
                self.result_mapping[task.id] = task_future  # Map task id to its future
                logging.info(f"Task {task.id} completed successfully with result {result}")
            except Exception as e:
                task_future = self.loop.create_future()
                task_future.set_exception(e)
                self.result_mapping[task.id] = task_future  # Map task id to its future even in case of error
                logging.error(f"Task {task.id} failed with exception {e}")
                self.processing = False  # Stop processing tasks after the first failure

            self.tasks.task_done()
            if self.tasks.empty():
                logging.info("No more tasks to process")
                self.processing = False

    async def add_task(self, task):
        async with self.lock:
            logging.info(f"Adding task {task.id} with dependency {task.prev_id}...")
            if not self.processing:
                logging.info("Restarting task processing...")
                self.processing = True
                self.consumer = asyncio.create_task(self.process_tasks())
            await self.tasks.put(task)

    async def wait_for_completion(self):
        async with self.lock:
            logging.info("Waiting for task processing to complete...")
            while self.processing:
                await asyncio.sleep(0)  # Give control back to the event loop
            if self.consumer:
                self.consumer.cancel()
                try:
                    await self.consumer
                except asyncio.CancelledError:
                    logging.info("Consumer task cancelled successfully")
            self.consumer = None
            for task_id, task_future in self.result_mapping.items():
                try:
                    _ = task_future.result()  # Check if the task was successful
                except Exception as e:  # The task failed
                    logging.error(f"Task {task_id} failed with exception {e}")
                    return {"status": "fail", "failure_reason": str(e)}
            logging.info(f"Task manager result mapping {self.result_mapping.keys()}, {self.tasks.qsize()}")
            return {"status": "success"}

    def __del__(self):
        if self.consumer is not None:
            self.consumer.cancel()
            try:
                self.loop.run_until_complete(self.consumer)
            except asyncio.CancelledError:
                pass
            self.consumer = None
