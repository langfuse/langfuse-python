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
        self.consumer = asyncio.create_task(self.process_tasks())
        self.lock = asyncio.Lock()  # Lock for synchronizing add_task and wait_for_completion

    async def process_tasks(self):
        logging.info("Prcessing tasks...")
        while True:
            try:
                task = await self.tasks.get()
                logging.info(f"Processing task {task.id} with dependency {task.prev_id}...")

                # Wait for the task's dependency to complete
                if task.prev_id is not None:
                    while task.prev_id not in self.result_mapping:
                        await asyncio.sleep(0)  # Give control back to the event loop

                prev_result = self.result_mapping.get(task.prev_id)[1] if task.prev_id else None
                logging.info(f"Task {task.id} dependency completed with result {prev_result}")
                try:
                    result = await task.coro(prev_result)
                    self.result_mapping[task.id] = (True, result)
                    logging.info(f"Dropping task {task.prev_id} from result mapping")
                    # self.result_mapping.pop(task.prev_id, None)
                    logging.info(f"Task {task.id} completed successfully with result {result}")
                except Exception as e:
                    self.result_mapping[task.id] = (False, e)  # Task failed, store exception
                    # self.clear_task_queue()  # Clear the task queue after a failure
                    logging.error(f"Task {task.id} failed with exception {e}")
                    break  # Stop processing tasks after the first failure

                self.tasks.task_done()
            except asyncio.CancelledError:
                logging.info("Task processing was cancelled")
                break  # Break the loop when cancelled
            finally:
                logging.info(f"Finally, {self.tasks.empty()}")
                if self.tasks.empty():
                    logging.info("No more tasks to process")
                    self.processing = False
                    break  # Break the loop when no more tasks

    async def add_task(self, task):
        async with self.lock:  # Acquire the lock before adding task
            try:
                logging.info(f"Adding task {task.id} with dependency {task.prev_id}...")
                if not self.processing:
                    logging.info("Starting task processing...")
                    self.consumer = asyncio.create_task(self.process_tasks())
                await self.tasks.put(task)
            except Exception as e:
                logging.error(f"Failed to add task {task.id} with exception {e}")

    def clear_task_queue(self):
        while not self.tasks.empty():
            self.tasks.get_nowait()

    async def wait_for_completion(self):
        async with self.lock:  # Acquire the lock before waiting for completion
            try:
                logging.info("Waiting for task processing to complete...")
                while self.processing:
                    await asyncio.sleep(0)

                for task_id, task in self.result_mapping.items():
                    if not task[0]:
                        return {"status": "fail", "failure_reason": task[1]}

                if self.consumer:
                    self.consumer.cancel()
                    try:
                        await self.consumer
                    except asyncio.CancelledError:
                        logging.info("Consumer task cancelled successfully")
                self.consumer = None

                logging.info(f"task manager result mapping {self.result_mapping.keys()}, {self.tasks.qsize()}")
                return {"status": "success"}
            except Exception as e:
                logging.error(f"Failed to wait for task completion with exception {e}")
                return {"status": "fail", "failure_reason": e}

    def __del__(self):
        # Cleanup consumer when the TaskManager instance is destroyed
        if self.consumer is not None:
            self.consumer.cancel()
            try:
                asyncio.get_event_loop().run_until_complete(self.consumer)
            except asyncio.CancelledError:
                pass
            self.consumer = None
