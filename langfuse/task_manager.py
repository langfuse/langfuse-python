import asyncio
import threading
from langfuse.log import logger


class Task:
    def __init__(self, id, coro, prev_id=None):
        self.id = id
        self.coro = coro
        self.prev_id = prev_id


class TaskManager:
    def __init__(self):
        self.tasks = asyncio.Queue()
        self.processing_tasks = set()
        self.task_mapping = {}
        self.result_mapping = {}  # New dictionary to track results or exceptions of tasks
        # self.all_tasks_done_event = asyncio.Event()
        # self.first_task_failed_event = asyncio.Event()  # New event to indicate the first failure
        # self.stop_processing = False
        # self.loop = None
        asyncio.create_task(self.process_tasks(self.tasks))

    async def process_tasks(self):
        if not self.tasks.empty():
            task = await self.tasks.get()
            logger.info(f"Started processing task with ID: {task.id}")  # Added logging
            self.processing_tasks.add(task.id)

            # Wait for the task's dependency to complete
            if task.prev_id is not None:
                while task.prev_id not in self.result_mapping:
                    await asyncio.sleep(0)  # Give control back to the event loop

            prev_result = self.result_mapping.get(task.prev_id) if task.prev_id else None

            print("prev_result", prev_result, task.id)

            try:
                result = await task.coro(prev_result)
                self.task_mapping[task.id] = result
                self.result_mapping[task.id] = (True, result)  # Task was successful
                logger.info(f"Finished processing task with ID: {task.id} and {result}")  # Added logging
            except Exception as e:
                self.result_mapping[task.id] = (False, e)  # Task failed, store exception
                logger.error(f"Task with ID: {task.id} failed with exception {e}")
                self.first_task_failed_event.set()  # Set failure event
                self.clear_task_queue()  # Clear the task queue after a failure
                break  # Stop processing tasks after the first failure
            finally:
                self.processing_tasks.remove(task.id)

            self.tasks.task_done()

        if self.all_tasks_done():
            print("self.all_tasks_don")
            self.all_tasks_done_event.set()
            self.stop_processing = True  # Break out of the while loop when all tasks are done
            if self.loop and not self.loop.is_closed():  # Check if the loop is running and open
                self.loop.call_soon_threadsafe(self.loop.stop)  # Stop the loop in a threadsafe manner
        else:
            print("Waiting for tasks", self.tasks.empty(), self.processing_tasks, self.processing_task)
        await asyncio.sleep(0)  # Give control back to the event loop

    async def add_task(self, task):
        logger.info(f"Adding task with ID: {task.id}")  # Added logging
        await self.tasks.put(task)

    def all_tasks_done(self):
        return len(self.processing_tasks) == 0 and self.tasks.empty()

    async def await_all_tasks_done(self):
        print("awaiting all tasks done")
        all_tasks_done_event_task = asyncio.create_task(self.all_tasks_done_event.wait())
        first_task_failed_event_task = asyncio.create_task(self.first_task_failed_event.wait())
        done, pending = await asyncio.wait([all_tasks_done_event_task, first_task_failed_event_task], return_when=asyncio.FIRST_COMPLETED)

        print("done", self.processing_task)
        self.processing_task.cancel()
        self.processing_task = None
        self.loop.stop()
        self.deamon.join(2)

        print("done", self.processing_task, self.loop, self.deamon)

        if all_tasks_done_event_task in done:
            return {"status": "success", "failure_reason": None}

        if first_task_failed_event_task in done:
            exceptions = [result for success, result in self.result_mapping.values() if not success]
            return {"status": "fail", "failure_reason": exceptions}
