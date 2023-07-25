import concurrent.futures
import asyncio
import traceback

from langfuse.log import logger


class FuturesStore:
    def __init__(self):
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=10)
        self.futures = {}

    def append(self, id: str, func, *args, future_id=None, **kwargs):
        logger.info(f"Appending task with ID: {id}")

        if future_id is None:
            print(f"Task {id} depends on task with ID: {future_id}")
            # If no future_id is provided, store the function and arguments for later execution
            self.futures[id] = (func, args, kwargs)
        else:
            # If a future_id is provided, create a dependent future and store it for later execution
            dependent_future = (func, args, kwargs, future_id)
            self.futures[id] = dependent_future

        return id

    async def flush(self):
        async def run_task(future_id, future_details):
            try:
                logger.info(f"Running task with ID: {future_id} and details: {future_details}")

                if len(future_details) == 3:  # If there are no dependencies
                    func, args, kwargs = future_details
                    result = await func(*args, **kwargs)
                else:  # If there is a dependency
                    func, args, kwargs, dependent_id = future_details
                    # Wait for the dependency to resolve before running this task
                    await tasks[dependent_id]
                    # get the result from the parent future
                    dependent_result = results[dependent_id]
                    result = await func(dependent_result, *args, **kwargs)
                results[future_id] = result
            except Exception as e:
                traceback.print_exception(e)
                raise e

        results = {}
        tasks = {}
        final_result = {"status": "success"}
        try:
            # First, create all the tasks but don't run them yet
            for future_id, future_details in self.futures.items():
                tasks[future_id] = asyncio.create_task(run_task(future_id, future_details))

            # Then, run all the tasks concurrently
            for task in tasks.values():
                await task
        except Exception as e:
            traceback.print_exception(e)
            final_result["status"] = "failed"
            final_result["error"] = str(e)
            raise e

        self.futures.clear()
        self.results = {}
        self.taks = {}

        return final_result
