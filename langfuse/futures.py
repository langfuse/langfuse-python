import concurrent.futures
import asyncio

class FuturesStore:
    def __init__(self):
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=10)
        self.futures = {}

    def append(self, id:  str, func, *args, future_id=None, **kwargs):


        if future_id is None:
            # If no future_id is provided, store the function and arguments for later execution
            self.futures[id] = (func, args, kwargs)
        else:
            # If a future_id is provided, create a dependent future and store it for later execution
            dependent_future = (func, args, kwargs, future_id)
            self.futures[id] = dependent_future

        return id

    def flush(self):
        return asyncio.run(self.async_flush())

    async def async_flush(self):
        async def run_task(future_id, future_details):
            if len(future_details) == 3:  # If there are no dependencies
                func, args, kwargs = future_details
                result = await func(*args, **kwargs)
            else:  # If there is a dependency
                func, args, kwargs, dependent_id = future_details
                # Wait for the dependency to resolve before running this task
                await tasks[dependent_id]
                dependent_result = results[dependent_id]  # get the result from the parent future
                result = await func(dependent_result, *args, **kwargs)
            results[future_id] = result

        results = {}
        tasks = {}
        final_result = {'status': 'success'}

        try:
            # First, create all the tasks but don't run them yet
            for future_id, future_details in self.futures.items():
                tasks[future_id] = asyncio.create_task(run_task(future_id, future_details))

            # Then, run all the tasks concurrently
            for task in tasks.values():
                await task
        except Exception as e:
            print(e)
            final_result['status'] = 'failed'
            final_result['error'] = str(e)

        self.futures.clear()

        return final_result



