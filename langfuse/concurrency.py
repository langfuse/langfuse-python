import concurrent.futures
import asyncio

class FuturesStore:
    def __init__(self):
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=10)
        self.futures = {}

    def append(self, id, func, *args, future_id=None, **kwargs):


        if future_id is None:
            # If no future_id is provided, store the function and arguments for later execution
            self.futures[id] = (func, args, kwargs)
        else:
            # If a future_id is provided, create a dependent future and store it for later execution
            dependent_future = (func, args, kwargs, future_id)
            self.futures[id] = dependent_future

        return id

    def flush(self):
        results = {}
        executed_futures = {}  # A dictionary to store executed futures

        for future_id, future_details in self.futures.items():
            if len(future_details) == 3:  # If there are no dependencies
                
                func, args, kwargs = future_details
                print('no dependencies', func, args, kwargs)
                executed_future = self.executor.submit(asyncio.run, func(*args, **kwargs))
            else:  # If there is a dependency
                func, args, kwargs, dependent_id = future_details
                dependent_result = executed_futures[dependent_id].result()  # get the result from the parent future

                executed_future = self.executor.submit(asyncio.run, func(dependent_result, *args, **kwargs))

            executed_futures[future_id] = executed_future

        # Waiting for all futures to complete
        for future_id, future in executed_futures.items():
            results[future_id] = future.result()

        self.futures.clear()  # Clearing the futures for the next batch of executions
        return results
