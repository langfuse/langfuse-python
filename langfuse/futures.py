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
        results = {}
        resolved_dependencies = set()
        waiting_for_resolution = set(self.futures.keys())

        # While there are futures that have not been resolved
        while waiting_for_resolution:
            for future_id in list(waiting_for_resolution):  # Iterate over a copy of the set
                future_details = self.futures[future_id]
                
                if len(future_details) == 3:  # If there are no dependencies
                    func, args, kwargs = future_details
                    future_result = asyncio.run(func(*args, **kwargs))
                    results[future_id] = future_result
                    resolved_dependencies.add(future_id)
                    waiting_for_resolution.remove(future_id)
                else:  # If there is a dependency
                    func, args, kwargs, dependent_id = future_details
                    if dependent_id in resolved_dependencies:  # If dependency is resolved
                        dependent_result = results[dependent_id]  # get the result from the parent future
                        future_result = asyncio.run(func(dependent_result, *args, **kwargs))
                        results[future_id] = future_result
                        resolved_dependencies.add(future_id)
                        waiting_for_resolution.remove(future_id)
        
        self.futures.clear()  # Clearing the futures for the next batch of executions
        return results


