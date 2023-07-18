import concurrent.futures
import asyncio

class FuturesStore:
    def __init__(self):
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=10)
        self.futures = {}

    def append(self, func, *args, future_id=None, **kwargs):
        new_future_id = id(func)  # generate a unique id for the future

        if future_id is None:
            # If no future_id is provided, submit the function for execution without any dependencies
            future = self.executor.submit(asyncio.run, func(*args, **kwargs))
        else:
            # If a future_id is provided, create a dependent future
            dependent_future = concurrent.futures.Future()

            def callback(future):
                result = future.result()  # get the result from the parent future
                dependent_result = asyncio.run(func(result, *args, **kwargs))
                dependent_future.set_result(dependent_result)

            self.futures[future_id].add_done_callback(callback)
            future = dependent_future

        self.futures[new_future_id] = future
        return new_future_id

    def flush(self):
        results = {}
        for future in concurrent.futures.as_completed(self.futures.values()):
            for future_id, future_obj in self.futures.items():
                if future == future_obj:
                    results[future_id] = future.result()
                    break
        self.futures.clear()
        return results


