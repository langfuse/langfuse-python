import atexit
import logging
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
import time


class Task:
    def __init__(self, task_id, function, predecessor_id: str = None):
        self.task_id = task_id
        self.predecessor_id = predecessor_id
        self.function = function
        self.result = None


class TaskManager:
    def __init__(self, num_workers, max_task_queue_size=10000):
        self.tasks = queue.Queue(max_task_queue_size)
        self.result_mapping = {}
        self.executor = ThreadPoolExecutor(num_workers)
        self.lock = threading.Lock()
        self.scheduler_thread = threading.Thread(target=self._scheduler_loop)
        self.scheduler_thread.start()

        atexit.register(self.clean_up)

    def _scheduler_loop(self):
        while True:
            task = self.tasks.get()
            if task is None:
                break

            with self.lock:
                if task.predecessor_id is not None and self.result_mapping.get(task.predecessor_id) is None:
                    self.tasks.put(task)
                    time.sleep(0.1)
                    continue

            predecessor_result = self.result_mapping.get(task.predecessor_id) if task.predecessor_id else None
            logging.info(f"Task {task.task_id} started with predecessor result {predecessor_result} from {task.predecessor_id}")
            future = self.executor.submit(task.function, predecessor_result)
            future.result()  # <-- Wait for the task to finish
            future.add_done_callback(lambda future: self._task_done_callback(task, future))

            logging.info(f"Task {task.task_id} done")

    def _task_done_callback(self, task, future):
        logging.info(f"Task {task.task_id} done callback started")
        with self.lock:
            logging.info(f"Task {task.task_id} lock acquired")
            exception = future.exception()

            if exception is None:
                task.result = future.result()
            else:
                task.result = exception

            self.result_mapping[task.task_id] = task.result
            self.tasks.task_done()
            logging.info(f"Task {task.task_id} done callback done with result {task.result}, {self.result_mapping}")

    def add_task(self, task_id, function, predecessor_id=None):
        logging.info(f"Adding task {task_id} with predecessor {predecessor_id}")
        task = Task(task_id, function, predecessor_id)
        self.tasks.put(task)
        with self.lock:
            self.result_mapping[task_id] = None
        logging.info(f"Task {task_id} added {self.result_mapping}")

    def clean_up(self):
        self.tasks.put(None)
        self.scheduler_thread.join()
        self.executor.shutdown(wait=True)

    def join(self):
        self.clean_up()

    def get_task_result(self, task_id):
        with self.lock:
            if task_id not in self.result_mapping:
                raise ValueError(f"Task {task_id} does not exist")

            task_result = self.result_mapping[task_id]
            if isinstance(task_result, Exception):
                raise task_result

        return task_result
