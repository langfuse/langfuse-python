import atexit
from enum import Enum
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
        self.lock = threading.Lock()
        self.type = TaskStatus.UNSCHEDULED


class TaskStatus(Enum):
    SUCCESS = "success"
    FAIL = "fail"
    UNSCHEDULED = "unscheduled"


class TaskManager:
    def __init__(self, num_workers, max_task_queue_size=10000):
        self.tasks = queue.Queue(max_task_queue_size)
        self.result_mapping = {}
        self.executor = ThreadPoolExecutor(num_workers)
        self.lock = threading.Lock()
        self.condition = threading.Condition(self.lock)
        self.scheduler_thread = threading.Thread(target=self._scheduler_loop)
        self.scheduler_thread.start()

        # cleans up when the python interpreter closes
        atexit.register(self.join)

    def _scheduler_loop(self):
        try:
            while True:
                task = self.tasks.get()
                if task is None:
                    logging.info("Received None, exiting scheduler loop")
                    break

                logging.info(f"Task {task.task_id} received from the queue")
                with self.lock:
                    if task.predecessor_id is not None:
                        predecessor_result = self.result_mapping.get(task.predecessor_id)
                        if predecessor_result.type == TaskStatus.UNSCHEDULED:
                            self.tasks.put(task)
                            time.sleep(0.1)
                            logging.info(f"Task {task.task_id} put back to the queue due to the unscheduled predecessor task.")
                            continue
                        elif predecessor_result.type == TaskStatus.FAIL:
                            logging.info(f"Task {task.task_id} skipped due to the failure of the predecessor task.")
                            with task.lock:
                                task.result = None
                                task.type = TaskStatus.FAIL
                            continue

                predecessor_result = self.result_mapping.get(task.predecessor_id).result if task.predecessor_id else None
                logging.info(f"Task {task.task_id} started with predecessor result {predecessor_result} from {task.predecessor_id}")
                self.executor.submit(self._execute_task, task, predecessor_result if predecessor_result else None)
                self.tasks.task_done()  # Mark the current task as done and remove it from the queue
        except Exception as e:
            logging.error(f"Exception in the scheduler loop {e}")

    def _execute_task(self, task: Task, predecessor_result: Task):
        try:
            logging.info(f"Task {task.task_id} executing")

            result = None
            with task.lock:
                try:
                    result = task.function(predecessor_result)
                    with self.lock:
                        self.result_mapping[task.task_id].result = result
                        self.result_mapping[task.task_id].type = TaskStatus.SUCCESS
                        logging.info(f"Task {task.task_id} done with result {result}")
                except Exception as e:
                    with self.lock:
                        self.result_mapping[task.task_id].result = e
                        self.result_mapping[task.task_id].type = TaskStatus.FAIL
                        logging.info(f"Task {task.task_id} failed with exception {e}")
        except Exception as e:
            logging.error(f"Exception in the task {task.task_id} {e}")

    def add_task(self, task_id, function, predecessor_id=None):
        try:
            logging.info(f"Adding task {task_id} with predecessor {predecessor_id}")
            task = Task(task_id, function, predecessor_id)
            self.tasks.put(task)
            with self.lock:
                self.result_mapping[task_id] = task
            logging.info(f"Task {task_id} added {self.result_mapping}")
        except Exception as e:
            logging.error(f"Exception in adding task {task_id} {e}")

    def join(self):
        try:
            logging.info("Joining TaskManager")

            while any(task_result.type == TaskStatus.UNSCHEDULED for task_result in self.result_mapping.values()) and not self.tasks.empty():
                logging.info(f"Waiting for all tasks to be scheduled {self.tasks.qsize()}")
                time.sleep(0.1)
                pass

            self.tasks.put(None)
            self.scheduler_thread.join()
            self.executor.shutdown(wait=True)
        except Exception as e:
            logging.error(f"Exception in joining TaskManager {e}")

    def get_task_result(self, task_id):
        with self.lock:
            if task_id not in self.result_mapping:
                raise ValueError(f"Task {task_id} does not exist")

            task_result = self.result_mapping[task_id]

        return {"result": task_result.result, "status": task_result.type}
