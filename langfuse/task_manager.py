import atexit
from datetime import datetime, timedelta
from enum import Enum
import logging
import queue
import threading
import time


class Task:
    def __init__(self, task_id, function, predecessor_id: str = None):
        self.task_id = task_id
        self.predecessor_id = predecessor_id
        self.function = function
        self.result = None
        self.timestamp = None
        self.lock = threading.Lock()
        self.status = TaskStatus.UNSCHEDULED


class TaskStatus(Enum):
    SUCCESS = "success"
    FAIL = "fail"
    UNSCHEDULED = "unscheduled"


class TaskManager:
    def __init__(self, max_task_queue_size=10000, max_task_age=600):
        self.max_task_queue_size = max_task_queue_size
        self.queue = queue.Queue(max_task_queue_size)
        self.consumer_thread = None
        self.result_mapping = {}
        self.max_task_age = max_task_age
        self.init_resources()

        # cleans up when the python interpreter closes
        atexit.register(self.join)

    def init_resources(self):
        self.consumer_thread = Consumer(self.queue, self.result_mapping, self.max_task_age)
        self.consumer_thread.start()

    def add_task(self, task_id, function, predecessor_id=None):
        try:
            logging.info(f"Adding task {task_id} with predecessor {predecessor_id}")
            if self.consumer_thread is None or not self.consumer_thread.is_alive():
                self.init_resources()
            task = Task(task_id, function, predecessor_id)

            self.queue.put(task)
            logging.info(f"Task {task_id} added to queue")
        except Exception as e:
            logging.error(f"Exception in adding task {task_id} {e}")

    def join(self):
        try:
            logging.info(f"Joining TaskManager qsize: {self.queue.qsize()}")
            self.queue.join()
            logging.info("TaskManager queue joined")
            if self.consumer_thread is not None:
                self.consumer_thread.pause()
                self.consumer_thread.join()
                self.consumer_thread = None

            logging.info("TaskManager joined")
        except Exception as e:
            logging.error(f"Exception in joining TaskManager {e}")

    def get_result(self, task_id):
        try:
            return self.result_mapping.get(task_id)
        except Exception as e:
            logging.error(f"Exception in getting result for task {task_id} {e}")


class Consumer(threading.Thread):
    def __init__(self, queue, result_mapping, max_task_age):
        """Create a consumer thread."""

        threading.Thread.__init__(self)
        # Make consumer a daemon thread so that it doesn't block program exit
        self.daemon = True
        self.queue = queue
        # It's important to set running in the constructor: if we are asked to
        # pause immediately after construction, we might set running to True in
        # run() *after* we set it to False in pause... and keep running
        # forever.
        self.running = True
        self.result_mapping = result_mapping
        self.max_task_age = max_task_age

    def run(self):
        """Runs the consumer."""
        logging.info("consumer is running...")
        while self.running:
            try:
                logging.info("consumer looping")
                self._prune_old_tasks(self.max_task_age)

                # elapsed = time.monotonic.monotonic() - start_time
                task = self.queue.get(block=True, timeout=1)

                self.result_mapping[task.task_id] = task

                logging.info(f"Task {task.task_id} received from the queue")
                if task.predecessor_id is not None:
                    predecessor_result = self.result_mapping.get(task.predecessor_id)

                    if predecessor_result is None or predecessor_result.status == TaskStatus.UNSCHEDULED:
                        self.queue.put(task)
                        time.sleep(0.2)
                        logging.info(f"Task {task.task_id} put back to the queue due to the unscheduled predecessor task.")
                        self.queue.task_done()
                        continue

                    elif predecessor_result.status == TaskStatus.FAIL:
                        logging.info(f"Task {task.task_id} skipped due to the failure or non-existence of the predecessor task.")
                        with task.lock:
                            task.result = None
                            task.status = TaskStatus.FAIL
                            task.timestamp = datetime.now()
                            self.queue.task_done()
                        continue

                predecessor_result = self.result_mapping.get(task.predecessor_id).result if task.predecessor_id else None
                logging.info(f"Task {task.task_id} started with predecessor result {predecessor_result} from {task.predecessor_id}")
                self._execute_task(task, predecessor_result if predecessor_result else None)
                logging.info(f"Task {task.task_id} done")
                self.queue.task_done()

            except queue.Empty:
                break

        logging.debug("consumer exited.")

    def pause(self):
        """Pause the consumer."""
        self.running = False

    def _execute_task(self, task: Task, predecessor_result: Task):
        try:
            logging.info(f"Task {task.task_id} executing")

            result = None
            with task.lock:
                try:
                    result = task.function(predecessor_result)
                    self.result_mapping[task.task_id].result = result
                    self.result_mapping[task.task_id].status = TaskStatus.SUCCESS
                    self.result_mapping[task.task_id].timestamp = datetime.now()
                    logging.info(f"Task {task.task_id} done with result {result}")
                except Exception as e:
                    self.result_mapping[task.task_id].result = e
                    self.result_mapping[task.task_id].status = TaskStatus.FAIL
                    self.result_mapping[task.task_id].timestamp = datetime.now()
                    logging.info(f"Task {task.task_id} failed with exception {e}")
        except Exception as e:
            logging.error(f"Exception in the task {task.task_id} {e}")

    def _prune_old_tasks(self, delta: int):
        try:
            logging.info("Pruning old tasks")
            now = datetime.now()

            to_remove = [task_id for task_id, task in self.result_mapping.items() if task.status in [TaskStatus.SUCCESS, TaskStatus.FAIL] and task.timestamp and now - task.timestamp > timedelta(seconds=delta)]
            for task_id in to_remove:
                self.result_mapping.pop(task_id, None)
                logging.info(f"Task {task_id} pruned due to age")
        except Exception as e:
            logging.error(f"Exception in pruning old tasks {e}")
