import threading
import queue
import logging


class Task:
    def __init__(self, id, func, prev_id=None):
        self.id = id
        self.func = func
        self.prev_id = prev_id


class TaskManager:
    def __init__(self):
        self.tasks = queue.Queue()
        self.result_mapping = {}
        self.processing = True
        self.thread = threading.Thread(target=self.process_tasks, daemon=True)
        self.lock = threading.Lock()  # Lock for synchronizing add_task and wait_for_completion

    def process_tasks(self):
        logging.info("Processing tasks...")
        while self.processing:
            task = self.tasks.get()
            logging.info(f"Processing task {task.id} with dependency {task.prev_id}...")

            # Wait for the task's dependency to complete
            if task.prev_id is not None:
                prev_task_result = self.result_mapping.get(task.prev_id)
                while prev_task_result is None:
                    prev_task_result = self.result_mapping.get(task.prev_id)  # Spin until the dependency is completed
            else:
                prev_task_result = None

            logging.info(f"Task {task.id} dependency completed with result {prev_task_result}")
            try:
                result = task.func(prev_task_result)
                self.result_mapping[task.id] = result  # Map task id to its result
                logging.info(f"Task {task.id} completed successfully with result {result}")
            except Exception as e:
                logging.error(f"Task {task.id} failed with exception {e}")
                self.result_mapping[task.id] = e  # Map task id to its exception
                self.processing = False  # Stop processing tasks after the first failure
            self.tasks.task_done()
            if self.tasks.empty():
                logging.info("No more tasks to process")
                self.processing = False

    def add_task(self, task):
        with self.lock:
            logging.info(f"Adding task {task.id} with dependency {task.prev_id}...")
            if self.thread is not None and not self.thread.is_alive():
                logging.info("Restarting task processing...")
                self.processing = True
                self.thread = threading.Thread(target=self.process_tasks, daemon=True)
                self.thread.start()
            self.tasks.put(task)

    def wait_for_completion(self):
        with self.lock:
            logging.info("Waiting for task processing to complete...")
            while self.processing:
                pass  # Spin until task processing is complete
            if self.thread is not None:
                self.thread.join()
            self.thread = None

            failed_tasks = []
            for task_id, result in self.result_mapping.items():
                if isinstance(result, Exception):  # The task failed
                    logging.error(f"Task {task_id} failed with exception {result}")
                    failed_tasks.append(task_id)
            logging.info(f"Task manager result mapping {self.result_mapping.keys()}, {self.tasks.qsize()}")

            if failed_tasks:  # If any tasks failed, return a failure status
                return {"status": "fail", "failed_tasks": failed_tasks}
            else:  # All tasks succeeded
                return {"status": "success"}

    def __del__(self):
        if self.thread is not None:
            self.thread.join()
            self.thread = None
