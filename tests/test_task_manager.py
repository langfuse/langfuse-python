import subprocess
import threading
import time

import pytest

from langfuse.task_manager import TaskManager, TaskStatus


@pytest.mark.timeout(10)
def test_multiple_tasks_without_predecessor():
    counter = 0

    def task_without_predecessor():
        nonlocal counter
        counter = counter + 1

    tm = TaskManager()

    tm.add_task(10, task_without_predecessor)
    tm.add_task(20, task_without_predecessor)
    tm.add_task(30, task_without_predecessor)

    tm.shutdown()
    assert counter == 3


@pytest.mark.timeout(10)
def test_task_manager_fail():
    retry_count = 0
    counter = 0

    def my_task():
        nonlocal counter
        counter = counter + 1
        time.sleep(1)

    def my_failing_task():
        nonlocal retry_count
        nonlocal counter
        time.sleep(1)
        retry_count += 1
        raise Exception(f"This task failed {retry_count}")

    tm = TaskManager(debug=False)

    tm.add_task(1, my_task)
    tm.add_task(2, my_task)
    tm.join()
    tm.add_task(3, my_failing_task)
    tm.add_task(4, my_task)

    tm.shutdown()

    assert counter == 3
    assert retry_count == 3


@pytest.mark.timeout(20)
def test_consumer_restart():
    counter = 0

    def short_task():
        nonlocal counter
        counter = counter + 1

    tm = TaskManager(debug=False)
    tm.add_task(1, short_task)
    tm.join()

    tm.add_task(2, short_task)
    tm.shutdown()

    assert counter == 2


@pytest.mark.timeout(10)
def test_concurrent_task_additions():
    counter = 0

    def concurrent_task():
        nonlocal counter
        counter = counter + 1

    def add_task_concurrently(tm, task_id, func):
        tm.add_task(task_id, func)

    tm = TaskManager(debug=False)
    threads = [threading.Thread(target=add_task_concurrently, args=(tm, i + 1, concurrent_task)) for i in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    tm.shutdown()

    assert counter == 10


@pytest.mark.timeout(10)
def test_atexit():
    python_code = """
import time
import logging
from langfuse.task_manager import TaskManager  # assuming task_manager is the module name

def dummy_function():
    logging.info("dummy_function")
    time.sleep(0.5)
    return 42

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
print("Adding task manager", TaskManager)
manager = TaskManager(debug=True)
a = manager.add_task(1, dummy_function)
manager.add_task(2, dummy_function)

"""

    process = subprocess.Popen(["python", "-c", python_code], stderr=subprocess.PIPE, text=True)

    logs = ""

    try:
        for line in process.stderr:
            logs += line.strip()
            print(line.strip())
    except subprocess.TimeoutExpired:
        pytest.fail("The process took too long to execute")
    process.communicate()

    returncode = process.returncode
    if returncode != 0:
        pytest.fail("Process returned with error code")

    print(process.stderr)

    assert "consumer thread joined" in logs


def test_flush():
    # set up the consumer with more requests than a single batch will allow
    def short_task():
        return 2

    tm = TaskManager(debug=False)  # debug=False to avoid logging

    for i in range(1000):
        tm.add_task(i, short_task)
    # We can't reliably assert that the queue is non-empty here; that's
    # a race condition. We do our best to load it up though.
    tm.flush()
    # Make sure that the client queue is empty after flushing
    assert tm.queue.empty()


def test_shutdown():
    # set up the consumer with more requests than a single batch will allow
    def short_task():
        return 2

    tm = TaskManager(debug=False)  # debug=False to avoid logging

    for i in range(1000):
        tm.add_task(i, short_task)

    tm.shutdown()
    # we expect two things after shutdown:
    # 1. client queue is empty
    # 2. consumer thread has stopped
    assert tm.queue.empty()

    for c in tm.consumers:
        assert not c.is_alive()
