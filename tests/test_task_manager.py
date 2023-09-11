import subprocess
import threading
import time

import pytest

from langfuse.task_manager import TaskManager, TaskStatus


@pytest.mark.timeout(10)
def test_multiple_tasks_without_predecessor():
    def task_without_predecessor():
        return 20

    tm = TaskManager()

    tm.add_task(10, task_without_predecessor)
    tm.add_task(20, task_without_predecessor)
    tm.add_task(30, task_without_predecessor)

    tm.shutdown()

    assert tm.get_result(10).result == 20
    assert tm.get_result(20).result == 20
    assert tm.get_result(30).result == 20


@pytest.mark.timeout(10)
def test_task_manager():
    def my_task():
        return 2

    def my_other_task():
        return 5

    tm = TaskManager(debug=True)

    # Add tasks
    tm.add_task(1, my_task)
    tm.add_task(2, my_task)
    tm.add_task(3, my_task)
    tm.add_task(10, my_other_task)
    tm.add_task(11, my_other_task)
    tm.add_task(12, my_other_task)

    # Check the results of the tasks
    tm.shutdown()

    assert tm.get_result(1).result == 2
    assert tm.get_result(2).result == 2
    assert tm.get_result(3).result == 2
    assert tm.get_result(10).result == 5
    assert tm.get_result(11).result == 5
    assert tm.get_result(12).result == 5


@pytest.mark.timeout(10)
def test_task_manager_fail():
    retry_count = 0

    def my_task():
        time.sleep(1)
        return 2

    def my_failing_task():
        nonlocal retry_count
        time.sleep(1)
        retry_count += 1
        raise Exception(f"This task failed {retry_count}")

    tm = TaskManager(debug=True)

    tm.add_task(1, my_task)
    tm.add_task(2, my_task)
    tm.join()
    tm.add_task(3, my_failing_task)
    tm.add_task(4, my_task)

    tm.shutdown()

    assert tm.get_result(1).result == 2
    assert tm.get_result(2).result == 2
    assert tm.get_result(3).status == TaskStatus.FAIL
    assert tm.get_result(4).status == TaskStatus.SUCCESS
    assert tm.get_result(4).result == 2
    assert retry_count == 3


# @pytest.mark.timeout(10)
# def test_task_manager_prune():
#     def first(prev_result):
#         return 2

#     def my_task(input):
#         return (input or 1) * 2

#     tm = TaskManager(max_task_age=0)

#     # Add tasks
#     tm.add_task(1, first)
#     tm.add_task(2, my_task, predecessor_id=1)
#     tm.join()

#     assert True if tm.get_result(1) is None else False
#     assert True if tm.get_result(2) is None else False


@pytest.mark.timeout(20)
def test_consumer_restart():
    def short_task():
        return 2

    tm = TaskManager(debug=True)
    tm.add_task(1, short_task)
    tm.join()

    tm.add_task(2, short_task)
    tm.shutdown()

    assert tm.get_result(2).result == 2


@pytest.mark.timeout(10)
def test_concurrent_task_additions():
    def concurrent_task():
        return 2

    def add_task_concurrently(tm, task_id, func):
        tm.add_task(task_id, func)

    tm = TaskManager(debug=True)
    threads = [threading.Thread(target=add_task_concurrently, args=(tm, i + 1, concurrent_task)) for i in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    tm.shutdown()

    for i in range(10):
        assert tm.get_result(i + 1).result == 2


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

    tm = TaskManager(debug=True)  # debug=False to avoid logging

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
    assert not tm.consumer_thread.is_alive()
