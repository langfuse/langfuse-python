import logging
import subprocess
import time

import pytest

from langfuse.task_manager import TaskManager, TaskStatus


@pytest.mark.timeout(10)
def test_task_manager():
    def my_task(prev_result):
        logging.info(f"my_task {prev_result}, returning {(prev_result or 1) * 2}")
        return (prev_result or 1) * 2

    def my_other_task(prev_result):
        logging.info(f"my_other_task {prev_result}")
        return (prev_result or 1) * 5

    tm = TaskManager()

    # Add tasks
    tm.add_task(1, my_task)
    tm.add_task(10, my_other_task)
    tm.add_task(2, my_task, predecessor_id=1)
    tm.add_task(11, my_other_task, predecessor_id=10)
    tm.add_task(3, my_task, predecessor_id=2)
    tm.add_task(12, my_other_task, predecessor_id=11)

    # Check the results of the tasks

    tm.join()

    assert tm.get_result(1).result == 2
    assert tm.get_result(2).result == 4
    assert tm.get_result(3).result == 8
    assert tm.get_result(10).result == 5
    assert tm.get_result(11).result == 25
    assert tm.get_result(12).result == 125


@pytest.mark.timeout(10)
def test_task_manager_fail():
    def first(prev_result):
        return 2

    def my_task(input):
        time.sleep(1)
        return (input or 1) * 2

    def my_failing_task(input):
        time.sleep(1)
        raise Exception("This task failed")

    tm = TaskManager()

    # Add tasks
    tm.add_task(1, first)
    tm.add_task(2, my_task, predecessor_id=1)
    tm.join()
    tm.add_task(3, my_failing_task, predecessor_id=2)
    tm.add_task(4, my_task, predecessor_id=3)

    tm.join()

    assert tm.get_result(1).result == 2
    assert tm.get_result(2).result == 4
    assert tm.get_result(3).status == TaskStatus.FAIL
    assert tm.get_result(4).status == TaskStatus.FAIL


@pytest.mark.timeout(10)
def test_atexit():
    python_code = """
import time
import logging
from langfuse.task_manager import TaskManager  # assuming task_manager is the module name

def dummy_function(result):
    logging.info(f"dummy_function {result}")
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
manager = TaskManager()
a = manager.add_task(1, dummy_function)
print(a)
manager.add_task(2, dummy_function, predecessor_id=1)

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

    assert "TaskManager joined" in logs
