import time

from langfuse.task_manager import Task, TaskManager


def test_task_manager():
    def my_task(input, multiplier):
        time.sleep(1)
        return (input or 1) * multiplier

    tm = TaskManager()

    # Add tasks
    tm.add_task(Task(1, lambda x: my_task(x, 2)))
    tm.add_task(Task(2, lambda x: my_task(x, 3), prev_id=1))
    tm.add_task(Task(3, lambda x: my_task(x, 4), prev_id=2))

    tm.add_task(Task(4, lambda x: my_task(x, 4), prev_id=3))

    tm.wait_for_completion()


def test_task_manager_fail():
    def my_task(input, multiplier):
        time.sleep(1)
        return (input or 1) * multiplier

    def my_failing_task(input, multiplier):
        time.sleep(1)
        raise Exception("This task failed")

    tm = TaskManager()

    # Add tasks
    tm.add_task(Task(1, lambda x: my_task(x, 2)))
    tm.add_task(Task(2, lambda x: my_task(x, 3), prev_id=1))
    tm.add_task(Task(3, lambda x: my_failing_task(x, 4), prev_id=2))

    tm.add_task(Task(4, lambda x: my_task(x, 4), prev_id=3))

    result = tm.wait_for_completion()
    print(result)
    assert result["status"] == "fail"
