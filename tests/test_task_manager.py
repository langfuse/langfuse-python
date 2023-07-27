import pytest
import asyncio

from langfuse.task_manager import Task, TaskManager


@pytest.mark.asyncio
async def test_task_manager():
    async def my_task(input, multiplier):
        await asyncio.sleep(1)  # simulates IO-bound task
        return (input or 1) * multiplier

    tm = TaskManager()

    # Add tasks
    await tm.add_task(Task(1, lambda x: my_task(x, 2)))
    await tm.add_task(Task(2, lambda x: my_task(x, 3), prev_id=1))
    await tm.add_task(Task(3, lambda x: my_task(x, 4), prev_id=2))

    await tm.add_task(Task(4, lambda x: my_task(x, 4), prev_id=3))

    await tm.wait_for_completion()


@pytest.mark.asyncio
async def test_task_manager_fail():
    async def my_task(input, multiplier):
        await asyncio.sleep(1)  # simulates IO-bound task
        return (input or 1) * multiplier

    async def my_failing_task(input, multiplier):
        await asyncio.sleep(1)  # simulates IO-bound task
        raise Exception("This task failed")

    tm = TaskManager()

    # Add tasks
    await tm.add_task(Task(1, lambda x: my_task(x, 2)))
    await tm.add_task(Task(2, lambda x: my_task(x, 3), prev_id=1))
    await tm.add_task(Task(3, lambda x: my_failing_task(x, 4), prev_id=2))

    await tm.add_task(Task(4, lambda x: my_task(x, 4), prev_id=3))

    result = await tm.wait_for_completion()

    assert result["status"] == "fail"
