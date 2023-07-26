import pytest
import asyncio

from langfuse.task_manager import Task, TaskManager


@pytest.mark.asyncio
async def test_task_manager():
    async def my_task(input, multiplier):
        await asyncio.sleep(1)  # simulates IO-bound task
        return (input or 1) * multiplier

    tm = TaskManager()
    tm.start()

    # Add tasks
    tm.add_task(Task(1, lambda x: my_task(x, 2)))
    tm.add_task(Task(2, lambda x: my_task(x, 3), prev_id=1))
    tm.add_task(Task(3, lambda x: my_task(x, 4), prev_id=2))

    # Wait until all tasks are done
    result = await tm.await_all_tasks_done()
    assert result["status"] == "success"

    tm.add_task(Task(4, lambda x: my_task(x, 4), prev_id=3))
    result = await tm.await_all_tasks_done()

    assert result["status"] == "success"
