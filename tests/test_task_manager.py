import logging
import subprocess
import threading
import time

import pytest

from langfuse.task_manager import TaskManager


@pytest.mark.asyncio
async def test_multiple_tasks_without_predecessor():
    counter = 0

    async def task_without_predecessor():
        nonlocal counter
        counter = counter + 1

    tm = await TaskManager.create(debug=True)

    tm.add_task(10, task_without_predecessor)
    tm.add_task(20, task_without_predecessor)
    tm.add_task(30, task_without_predecessor)

    print("Waiting for tasks to finish")
    await tm.flush()
    print("Flushed")
    await tm.join()
    print("Joined")
    assert counter == 3


@pytest.mark.asyncio
async def test_task_manager_fail():
    retry_count = 0
    counter = 0

    async def my_task():
        nonlocal counter
        counter = counter + 1
        time.sleep(1)

    async def my_failing_task():
        nonlocal retry_count
        nonlocal counter
        time.sleep(1)
        retry_count += 1
        raise Exception(f"This task failed {retry_count}")

    tm = await TaskManager.create(debug=True)

    tm.add_task(1, my_task)
    tm.add_task(2, my_task)
    await tm.flush()

    tm.add_task(3, my_failing_task)
    tm.add_task(4, my_task)

    await tm.flush()
    await tm.join()

    assert counter == 3
    assert retry_count == 3


@pytest.mark.asyncio
async def test_consumer_restart():
    counter = 0

    async def short_task():
        nonlocal counter
        counter = counter + 1

    tm = await TaskManager.create(debug=True)
    tm.add_task(1, short_task)
    await tm.flush()

    tm.add_task(2, short_task)
    await tm.shutdown()

    assert counter == 2


@pytest.mark.asyncio
async def test_concurrent_task_additions():
    counter = 0

    async def concurrent_task():
        nonlocal counter
        counter = counter + 1

    def add_task_concurrently(tm, task_id, func):
        tm.add_task(task_id, func)

    tm = await TaskManager.create(debug=True)
    threads = [threading.Thread(target=add_task_concurrently, args=(tm, i + 1, concurrent_task)) for i in range(50)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    await tm.shutdown()

    logging.warning(counter)
    assert counter == 50


@pytest.mark.timeout(10)
def test_atexit():
    python_code = """
import time
import logging
import asyncio
from langfuse.task_manager import TaskManager

async def dummy_function():
    logging.info("dummy_function")
    await asyncio.sleep(0.5)
    return 42

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
print("Adding task manager", TaskManager)
loop = asyncio.get_event_loop()
try:
    manager = loop.run_until_complete(TaskManager.create(debug=True))
    manager.add_task(1, dummy_function)
    manager.add_task(2, dummy_function)
finally:
    loop.close()

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

    assert "successfully joined all consumers" in logs


@pytest.mark.asyncio
async def test_flush():
    # set up the consumer with more requests than a single batch will allow
    async def short_task():
        return 2

    tm = await TaskManager.create(debug=False)  # debug=False to avoid logging

    for i in range(1000):
        tm.add_task(i, short_task)
    # We can't reliably assert that the queue is non-empty here; that's
    # a race condition. We do our best to load it up though.
    await tm.flush()
    # Make sure that the client queue is empty after flushing
    assert tm.queue.empty()


@pytest.mark.asyncio
async def test_shutdown():
    # set up the consumer with more requests than a single batch will allow
    async def short_task():
        return 2

    tm = await TaskManager.create(debug=False)  # debug=False to avoid logging

    for i in range(1000):
        tm.add_task(i, short_task)

    await tm.shutdown()
    # we expect two things after shutdown:
    # 1. client queue is empty
    # 2. consumer thread has stopped
    assert tm.queue.empty()

    for c in tm.consumers:
        assert not c.running
