from urllib.parse import urlparse, urlunparse
from langfuse.request import LangfuseClient
import subprocess
import threading
from werkzeug.wrappers import Request, Response

import pytest

from langfuse.task_manager import TaskManager


def setup_server(httpserver, expected_body: dict):
    httpserver.expect_request("/api/public/ingestion", method="POST", json=expected_body).respond_with_data("success")


def setup_langfuse_client(server: str):
    return LangfuseClient("public_key", "secret_key", server, "1.0.0")


def get_host(url):
    parsed_url = urlparse(url)
    new_url = urlunparse((parsed_url.scheme, parsed_url.netloc, "", "", "", ""))
    return new_url


@pytest.mark.timeout(10)
def test_multiple_tasks_without_predecessor(httpserver):
    setup_server(httpserver, {"batch": [{"foo": "bar"}, {"foo_1": "bar_1"}, {"foo_2": "bar_2"}]})

    langfuse_client = setup_langfuse_client(get_host(httpserver.url_for("/api/public/ingestion")))

    tm = TaskManager(langfuse_client, debug=True)

    tm.add_task(10, {"foo": "bar"})
    tm.add_task(20, {"foo_1": "bar_1"})
    tm.add_task(30, {"foo_2": "bar_2"})

    tm.flush()


@pytest.mark.timeout(10)
def test_task_manager_fail(httpserver):
    count = 0

    def handler(request: Request):
        nonlocal count
        count = count + 1
        return Response("failure", status=500)

    httpserver.expect_request("/api/public/ingestion", method="POST", json={"batch": [{"foo": "bar"}, {"foo": "bar"}]}).respond_with_handler(handler)

    langfuse_client = setup_langfuse_client(get_host(httpserver.url_for("/api/public/ingestion")))

    tm = TaskManager(langfuse_client, debug=True)

    tm.add_task(1, {"foo": "bar"})
    tm.add_task(2, {"foo": "bar"})
    tm.flush()

    assert count == 3


@pytest.mark.timeout(20)
def test_consumer_restart(httpserver):
    httpserver.expect_ordered_request("/api/public/ingestion", method="POST", json={"batch": [{"foo": "bar"}]}).respond_with_data("success")
    httpserver.expect_ordered_request("/api/public/ingestion", method="POST", json={"batch": [{"foo_1": "bar_1"}]}).respond_with_data("success")

    langfuse_client = setup_langfuse_client(get_host(httpserver.url_for("/api/public/ingestion")))

    tm = TaskManager(langfuse_client, debug=True)

    tm.add_task(1, {"foo": "bar"})
    tm.flush()

    tm.add_task(2, {"foo_1": "bar_1"})
    tm.flush()


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

    assert "consumer thread 0 joined" in logs


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

    tm = TaskManager(debug=False, number_of_consumers=5)  # debug=False to avoid logging

    for i in range(1000):
        tm.add_task(i, short_task)

    tm.shutdown()
    # we expect two things after shutdown:
    # 1. client queue is empty
    # 2. consumer thread has stopped
    assert tm.queue.empty()

    assert len(tm.consumers) == 5
    for c in tm.consumers:
        assert not c.is_alive()
    assert tm.queue.empty()
