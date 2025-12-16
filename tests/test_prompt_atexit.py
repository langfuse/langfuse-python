import subprocess

import pytest


@pytest.mark.timeout(10)
def test_prompts_atexit():
    python_code = """
import time
import logging

from langfuse.logger import langfuse_logger
from langfuse._utils.prompt_cache import PromptCache

langfuse_logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
langfuse_logger.addHandler(handler)

print("Adding prompt cache", PromptCache)
prompt_cache = PromptCache(max_prompt_refresh_workers=10)

# example task that takes 2 seconds but we will force it to exit earlier
def wait_2_sec():
    time.sleep(2)

# 8 times
for i in range(8):
    prompt_cache.add_refresh_prompt_task(f"key_wait_2_sec_i_{i}", lambda: wait_2_sec())
"""

    process = subprocess.Popen(
        ["python", "-c", python_code], stderr=subprocess.PIPE, text=True
    )

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

    shutdown_count = logs.count("Shutdown of prompt refresh task manager completed.")
    assert shutdown_count == 1, (
        f"Expected 1 shutdown messages, but found {shutdown_count}"
    )


@pytest.mark.timeout(10)
def test_prompts_atexit_async():
    python_code = """
import time
import asyncio
import logging

from langfuse.logger import langfuse_logger
from langfuse._utils.prompt_cache import PromptCache

langfuse_logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
langfuse_logger.addHandler(handler)

async def main():
    print("Adding prompt cache", PromptCache)
    prompt_cache = PromptCache(max_prompt_refresh_workers=10)

    # example task that takes 2 seconds but we will force it to exit earlier
    def wait_2_sec():
        time.sleep(2)

    async def add_new_prompt_refresh(i: int):
        prompt_cache.add_refresh_prompt_task(f"key_wait_2_sec_i_{i}", lambda: wait_2_sec())
    
    # 8 times
    tasks = [add_new_prompt_refresh(i) for i in range(8)]
    await asyncio.gather(*tasks)

async def run_multiple_mains():
    main_tasks = [main() for _ in range(3)]
    await asyncio.gather(*main_tasks)

if __name__ == "__main__":
    asyncio.run(run_multiple_mains())
"""

    process = subprocess.Popen(
        ["python", "-c", python_code], stderr=subprocess.PIPE, text=True
    )

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

    shutdown_count = logs.count("Shutdown of prompt refresh task manager completed.")
    assert shutdown_count == 3, (
        f"Expected 3 shutdown messages, but found {shutdown_count}"
    )
