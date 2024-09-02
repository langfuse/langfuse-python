import pytest
import subprocess


@pytest.mark.timeout(10)
def test_prompts_atexit():
    python_code = """
import time
import logging
from langfuse.prompt_cache import PromptCache  # assuming task_manager is the module name

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)

print("Adding prompt cache", PromptCache)
prompt_cache = PromptCache(max_prompt_refresh_workers=10)

# example task that takes 2 seconds but we will force it to exit earlier
def wait_2_sec():
    time.sleep(2)

# 8 times
for i in range(8):
    prompt_cache.refresh_prompt(f"key_wait_2_sec_i_{i}", lambda: wait_2_sec())
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

    assert "Shutdown of prompt refresh executor completed." in logs


@pytest.mark.timeout(10)
def test_prompts_atexit_async():
    python_code = """
import time
import asyncio
import logging
from langfuse.prompt_cache import PromptCache  # assuming task_manager is the module name

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)

async def main():
    print("Adding prompt cache", PromptCache)
    prompt_cache = PromptCache(max_prompt_refresh_workers=10)

    # example task that takes 2 seconds but we will force it to exit earlier
    def wait_2_sec():
        time.sleep(2)

    async def add_new_prompt_refresh(i: int):
        prompt_cache.refresh_prompt(f"key_wait_2_sec_i_{i}", lambda: wait_2_sec())
    
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

    shutdown_count = logs.count("Shutdown of prompt refresh executor completed.")
    assert (
        shutdown_count == 3
    ), f"Expected 3 shutdown messages, but found {shutdown_count}"
