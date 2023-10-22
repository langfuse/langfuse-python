#!/usr/bin/env python

import time
from langfuse.client import Langfuse
from langfuse.model import InitialGeneration, UpdateGeneration
import asyncio
import cProfile


def start():
    import asyncio

    cProfile.run("asyncio.run(lets_go())")
    # asyncio.run(lets_go())


async def update_generation(i, langfuse: Langfuse):
    # api = get_api()
    print(f"update {i}")
    generation = langfuse.generation(InitialGeneration(name="1-B"))
    generation.update(UpdateGeneration(metadata={"dict": "value"}))
    print(f"updated {i}")


async def lets_go():
    start = time.time()

    langfuse = Langfuse(debug=False)

    await asyncio.gather(*(update_generation(i, langfuse) for i in range(1000)))
    # pool.close()
    # pool.join()
    print(langfuse.task_manager.queue.qsize())
    print("done")

    langfuse.flush()
    print(f"flushed {str(langfuse.task_manager.queue.qsize())}")

    for x in langfuse.task_manager.consumers:
        print(f"{x.identifier}, {x.running}")

    langfuse.join()
    for x in langfuse.task_manager.consumers:
        print(f"{x.identifier}, {x.running}")

    print("joined")
    end = time.time()
    time_in_seconds = end - start
    print(f"Time taken: {time_in_seconds} seconds")

    print("shutdown")
    print(f"flushed {str(langfuse.task_manager.queue.qsize())}")
