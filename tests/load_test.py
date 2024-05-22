# create 5 different trace names
from asyncio import gather
from langfuse.client import Langfuse
from langfuse.utils import _get_timestamp
from tests.utils import create_uuid


trace_names = [create_uuid() for _ in range(5)]

# create 20 different generation names
generation_names = [create_uuid() for _ in range(20)]

# create 2000 different user ids
user_ids = [create_uuid() for _ in range(2000)]


async def execute():
    start = _get_timestamp()

    async def update_generation(i, langfuse: Langfuse):
        trace = langfuse.trace(name=trace_names[i % 4], user_id=user_ids[i % 1999])
        # random amount of generations, 1-10
        for _ in range(i % 10):
            generation = trace.generation(name=generation_names[i % 19])
            generation.update(metadata={"count": str(i)})

    langfuse = Langfuse(debug=False, threads=100)
    print("start")
    await gather(*(update_generation(i, langfuse) for i in range(100_000)))
    print("flush")
    langfuse.flush()
    diff = _get_timestamp() - start
    print(diff)
