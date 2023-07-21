import asyncio
from langfuse.futures import FuturesStore


def test_futures_store():
    async def sample_function(x, y):
        return x + y

    async def dependent_function(prev_result, z):
        return prev_result * z

    async def second_dependent_function(prev_result, a):
        return prev_result + a

    store = FuturesStore()

    # Appending multiple functions for execution
    id1 = store.append("2", sample_function, 1, 2)  # Returns 3
    id2 = store.append("3", dependent_function, 2, future_id=id1)  # Returns 3 * 2 = 6
    store.append("4", second_dependent_function, 4, future_id=id2)  # Returns 6 + 4 = 10

    # Flush and execute all appended functions
    result = asyncio.run(store.flush())
    assert result["status"] == "success"
