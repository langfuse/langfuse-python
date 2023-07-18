from langfuse.futures import FuturesStore

def test_futures_store():

    async def sample_function(x, y):
        return x + y

    async def dependent_function(prev_result, z):
        return prev_result * z

    store = FuturesStore()

    # Appending multiple functions for execution
    id1 = store.append('2', sample_function, 1, 2)  # Returns 3
    id2 = store.append('3', dependent_function, 2, future_id=id1)  # Returns 3 * 2 = 6

    # Flush and execute all appended functions
    results = store.flush()

    assert results[id1] == 3
    assert results[id2] == 6



