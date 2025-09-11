"""Test suite for utility functions in langfuse._client.utils module."""

import asyncio
import threading
from unittest import mock

import pytest

from langfuse._client.utils import run_async_safely


class TestRunAsyncSafely:
    """Test suite for the run_async_safely function."""

    def test_run_sync_context_simple(self):
        """Test run_async_safely in sync context with simple coroutine."""

        async def simple_coro():
            await asyncio.sleep(0.01)
            return "hello"

        result = run_async_safely(simple_coro())
        assert result == "hello"

    def test_run_sync_context_with_value(self):
        """Test run_async_safely in sync context with parameter passing."""

        async def coro_with_params(value, multiplier=2):
            await asyncio.sleep(0.01)
            return value * multiplier

        result = run_async_safely(coro_with_params(5, multiplier=3))
        assert result == 15

    def test_run_sync_context_with_exception(self):
        """Test run_async_safely properly propagates exceptions in sync context."""

        async def failing_coro():
            await asyncio.sleep(0.01)
            raise ValueError("Test error")

        with pytest.raises(ValueError, match="Test error"):
            run_async_safely(failing_coro())

    @pytest.mark.asyncio
    async def test_run_async_context_simple(self):
        """Test run_async_safely from within async context (uses threading)."""

        async def simple_coro():
            await asyncio.sleep(0.01)
            return "from_thread"

        # This should use threading since we're already in an async context
        result = run_async_safely(simple_coro())
        assert result == "from_thread"

    @pytest.mark.asyncio
    async def test_run_async_context_with_exception(self):
        """Test run_async_safely properly propagates exceptions from thread."""

        async def failing_coro():
            await asyncio.sleep(0.01)
            raise RuntimeError("Thread error")

        with pytest.raises(RuntimeError, match="Thread error"):
            run_async_safely(failing_coro())

    @pytest.mark.asyncio
    async def test_run_async_context_thread_isolation(self):
        """Test that threaded execution is properly isolated."""
        # Set a thread-local value in the main async context
        threading.current_thread().test_value = "main_thread"

        async def check_thread_isolation():
            # This should run in a different thread
            current_thread = threading.current_thread()
            # Should not have the test_value from main thread
            assert not hasattr(current_thread, "test_value")
            return "isolated"

        result = run_async_safely(check_thread_isolation())
        assert result == "isolated"

    def test_multiple_calls_sync_context(self):
        """Test multiple sequential calls in sync context."""

        async def counter_coro(count):
            await asyncio.sleep(0.001)
            return count * 2

        results = []
        for i in range(5):
            result = run_async_safely(counter_coro(i))
            results.append(result)

        assert results == [0, 2, 4, 6, 8]

    @pytest.mark.asyncio
    async def test_multiple_calls_async_context(self):
        """Test multiple sequential calls in async context (each uses threading)."""

        async def counter_coro(count):
            await asyncio.sleep(0.001)
            return count * 3

        results = []
        for i in range(3):
            result = run_async_safely(counter_coro(i))
            results.append(result)

        assert results == [0, 3, 6]

    def test_concurrent_calls_sync_context(self):
        """Test concurrent calls in sync context using threading."""

        async def slow_coro(value):
            await asyncio.sleep(0.02)
            return value**2

        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = []
            for i in range(3):
                future = executor.submit(run_async_safely, slow_coro(i + 1))
                futures.append(future)

            results = [future.result() for future in futures]

        # Results should be squares: 1^2, 2^2, 3^2
        assert sorted(results) == [1, 4, 9]

    def test_event_loop_detection_mock(self):
        """Test event loop detection logic with mocking."""

        async def simple_coro():
            return "mocked"

        # Mock no running loop - should use asyncio.run
        with mock.patch(
            "asyncio.get_running_loop", side_effect=RuntimeError("No loop")
        ):
            with mock.patch(
                "asyncio.run", return_value="asyncio_run_called"
            ) as mock_run:
                result = run_async_safely(simple_coro())
                assert result == "asyncio_run_called"
                mock_run.assert_called_once()

    def test_complex_coroutine(self):
        """Test with a more complex coroutine that does actual async work."""

        async def complex_coro():
            # Simulate some async operations
            results = []
            for i in range(3):
                await asyncio.sleep(0.001)
                results.append(i**2)

            # Simulate concurrent operations
            async def sub_task(x):
                await asyncio.sleep(0.001)
                return x * 10

            tasks = [sub_task(x) for x in range(2)]
            concurrent_results = await asyncio.gather(*tasks)
            results.extend(concurrent_results)

            return results

        result = run_async_safely(complex_coro())
        assert result == [0, 1, 4, 0, 10]  # [0^2, 1^2, 2^2, 0*10, 1*10]

    @pytest.mark.asyncio
    async def test_nested_async_calls(self):
        """Test that nested calls to run_async_safely work correctly."""

        async def inner_coro(value):
            await asyncio.sleep(0.001)
            return value * 2

        async def outer_coro(value):
            # This is already in an async context, so the inner call
            # will also use threading
            inner_result = run_async_safely(inner_coro(value))
            await asyncio.sleep(0.001)
            return inner_result + 1

        result = run_async_safely(outer_coro(5))
        assert result == 11  # (5 * 2) + 1

    def test_exception_types_preserved(self):
        """Test that different exception types are properly preserved."""

        async def custom_exception_coro():
            await asyncio.sleep(0.001)

            class CustomError(Exception):
                pass

            raise CustomError("Custom error message")

        with pytest.raises(Exception) as exc_info:
            run_async_safely(custom_exception_coro())

        # The exception type should be preserved
        assert "Custom error message" in str(exc_info.value)

    def test_return_types_preserved(self):
        """Test that various return types are properly preserved."""

        async def dict_coro():
            await asyncio.sleep(0.001)
            return {"key": "value", "number": 42}

        async def list_coro():
            await asyncio.sleep(0.001)
            return [1, 2, 3, "string"]

        async def none_coro():
            await asyncio.sleep(0.001)
            return None

        dict_result = run_async_safely(dict_coro())
        assert dict_result == {"key": "value", "number": 42}
        assert isinstance(dict_result, dict)

        list_result = run_async_safely(list_coro())
        assert list_result == [1, 2, 3, "string"]
        assert isinstance(list_result, list)

        none_result = run_async_safely(none_coro())
        assert none_result is None

    @pytest.mark.asyncio
    async def test_real_world_scenario_jupyter_simulation(self):
        """Test scenario simulating Jupyter notebook environment."""
        # This simulates being called from a Jupyter cell where there's
        # already an event loop running

        async def simulate_llm_call(prompt):
            """Simulate an LLM API call."""
            await asyncio.sleep(0.01)  # Simulate network delay
            return f"Response to: {prompt}"

        async def simulate_experiment_task(item):
            """Simulate an experiment task function."""
            response = await simulate_llm_call(item["input"])
            await asyncio.sleep(0.001)  # Additional processing
            return response

        # This should work even though we're in an async context
        result = run_async_safely(simulate_experiment_task({"input": "test prompt"}))
        assert result == "Response to: test prompt"
