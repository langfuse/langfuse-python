import asyncio
import logging
from functools import wraps
from typing import (
    Any,
    Callable,
    Literal,
    Optional,
    TypeVar,
    Union,
    cast,
    overload,
)

from typing_extensions import ParamSpec

from langfuse.otel._get_client import get_client
from langfuse.types import TraceContext

F = TypeVar("F", bound=Callable[..., Any])
P = ParamSpec("P")
R = TypeVar("R")


class LangfuseDecorator:
    """Implementation of the @observe decorator for seamless Langfuse tracing integration.

    This class provides the core functionality for the @observe decorator, which enables
    automatic tracing of functions and methods in your application with Langfuse.
    It handles both synchronous and asynchronous functions, maintains proper trace context,
    and intelligently routes to the correct Langfuse client instance.

    The implementation follows a singleton pattern where a single decorator instance
    handles all @observe decorations throughout the application codebase.

    Features:
    - Automatic span creation and management for both sync and async functions
    - Proper trace context propagation between decorated functions
    - Specialized handling for LLM-related spans with the 'as_type="generation"' parameter
    - Type-safe decoration that preserves function signatures and type hints
    - Support for explicit trace and parent span ID specification
    - Thread-safe client resolution when multiple Langfuse projects are used
    """

    _log = logging.getLogger("langfuse")

    # Type overload for observe decorator with no arguments
    @overload
    def observe(self, func: F) -> F: ...

    # Type overload for observe decorator with arguments
    @overload
    def observe(
        self,
        func: None = None,
        *,
        name: Optional[str] = None,
        as_type: Optional[Literal["generation"]] = None,
        # TODO: add back IO capture
    ) -> Callable[[F], F]: ...

    # Implementation of observe decorator
    def observe(
        self,
        func: Optional[F] = None,
        *,
        name: Optional[str] = None,
        as_type: Optional[Literal["generation"]] = None,
    ) -> Union[F, Callable[[F], F]]:
        """Wrap a function to create and manage Langfuse tracing around its execution, supporting both synchronous and asynchronous functions.

        This decorator provides seamless integration of Langfuse observability into your codebase. It automatically creates
        spans or generations around function execution, capturing timing, inputs/outputs, and error states. The decorator
        intelligently handles both synchronous and asynchronous functions, preserving function signatures and type hints.

        Using OpenTelemetry's distributed tracing system, it maintains proper trace context propagation throughout your application,
        enabling you to see hierarchical traces of function calls with detailed performance metrics and function-specific details.

        Args:
            func (Optional[Callable]): The function to decorate. When used with parentheses @observe(), this will be None.
            name (Optional[str]): Custom name for the created trace or span. If not provided, the function name is used.
            as_type (Optional[Literal["generation"]]): Set to "generation" to create a specialized LLM generation span
                    with model metrics support, suitable for tracking language model outputs.

        Returns:
            Callable: A wrapped version of the original function that automatically creates and manages Langfuse spans.

        Example:
            For general function tracing with automatic naming:
            ```python
            @observe()
            def process_user_request(user_id, query):
                # Function is automatically traced with name "process_user_request"
                return get_response(query)
            ```

            For language model generation tracking:
            ```python
            @observe(name="answer-generation", as_type="generation")
            async def generate_answer(query):
                # Creates a generation-type span with extended LLM metrics
                response = await openai.chat.completions.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": query}]
                )
                return response.choices[0].message.content
            ```

            For trace context propagation between functions:
            ```python
            @observe()
            def main_process():
                # Parent span is created
                return sub_process()  # Child span automatically connected to parent

            @observe()
            def sub_process():
                # Automatically becomes a child span of main_process
                return "result"
            ```

        Raises:
            Exception: Propagates any exceptions from the wrapped function after logging them in the trace.

        Notes:
            - The decorator preserves the original function's signature, docstring, and return type.
            - Proper parent-child relationships between spans are automatically maintained.
            - Special keyword arguments can be passed to control tracing:
              - langfuse_trace_id: Explicitly set the trace ID for this function call
              - langfuse_parent_observation_id: Explicitly set the parent span ID
              - langfuse_public_key: Use a specific Langfuse project (when multiple clients exist)
            - For async functions, the decorator returns an async function wrapper.
            - For sync functions, the decorator returns a synchronous wrapper.
        """

        def decorator(func: F) -> F:
            return (
                self._async_observe(
                    func,
                    name=name,
                    as_type=as_type,
                )
                if asyncio.iscoroutinefunction(func)
                else self._sync_observe(
                    func,
                    name=name,
                    as_type=as_type,
                )
            )

        """Handle decorator with or without parentheses.
        
        This logic enables the decorator to work both with and without parentheses:
        - @observe - Python passes the function directly to the decorator
        - @observe() - Python calls the decorator first, which must return a function decorator
        
        When called without arguments (@observe), the func parameter contains the function to decorate,
        so we directly apply the decorator to it. When called with parentheses (@observe()), 
        func is None, so we return the decorator function itself for Python to apply in the next step.
        """
        if func is None:
            return decorator
        else:
            return decorator(func)

    def _async_observe(
        self,
        func: F,
        *,
        name: Optional[str],
        as_type: Optional[Literal["generation"]],
    ) -> F:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            trace_id = kwargs.pop("langfuse_trace_id", None)
            parent_observation_id = kwargs.pop("langfuse_parent_observation_id", None)
            trace_context: Optional[TraceContext] = (
                {
                    "trace_id": trace_id,
                    "parent_span_id": parent_observation_id,
                }
                if trace_id
                else None
            )
            final_name = name or func.__name__
            public_key = kwargs.pop("langfuse_public_key", None)
            langfuse_client = get_client(public_key=public_key)
            context_manager = (
                (
                    langfuse_client.start_as_current_generation(
                        name=final_name, trace_context=trace_context
                    )
                    if as_type == "generation"
                    else langfuse_client.start_as_current_span(
                        name=final_name, trace_context=trace_context
                    )
                )
                if langfuse_client
                else None
            )

            if context_manager is None:
                return await func(*args, **kwargs)

            with context_manager:
                return await func(*args, **kwargs)

        return cast(F, async_wrapper)

    def _sync_observe(
        self,
        func: F,
        *,
        name: Optional[str],
        as_type: Optional[Literal["generation"]],
    ) -> F:
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            trace_id = kwargs.pop("langfuse_trace_id", None)
            parent_observation_id = kwargs.pop("langfuse_parent_observation_id", None)
            trace_context: Optional[TraceContext] = (
                {
                    "trace_id": trace_id,
                    "parent_span_id": parent_observation_id,
                }
                if trace_id
                else None
            )
            final_name = name or func.__name__
            public_key = kwargs.pop("langfuse_public_key", None)
            langfuse_client = get_client(public_key=public_key)
            context_manager = (
                (
                    langfuse_client.start_as_current_generation(
                        name=final_name, trace_context=trace_context
                    )
                    if as_type == "generation"
                    else langfuse_client.start_as_current_span(
                        name=final_name, trace_context=trace_context
                    )
                )
                if langfuse_client
                else None
            )

            if context_manager is None:
                return func(*args, **kwargs)

            with context_manager:
                return func(*args, **kwargs)

        return cast(F, sync_wrapper)


_decorator = LangfuseDecorator()

observe = _decorator.observe
