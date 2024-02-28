from collections import defaultdict
from contextvars import ContextVar
from datetime import datetime
from functools import wraps
import logging
import os
from typing import Any, Callable, DefaultDict, List, Optional, Union

from langfuse.client import Langfuse, StatefulSpanClient, StatefulTraceClient
from langfuse.llama_index import LlamaIndexCallbackHandler
from langfuse.types import ObservationParams, SpanLevel
from langfuse.utils import _get_timestamp


langfuse_context: ContextVar[Optional[Langfuse]] = ContextVar(
    "langfuse_context", default=None
)
observation_stack_context: ContextVar[
    List[Union[StatefulTraceClient, StatefulSpanClient]]
] = ContextVar("observation_stack_context", default=[])
observation_params_context: ContextVar[
    DefaultDict[str, ObservationParams]
] = ContextVar(
    "observation_params_context",
    default=defaultdict(
        lambda: {
            "name": None,
            "user_id": None,
            "session_id": None,
            "version": None,
            "release": None,
            "metadata": None,
            "tags": None,
            "input": None,
            "output": None,
            "level": None,
            "status_message": None,
            "start_time": None,
            "end_time": None,
        },
    ),
)


class LangfuseDecorator:
    log = logging.getLogger("langfuse")

    def __init__(self):
        self._langfuse: Optional[Langfuse] = None

    def trace(self, func: Callable) -> Callable:
        """
        Wraps a function to automatically create and manage Langfuse tracing around its execution.

        This decorator captures the start and end times of the function, along with input parameters and output results, and automatically updates the observation context.
        It creates traces for top-level function calls and spans for nested function calls, ensuring that each observation is correctly associated with its parent observation.
        In the event of an exception, the observation is updated with error details.

        Usage:
            @langfuse.trace\n
            def your_function_name(args):
                # Your function implementation

        Parameters:
            func (Callable): The function to be wrapped by the decorator.

        Returns:
            Callable: A wrapper function that, when called, executes the original function within a Langfuse observation context.

        Raises:
            Exception: Propagates any exceptions raised by the wrapped function, after logging and updating the observation with error details.

        Note:
            - The decorator automatically manages observation IDs and context, but you can manually specify an observation ID using the `langfuse_observation_id` keyword argument when calling the wrapped function.
            - To update observation or trace parameters (e.g., metadata, session_id), use `langfuse.update_current_observation` and `langfuse.set_current_trace_params` within the wrapped function.
        """

        @wraps(func)
        def wrapper(*args, **kwargs):
            langfuse = self._get_langfuse()
            stack = observation_stack_context.get().copy()
            parent = stack[-1] if stack else None

            # Collect default observation data
            name = func.__name__
            observation_id = kwargs.pop("langfuse_observation_id", None)
            id = str(observation_id) if observation_id else None
            input = {"args": args, "kwargs": kwargs}
            start_time = _get_timestamp()

            # Create observation
            observation = (
                parent.span(id=id, name=name, start_time=start_time, input=input)
                if parent
                else langfuse.trace(
                    id=id, name=name, start_time=start_time, input=input
                )
            )

            # Add observation to top of stack
            observation_stack_context.set(stack + [observation])

            # Call the wrapped function
            result = None
            try:
                result = func(*args, **kwargs)
            except Exception as e:
                observation_params_context.get()[observation.id].update(
                    level="ERROR", status_message=str(e)
                )
                raise e
            finally:
                # Collect final observation data
                observation_params = observation_params_context.get()[observation.id]
                end_time = observation_params["end_time"] or _get_timestamp()
                output = observation_params["output"] or str(result) if result else None
                observation_params.update(end_time=end_time, output=output)

                if isinstance(observation, StatefulSpanClient):
                    observation.end(**observation_params)
                elif isinstance(observation, StatefulTraceClient):
                    observation.update(**observation_params)

                # Remove observation from top of stack by resetting to initial stack
                observation_stack_context.set(stack)

            return result

        return wrapper

    def get_current_llama_index_handler(self):
        """
        Retrieves the current LlamaIndexCallbackHandler associated with the most recent observation in the observation stack.

        This method fetches the current observation from the observation stack and returns a LlamaIndexCallbackHandler initialized with this observation.
        It is intended to be used within the context of a trace, allowing access to a callback handler for operations that require interaction with the LlamaIndex API based on the current observation context.

        See the Langfuse documentation for more information on integrating the LlamaIndexCallbackHandler.

        Returns:
            LlamaIndexCallbackHandler or None: Returns a LlamaIndexCallbackHandler instance if there is an active observation in the current context; otherwise, returns None if no observation is found.

        Note:
            - This method should be called within the context of a trace (i.e., within a function wrapped by @langfuse.trace) to ensure that an observation context exists.
            - If no observation is found in the current context (e.g., if called outside of a trace or if the observation stack is empty), the method logs a warning and returns None.
        """

        observation = observation_stack_context.get()[-1]

        if observation is None:
            self.log.warn("No observation found in the current context")

            return None

        callback_handler = LlamaIndexCallbackHandler()
        callback_handler.set_root(observation)

        return callback_handler

    def get_current_langchain_handler(self):
        """
        Retrieves the current LangchainCallbackHandler associated with the most recent observation in the observation stack.

        This method fetches the current observation from the observation stack and returns a LangchainCallbackHandler initialized with this observation.
        It is intended to be used within the context of a trace, allowing access to a callback handler for operations that require interaction with Langchain based on the current observation context.

        See the Langfuse documentation for more information on integrating the LangchainCallbackHandler.

        Returns:
            LangchainCallbackHandler or None: Returns a LangchainCallbackHandler instance if there is an active observation in the current context; otherwise, returns None if no observation is found.

        Note:
            - This method should be called within the context of a trace (i.e., within a function wrapped by @langfuse.trace) to ensure that an observation context exists.
            - If no observation is found in the current context (e.g., if called outside of a trace or if the observation stack is empty), the method logs a warning and returns None.
        """
        observation = observation_stack_context.get()[-1]

        if observation is None:
            self.log.warn("No observation found in the current context")

            return None

        return observation.get_langchain_handler()

    def get_current_trace_id(self):
        """
        Retrieves the ID of the current trace from the observation stack context.

        This method examines the observation stack to find the root trace and returns its ID. It is useful for operations that require the trace ID,
        such as setting trace parameters or querying trace information. The trace ID is typically the ID of the first observation in the stack,
        representing the entry point of the traced execution context.

        Returns:
            str or None: The ID of the current trace if available; otherwise, None. A return value of None indicates that there is no active trace in the current context,
            possibly due to the method being called outside of any @langfuse.trace-decorated function execution.

        Note:
            - This method should be called within the context of a trace (i.e., inside a function wrapped with the @langfuse.trace decorator) to ensure that a current trace is indeed present and its ID can be retrieved.
            - If called outside of a trace context, or if the observation stack has somehow been corrupted or improperly managed, this method will log a warning and return None, indicating the absence of a traceable context.
        """
        stack = observation_stack_context.get()

        if not stack:
            self.log.warn("No trace found in the current context")

            return None

        return stack[0].id

    def set_current_trace_params(
        self,
        name: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        version: Optional[str] = None,
        release: Optional[str] = None,
        metadata: Optional[Any] = None,
        tags: Optional[List[str]] = None,
    ):
        """
        Sets parameters for the current trace, updating the trace's metadata and context information.

        This method allows for dynamically updating the trace parameters at any point during the execution of a trace.
        It updates the parameters of the current trace based on the provided arguments. These parameters include metadata, session information,
        and other trace attributes that can be useful for categorization, filtering, and analysis in the Langfuse UI.


        Parameters:
        - name (Optional[str]): Identifier of the trace. Useful for sorting/filtering in the UI..
        - user_id (Optional[str]): The id of the user that triggered the execution. Used to provide user-level analytics.
        - session_id (Optional[str]): Used to group multiple traces into a session in Langfuse. Use your own session/thread identifier.
        - version (Optional[str]): The version of the trace type. Used to understand how changes to the trace type affect metrics. Useful in debugging.
        - release (Optional[str]): The release identifier of the current deployment. Used to understand how changes of different deployments affect metrics. Useful in debugging.
        - metadata (Optional[Any]): Additional metadata of the trace. Can be any JSON object. Metadata is merged when being updated via the API.
        - tags (Optional[List[str]]): Tags are used to categorize or label traces. Traces can be filtered by tags in the Langfuse UI and GET API.

        Returns:
            None

        Note:
            - This method should be used within the context of an active trace, typically within a function that is being traced using the @langfuse.trace decorator.
            - The method updates the trace parameters for the currently executing trace. In nested trace scenarios, it affects the most recent trace context.
            - If called outside of an active trace context, a warning is logged, and a ValueError is raised to indicate the absence of a traceable context.
        """
        trace_id = self.get_current_trace_id()

        if trace_id is None:
            self.log.warn("No trace found in the current context")

            return

        observation_params_context.get()[trace_id].update(
            {
                "name": name,
                "user_id": user_id,
                "session_id": session_id,
                "version": version,
                "release": release,
                "metadata": metadata,
                "tags": tags,
            }
        )

    def update_current_observation(
        self,
        *,
        input: Optional[Any] = None,
        output: Optional[Any] = None,
        name: Optional[str] = None,
        version: Optional[str] = None,
        metadata: Optional[Any] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        release: Optional[str] = None,
        tags: Optional[List[str]] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        level: Optional[SpanLevel] = None,
        status_message: Optional[str] = None,
    ):
        """
        Updates parameters for the current observation within an active trace context.

        This method dynamically adjusts the parameters of the most recent observation on the observation stack.
        It allows for the enrichment of observation data with additional details such as input parameters, output results, metadata, and more,
        enhancing the observability and traceability of the execution context.

        Parameters:
            - input (Optional[Any]): The input parameters of the observation, providing context about the observed operation or function call.
            - output (Optional[Any]): The output or result of the observation
            - name (Optional[str]): Identifier of the trace. Useful for sorting/filtering in the UI..
            - user_id (Optional[str]): The id of the user that triggered the execution. Used to provide user-level analytics.
            - session_id (Optional[str]): Used to group multiple traces into a session in Langfuse. Use your own session/thread identifier.
            - version (Optional[str]): The version of the trace type. Used to understand how changes to the trace type affect metrics. Useful in debugging.
            - release (Optional[str]): The release identifier of the current deployment. Used to understand how changes of different deployments affect metrics. Useful in debugging.
            - metadata (Optional[Any]): Additional metadata of the trace. Can be any JSON object. Metadata is merged when being updated via the API.
            - tags (Optional[List[str]]): Tags are used to categorize or label traces. Traces can be filtered by tags in the Langfuse UI and GET API.
            - start_time (Optional[datetime]): The start time of the observation, allowing for custom time range specification.
            - end_time (Optional[datetime]): The end time of the observation, enabling precise control over the observation duration.
            - level (Optional[SpanLevel]): The severity or importance level of the observation, such as "INFO", "WARNING", or "ERROR".
            - status_message (Optional[str]): A message or description associated with the observation's status, particularly useful for error reporting.

        Returns:
            None

        Raises:
            ValueError: If no current observation is found in the context, indicating that this method was called outside of an observation's execution scope.

        Note:
            - This method is intended to be used within the context of an active observation, typically within a function wrapped by the @langfuse.trace decorator.
            - It updates the parameters of the most recently created observation on the observation stack. Care should be taken in nested observation contexts to ensure the updates are applied as intended.
            - Parameters set to `None` will not overwrite existing values for those parameters. This behavior allows for selective updates without clearing previously set information.
        """
        stack = observation_stack_context.get()
        observation = stack[-1] if stack else None

        if not observation:
            self.log.warn("No observation found in the current context")

            return

        observation_params_context.get()[observation.id].update(
            input=input,
            output=output,
            name=name,
            version=version,
            metadata=metadata,
            start_time=start_time,
            end_time=end_time,
            release=release,
            tags=tags,
            user_id=user_id,
            session_id=session_id,
            level=level,
            status_message=status_message,
        )

    def flush(self):
        """
        Forces the immediate flush of all buffered observations to the Langfuse backend.

        This method triggers the explicit sending of all accumulated trace and observation data that has not yet been sent to Langfuse servers.
        It is typically used to ensure that data is promptly available for analysis, especially at the end of an execution context or before the application exits.

        Usage:
            - This method can be called at strategic points in the application where it's crucial to ensure that all telemetry data captured up to that point is made persistent and visible on the Langfuse platform.
            - It's particularly useful in scenarios where the application might terminate abruptly or in batch processing tasks that require periodic flushing of trace data.

        Returns:
            None

        Raises:
            ValueError: If it fails to find a Langfuse client object in the current context, indicating potential misconfiguration or initialization issues.

        Note:
            - The flush operation may involve network I/O to send data to the Langfuse backend, which could impact performance if called too frequently in performance-sensitive contexts.
            - In long-running applications, it's often sufficient to rely on the automatic flushing mechanism provided by the Langfuse client.
            However, explicit calls to `flush` can be beneficial in certain edge cases or for debugging purposes.
        """

        langfuse = self._get_langfuse()
        if langfuse:
            langfuse.flush()
        else:
            self.log.warn("No langfuse object found in the current context")

    def _get_langfuse(self) -> Langfuse:
        if self._langfuse:
            return self._langfuse

        public_key = os.environ.get("LANGFUSE_PUBLIC_KEY")
        secret_key = os.environ.get("LANGFUSE_SECRET_KEY")
        host = os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com")

        if public_key and secret_key:
            self._langfuse = Langfuse(
                public_key=public_key, secret_key=secret_key, host=host
            )

            return self._langfuse

        else:
            raise ValueError(
                "Missing LANGFUSE_SECRET_KEY or LANGFUSE_PUBLIC_KEY environment variables"
            )


langfuse = LangfuseDecorator()
