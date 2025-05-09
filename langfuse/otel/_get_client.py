from typing import Optional

from ..otel import Langfuse
from ._logger import langfuse_logger
from ._tracer import LangfuseTracer


def get_client(*, public_key: Optional[str] = None) -> Langfuse:
    """Retrieve or create a Langfuse client based on optionally provided public key.

    Client Management Strategy:
    1. Single-Project Case:
       - When only one Langfuse client exists in the process, returns that client
       - If no clients exist yet, initializes a new client using environment variables

    2. Multi-Project Case:
       - With explicit public_key: Returns the specific client for that project
       - Without public_key: Returns a disabled tracing client to prevent data leakage

    Security Features:
    The function implements a security-first approach to prevent cross-project data
    leakage. When multiple projects are active in the same process, it requires explicit
    project identification through the public_key parameter. Without this parameter,
    it returns a disabled client that prevents any data from being incorrectly attributed
    to the wrong project.

    Client Lifecycle:
    The function does not create new client instances unnecessarily. It follows a
    singleton pattern per public key, retrieving existing instances where possible
    and only creating new ones when required. This approach conserves system resources
    and maintains consistent client state.

    Args:
        public_key (Optional[str]): The Langfuse public key for project identification.
            - If provided: Returns the client associated with this specific key
            - If omitted: Returns either the single active client (if only one exists)
              or creates a new default client (if none exist) or a disabled client
              (if multiple exist)

    Returns:
        Langfuse: A properly configured Langfuse client instance with one of three states:
            1. Existing client for the specified public_key
            2. New or existing default client when only one project is in use
            3. Disabled client (tracing_enabled=False) to prevent cross-project data leakage
               when multiple projects are active but no specific key is provided

    Security Note:
        This method enforces strict project isolation by disabling tracing when
        multiple projects are used without explicit project identification, protecting
        against accidental data cross-contamination between different projects.

    Example Usage:
        ```python
        # In single-project usage:
        client = get_client()  # Returns the default client

        # In multi-project usage:
        client_a = get_client(public_key="project_a_key")  # Returns project A's client
        client_b = get_client(public_key="project_b_key")  # Returns project B's client

        # Without specific key in multi-project setup:
        client = get_client()  # Returns disabled client for safety
        ```
    """
    with LangfuseTracer._lock:
        active_instances = LangfuseTracer._instances

        if not public_key:
            if len(active_instances) == 0:
                # No clients initialized yet, create default instance
                return Langfuse()

            if len(active_instances) == 1:
                # Only one client exists, safe to use without specifying key
                return Langfuse(public_key=public_key)

            else:
                # Multiple clients exist but no key specified - disable tracing
                # to prevent cross-project data leakage
                langfuse_logger.warning(
                    "No 'langfuse_public_key' passed to decorated function, but multiple langfuse clients are instantiated in current process. Skipping tracing for this function to avoid cross-project leakage."
                )
                return Langfuse(tracing_enabled=False)

        else:
            # Specific key provided, look up existing instance
            instance = active_instances.get(public_key, None)

            if instance is None:
                # No instance found with this key - client not initialized properly
                langfuse_logger.warning(
                    f"No Langfuse client with public key {public_key} has been initialized. Skipping tracing for decorated function."
                )
                return Langfuse(tracing_enabled=False)

            return Langfuse(public_key=public_key)
