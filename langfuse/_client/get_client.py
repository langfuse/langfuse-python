from typing import Optional

from langfuse._client.client import Langfuse
from langfuse._client.resource_manager import LangfuseResourceManager
from langfuse.logger import langfuse_logger


def get_client(*, public_key: Optional[str] = None) -> Langfuse:
    """Get or create a Langfuse client instance.

    Returns an existing Langfuse client or creates a new one if none exists. In multi-project setups,
    providing a public_key is required. Multi-project support is experimental - see Langfuse docs.

    Behavior:
    - Single project: Returns existing client or creates new one
    - Multi-project: Requires public_key to return specific client
    - No public_key in multi-project: Returns disabled client to prevent data leakage

    The function uses a singleton pattern per public_key to conserve resources and maintain state.

    Args:
        public_key (Optional[str]): Project identifier
            - With key: Returns client for that project
            - Without key: Returns single client or disabled client if multiple exist

    Returns:
        Langfuse: Client instance in one of three states:
            1. Client for specified public_key
            2. Default client for single-project setup
            3. Disabled client when multiple projects exist without key

    Security:
        Disables tracing when multiple projects exist without explicit key to prevent
        cross-project data leakage. Multi-project setups are experimental.

    Example:
        ```python
        # Single project
        client = get_client()  # Default client

        # In multi-project usage:
        client_a = get_client(public_key="project_a_key")  # Returns project A's client
        client_b = get_client(public_key="project_b_key")  # Returns project B's client

        # Without specific key in multi-project setup:
        client = get_client()  # Returns disabled client for safety
        ```
    """
    with LangfuseResourceManager._lock:
        active_instances = LangfuseResourceManager._instances

        if not public_key:
            if len(active_instances) == 0:
                # No clients initialized yet, create default instance
                return Langfuse()

            if len(active_instances) == 1:
                # Only one client exists, safe to use without specifying key
                instance = list(active_instances.values())[0]

                # Initialize with the credentials bound to the instance
                # This is important if the original instance was instantiated
                # via constructor arguments
                return Langfuse(
                    public_key=instance.public_key,
                    secret_key=instance.secret_key,
                    host=instance.host,
                    tracing_enabled=instance.tracing_enabled,
                )

            else:
                # Multiple clients exist but no key specified - disable tracing
                # to prevent cross-project data leakage
                langfuse_logger.warning(
                    "No 'langfuse_public_key' passed to decorated function, but multiple langfuse clients are instantiated in current process. Skipping tracing for this function to avoid cross-project leakage."
                )
                return Langfuse(
                    tracing_enabled=False, public_key="fake", secret_key="fake"
                )

        else:
            # Specific key provided, look up existing instance
            instance = active_instances.get(public_key, None)

            if instance is None:
                # No instance found with this key - client not initialized properly
                langfuse_logger.warning(
                    f"No Langfuse client with public key {public_key} has been initialized. Skipping tracing for decorated function."
                )
                return Langfuse(
                    tracing_enabled=False, public_key="fake", secret_key="fake"
                )

            return Langfuse(
                public_key=public_key,
                secret_key=instance.secret_key,
                host=instance.host,
                tracing_enabled=instance.tracing_enabled,
            )
