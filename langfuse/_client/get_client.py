from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator, List, Optional, Tuple

from langfuse._client.client import Langfuse
from langfuse._client.resource_manager import LangfuseResourceManager
from langfuse.logger import langfuse_logger

# Context variable to track the current (langfuse_public_key, langfuse_project_id)
# in execution context
_current_public_key: ContextVar[Optional[Tuple[Optional[str], Optional[str]]]] = (
    ContextVar("langfuse_public_key", default=None)
)


@contextmanager
def _set_current_public_key(
    public_key: Optional[str], project_id: Optional[str] = None
) -> Iterator[None]:
    """Context manager to set and restore the current client identity in execution context.

    Args:
        public_key: The public key to set in context.
        project_id: The project ID to set in context.
            If both are None, context is not modified.

    Yields:
        None
    """
    if public_key is None and project_id is None:
        yield  # Don't modify context if no key provided
        return

    token = _current_public_key.set((public_key, project_id))
    try:
        yield
    finally:
        _current_public_key.reset(token)


def _create_client_from_instance(
    instance: "LangfuseResourceManager", public_key: Optional[str] = None
) -> Langfuse:
    """Create a Langfuse client from a resource manager instance with all settings preserved."""
    return Langfuse(
        public_key=public_key or instance.public_key,
        secret_key=instance.secret_key,
        base_url=instance.base_url,
        project_id=instance.project_id,
        tracing_enabled=instance.tracing_enabled,
        environment=instance.environment,
        timeout=instance.timeout,
        flush_at=instance.flush_at,
        flush_interval=instance.flush_interval,
        release=instance.release,
        media_upload_thread_count=instance.media_upload_thread_count,
        sample_rate=instance.sample_rate,
        mask=instance.mask,
        mask_otel_spans=instance.mask_otel_spans,
        should_export_span=instance.should_export_span,
        additional_headers=instance.additional_headers,
        tracer_provider=instance.tracer_provider,
        id_generator=instance.id_generator,
        span_exporter=instance.span_exporter,
        otel_compression=instance.otel_compression,
        skills=instance.skills,
        httpx_client=instance.httpx_client,
    )


def get_client(
    *, public_key: Optional[str] = None, project_id: Optional[str] = None
) -> Langfuse:
    """Get or create a Langfuse client instance.

    Returns an existing Langfuse client or creates a new one if none exists. In multi-project setups,
    providing a public_key is required. Multi-project support is experimental - see Langfuse docs.

    A client is identified by its public key and its project ID. Project API keys leave
    the project ID unset. An organization API key can back several clients, one per
    project, so pass the project_id as well to pick one of them.

    Behavior:
    - Single project: Returns existing client or creates new one
    - Multi-project: Requires public_key (and project_id for an organization key shared
      by several clients) to return a specific client
    - No public_key in multi-project, or a public_key shared by several clients without
      a project_id: Returns disabled client to prevent data leakage

    The function uses a singleton pattern per (public_key, project_id) to conserve resources and maintain state.

    Args:
        public_key (Optional[str]): Project identifier
            - With key: Returns client for that project
            - Without key: Returns single client or disabled client if multiple exist
        project_id (Optional[str]): Project ID the client was created with. Only needed
            to choose between several clients that share an organization API key.

    Returns:
        Langfuse: Client instance in one of three states:
            1. Client for specified public_key (and project_id)
            2. Default client for single-project setup
            3. Disabled client when the lookup matches no client or more than one

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

        # One organization API key writing to two projects:
        Langfuse(public_key="org_key", secret_key="...", project_id="project_a_id")
        Langfuse(public_key="org_key", secret_key="...", project_id="project_b_id")

        client_a = get_client(public_key="org_key", project_id="project_a_id")
        client_b = get_client(public_key="org_key", project_id="project_b_id")
        client = get_client(public_key="org_key")  # Ambiguous: disabled client
        ```
    """
    with LangfuseResourceManager._lock:
        active_instances = LangfuseResourceManager._instances

        # If no explicit client identity provided, check execution context
        if not public_key and not project_id:
            public_key, project_id = _current_public_key.get(None) or (None, None)

        if not public_key and not project_id:
            if len(active_instances) == 0:
                # No clients initialized yet, create default instance
                return Langfuse()

            if len(active_instances) == 1:
                # Only one client exists, safe to use without specifying key
                instance = list(active_instances.values())[0]

                # Initialize with the credentials bound to the instance
                # This is important if the original instance was instantiated
                # via constructor arguments
                return _create_client_from_instance(instance)

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
            # Specific identity provided, look up matching instances
            matching_instances: List[LangfuseResourceManager] = [
                instance
                for (instance_public_key, instance_project_id), instance in (
                    active_instances.items()
                )
                if (not public_key or instance_public_key == public_key)
                and (not project_id or instance_project_id == project_id)
            ]

            if len(matching_instances) == 0:
                # No instance found with this key - client not initialized properly
                langfuse_logger.warning(
                    "No Langfuse client with public key %s and project id %s has been "
                    "initialized. Skipping tracing for decorated function.",
                    public_key,
                    project_id,
                )
                return Langfuse(
                    tracing_enabled=False, public_key="fake", secret_key="fake"
                )

            if len(matching_instances) > 1:
                # Several projects share this organization key - disable tracing
                # to prevent cross-project data leakage
                langfuse_logger.warning(
                    "Multiple Langfuse clients with public key %s are instantiated for "
                    "different projects, but no 'langfuse_project_id' was passed. "
                    "Skipping tracing to avoid cross-project leakage.",
                    public_key,
                )
                return Langfuse(
                    tracing_enabled=False, public_key="fake", secret_key="fake"
                )

            return _create_client_from_instance(matching_instances[0], public_key)
