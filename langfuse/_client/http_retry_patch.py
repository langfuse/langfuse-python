"""HTTP client patch for handling unknown field errors.

This module provides functionality to automatically retry API requests when the server
rejects unknown or unexpected fields. This enables backward compatibility when using
newer SDK versions with older server versions.
"""

import logging
from typing import Any, Set

import httpx

logger = logging.getLogger(__name__)


def patch_client_for_unknown_field_retry(client: httpx.Client) -> None:
    """Patch an httpx.Client instance to automatically retry on unknown field errors.

    When the server returns a 400/422 error with unrecognized_keys, this automatically
    retries the request with those fields removed.

    Args:
        client: The httpx.Client instance to patch

    Example:
        >>> client = httpx.Client()
        >>> patch_client_for_unknown_field_retry(client)
        >>> # Now all requests through this client will handle unknown field errors
    """
    original_request = client.request

    def request_with_retry(
        method: str, url: str | httpx.URL, **kwargs: Any
    ) -> httpx.Response:
        """Wrapped request that handles unknown field errors with retry."""
        response = original_request(method, url, **kwargs)

        # Retry if server rejected unrecognized keys
        if response.status_code in [400, 422] and "json" in kwargs:
            try:
                unknown_keys = _extract_unknown_keys(response)
                if unknown_keys:
                    logger.warning(
                        "Server rejected unrecognized keys %s for %s %s. Retrying without these fields.",
                        unknown_keys,
                        method,
                        url,
                    )
                    kwargs["json"] = _remove_fields(kwargs["json"], unknown_keys)
                    response = original_request(method, url, **kwargs)
            except Exception as e:
                logger.debug("Failed to parse unknown field error: %s", e)

        return response

    client.request = request_with_retry  # type: ignore[method-assign]


def patch_async_client_for_unknown_field_retry(client: httpx.AsyncClient) -> None:
    """Patch an httpx.AsyncClient instance to automatically retry on unknown field errors.

    Async version of patch_client_for_unknown_field_retry.

    Args:
        client: The httpx.AsyncClient instance to patch
    """
    original_request = client.request

    async def request_with_retry(
        method: str, url: str | httpx.URL, **kwargs: Any
    ) -> httpx.Response:
        """Wrapped async request that handles unknown field errors with retry."""
        response = await original_request(method, url, **kwargs)

        # Retry if server rejected unrecognized keys
        if response.status_code in [400, 422] and "json" in kwargs:
            try:
                unknown_keys = _extract_unknown_keys(response)
                if unknown_keys:
                    logger.warning(
                        "Server rejected unrecognized keys %s for %s %s. Retrying without these fields.",
                        unknown_keys,
                        method,
                        url,
                    )
                    kwargs["json"] = _remove_fields(kwargs["json"], unknown_keys)
                    response = await original_request(method, url, **kwargs)
            except Exception as e:
                logger.debug("Failed to parse unknown field error: %s", e)

        return response

    client.request = request_with_retry  # type: ignore[method-assign]


def _extract_unknown_keys(response: httpx.Response) -> Set[str]:
    """Extract unknown keys from server error response.

    Args:
        response: The HTTP response from the server

    Returns:
        Set of field names that were rejected as unrecognized
    """
    body = response.json()
    if isinstance(body, dict) and "error" in body:
        unknown_keys = set()
        for error in body.get("error", []):
            if isinstance(error, dict) and error.get("code") == "unrecognized_keys":
                unknown_keys.update(error.get("keys", []))
        return unknown_keys
    return set()


def _remove_fields(data: Any, fields: Set[str]) -> Any:
    """Remove specified fields from nested dict/list structures.

    Args:
        data: The data structure to filter (dict, list, or primitive)
        fields: Set of field names to remove

    Returns:
        Filtered data structure with specified fields removed
    """
    if isinstance(data, dict):
        return {
            k: _remove_fields(v, fields) for k, v in data.items() if k not in fields
        }
    elif isinstance(data, list):
        return [_remove_fields(item, fields) for item in data]
    else:
        return data
