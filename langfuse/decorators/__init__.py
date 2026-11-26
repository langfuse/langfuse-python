"""Removed in Python SDK v3. Use `from langfuse import observe` instead."""

_message = (
    "langfuse.decorators was removed in Python SDK v3. "
    "Use `from langfuse import observe`. "
    "Migration guide: https://langfuse.com/docs/observability/sdk/upgrade-path/python-v2-to-v3"
)
raise ImportError(_message)
