"""Exception hierarchy for the Langfuse Python SDK.

All exceptions defined in hand-written SDK code derive from
:class:`LangfuseError`, so applications can catch every SDK-raised error with
a single ``except LangfuseError`` clause while fine-grained subclasses remain
available for callers that need to distinguish failure modes. Every class in
this hierarchy is also an ``Exception`` subclass, so existing
``except Exception`` handlers keep working.

Errors raised by the generated API client (``langfuse.api``) are not part of
this hierarchy; they derive from the generated ``langfuse.api.core.ApiError``
base instead.
"""

__all__ = ["AuthError", "LangfuseError"]


class LangfuseError(Exception):
    """Base class for all exceptions raised by the Langfuse SDK."""


class AuthError(LangfuseError):
    """Raised when authentication with the Langfuse API fails.

    Example: ``Langfuse.auth_check()`` raises this when the configured
    credentials are accepted but resolve to no project.
    """
