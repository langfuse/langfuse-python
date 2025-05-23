"""Environment variable definitions for Langfuse OpenTelemetry integration.

This module defines environment variables used to configure the Langfuse OpenTelemetry integration.
Each environment variable includes documentation on its purpose, expected values, and defaults.
"""

LANGFUSE_TRACING_ENVIRONMENT = "LANGFUSE_TRACING_ENVIRONMENT"
"""
.. envvar:: LANGFUSE_TRACING_ENVIRONMENT

The tracing environment. Can be any lowercase alphanumeric string with hyphens and underscores that does not start with 'langfuse'.

**Default value:** ``"default"``
"""

LANGFUSE_RELEASE = "LANGFUSE_RELEASE"
"""
.. envvar:: LANGFUSE_RELEASE

Release number/hash of the application to provide analytics grouped by release.
"""


LANGFUSE_PUBLIC_KEY = "LANGFUSE_PUBLIC_KEY"
"""
.. envvar:: LANGFUSE_PUBLIC_KEY

Public API key of Langfuse project
"""

LANGFUSE_SECRET_KEY = "LANGFUSE_SECRET_KEY"
"""
.. envvar:: LANGFUSE_SECRET_KEY

Secret API key of Langfuse project
"""

LANGFUSE_HOST = "LANGFUSE_HOST"
"""
.. envvar:: LANGFUSE_HOST

Host of Langfuse API. Can be set via `LANGFUSE_HOST` environment variable.

**Default value:** ``"https://cloud.langfuse.com"``
"""

LANGFUSE_DEBUG = "LANGFUSE_DEBUG"
"""
.. envvar:: LANGFUSE_DEBUG

Enables debug mode for more verbose logging.

**Default value:** ``"False"``
"""

LANGFUSE_TRACING_ENABLED = "LANGFUSE_TRACING_ENABLED"
"""
.. envvar:: LANGFUSE_TRACING_ENABLED

Enables or disables the Langfuse client. If disabled, all observability calls to the backend will be no-ops. Default is True. Set to `False` to disable tracing.

**Default value:** ``"True"``
"""

LANGFUSE_MEDIA_UPLOAD_THREAD_COUNT = "LANGFUSE_MEDIA_UPLOAD_THREAD_COUNT"
"""
.. envvar:: LANGFUSE_MEDIA_UPLOAD_THREAD_COUNT 

Number of background threads to handle media uploads from trace ingestion.

**Default value:** ``1``
"""

LANGFUSE_FLUSH_AT = "LANGFUSE_FLUSH_AT"
"""
.. envvar:: LANGFUSE_FLUSH_AT

Max batch size until a new ingestion batch is sent to the API.
**Default value:** ``15``
"""

LANGFUSE_FLUSH_INTERVAL = "LANGFUSE_FLUSH_INTERVAL"
"""
.. envvar:: LANGFUSE_FLUSH_INTERVAL

Max delay until a new ingestion batch is sent to the API.
**Default value:** ``1``
"""

LANGFUSE_SAMPLE_RATE = "LANGFUSE_SAMPLE_RATE"
"""
.. envvar: LANGFUSE_SAMPLE_RATE

Float between 0 and 1 indicating the sample rate of traces to bet sent to Langfuse servers.

**Default value**: ``1.0``
"""
