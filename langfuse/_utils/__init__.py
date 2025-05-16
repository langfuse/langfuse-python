"""@private"""

import logging
import typing
from datetime import datetime, timezone

from langfuse.model import PromptClient

log = logging.getLogger("langfuse")


def _get_timestamp():
    return datetime.now(timezone.utc)


def _create_prompt_context(
    prompt: typing.Optional[PromptClient] = None,
):
    if prompt is not None and not prompt.is_fallback:
        return {"prompt_version": prompt.version, "prompt_name": prompt.name}

    return {"prompt_version": None, "prompt_name": None}
