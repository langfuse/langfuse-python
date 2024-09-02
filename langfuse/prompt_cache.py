"""@private"""

from datetime import datetime
from typing import Optional, Dict
import threading
from concurrent.futures import ThreadPoolExecutor
import atexit
import logging

from langfuse.model import PromptClient


DEFAULT_PROMPT_CACHE_TTL_SECONDS = 60


class PromptCacheItem:
    def __init__(self, prompt: PromptClient, ttl_seconds: int):
        self.value = prompt
        self._expiry = ttl_seconds + self.get_epoch_seconds()

    def is_expired(self) -> bool:
        return self.get_epoch_seconds() > self._expiry

    @staticmethod
    def get_epoch_seconds() -> int:
        return int(datetime.now().timestamp())


class PromptCache:
    _cache: Dict[str, PromptCacheItem]

    _refreshing_keys: Dict[str, threading.Event]
    """Keys that are currently being refreshed"""

    _refresh_executor: ThreadPoolExecutor
    """Executor for refreshing cache"""

    _log = logging.getLogger("langfuse")

    def __init__(self, max_prompt_refresh_workers: int = 1):
        self._cache = {}
        self._refreshing_keys = {}
        self._refresh_executor = ThreadPoolExecutor(
            max_workers=max_prompt_refresh_workers,
            thread_name_prefix="LangfusePromptCacheRefreshExecutor",
        )
        atexit.register(self._shutdown_refresh_executor)
        self._log.debug(f"Prompt cache initialized.")

    def get(self, key: str) -> Optional[PromptCacheItem]:
        return self._cache.get(key, None)

    def set(self, key: str, value: PromptClient, ttl_seconds: Optional[int]):
        if ttl_seconds is None:
            ttl_seconds = DEFAULT_PROMPT_CACHE_TTL_SECONDS

        self._cache[key] = PromptCacheItem(value, ttl_seconds)

    def refresh_prompt(self, key: str, fetch_func):
        if key not in self._refreshing_keys:
            self._log.debug(f"Submitting refresh task for key: {key}")
            self._refreshing_keys[key] = threading.Event()
            self._refresh_executor.submit(self._refresh_task, key, fetch_func)
        else:
            self._log.debug(f"Refresh task already submitted for key: {key}")

    def _refresh_task(self, key: str, fetch_func):
        """Run refresh task which updates cache itself, removes key from refreshing keys when done."""
        try:
            self._log.debug(f"Refreshing prompt cache for key: {key}")
            fetch_func()
            self._log.debug(f"Refreshed prompt cache for key: {key}")
        finally:
            self._refreshing_keys.pop(key, None)

    def _shutdown_refresh_executor(self):
        self._log.debug(f"Shutting down prompt refresh executor ...")
        self._refresh_executor.shutdown(wait=False)
        self._log.debug(f"Shutdown of prompt refresh executor completed.")

    @staticmethod
    def generate_cache_key(
        name: str, *, version: Optional[int], label: Optional[str]
    ) -> str:
        parts = [name]

        if version is not None:
            parts.append(f"version:{version}")

        elif label is not None:
            parts.append(f"label:{label}")

        else:
            # Default to production labeled prompt
            parts.append("label:production")

        return "-".join(parts)
