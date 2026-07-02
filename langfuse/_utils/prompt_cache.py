"""@private"""

import atexit
import os
from datetime import datetime
from queue import Queue
from threading import RLock, Thread
from typing import Callable, Dict, List, Optional, Set

from langfuse._client.environment_variables import (
    LANGFUSE_PROMPT_CACHE_DEFAULT_TTL_SECONDS,
)
from langfuse.logger import langfuse_logger as logger
from langfuse.model import PromptClient

DEFAULT_PROMPT_CACHE_TTL_SECONDS = int(
    os.getenv(LANGFUSE_PROMPT_CACHE_DEFAULT_TTL_SECONDS, 60)
)

DEFAULT_PROMPT_CACHE_REFRESH_WORKERS = 1
_SHUTDOWN_SENTINEL = object()


class PromptCacheItem:
    def __init__(self, prompt: PromptClient, ttl_seconds: int):
        self.value = prompt
        self._expiry = ttl_seconds + self.get_epoch_seconds()

    def is_expired(self) -> bool:
        return self.get_epoch_seconds() > self._expiry

    @staticmethod
    def get_epoch_seconds() -> int:
        return int(datetime.now().timestamp())


class PromptCacheRefreshConsumer(Thread):
    _queue: Queue
    _identifier: int
    running: bool = True

    def __init__(self, queue: Queue, identifier: int):
        super().__init__()
        self.daemon = True
        self._queue = queue
        self._identifier = identifier

    def run(self) -> None:
        while self.running:
            task = self._queue.get()

            if task is _SHUTDOWN_SENTINEL:
                self._queue.task_done()
                break

            logger.debug(
                f"PromptCacheRefreshConsumer processing task, {self._identifier}"
            )
            try:
                task()
            # Task failed, but we still consider it processed
            except Exception as e:
                logger.warning(
                    f"PromptCacheRefreshConsumer encountered an error, cache was not refreshed: {self._identifier}, {e}"
                )

            self._queue.task_done()

    def pause(self) -> None:
        """Pause the consumer."""
        self.running = False


class PromptCacheTaskManager(object):
    _consumers: List[PromptCacheRefreshConsumer]
    _threads: int
    _queue: Queue
    _processing_keys: Set[str]
    _lock: RLock

    def __init__(self, threads: int = 1):
        self._queue = Queue()
        self._consumers = []
        self._threads = threads
        self._processing_keys = set()
        self._lock = RLock()

        for i in range(self._threads):
            consumer = PromptCacheRefreshConsumer(self._queue, i)
            consumer.start()
            self._consumers.append(consumer)

        atexit.register(self.shutdown)

    def add_task(self, key: str, task: Callable[[], None]) -> None:
        with self._lock:
            if key not in self._processing_keys:
                logger.debug(f"Adding prompt cache refresh task for key: {key}")
                self._processing_keys.add(key)
                wrapped_task = self._wrap_task(key, task)
                self._queue.put((wrapped_task))
            else:
                logger.debug(
                    f"Prompt cache refresh task already submitted for key: {key}"
                )

    def active_tasks(self) -> int:
        with self._lock:
            return len(self._processing_keys)

    def wait_for_idle(self) -> None:
        self._queue.join()

    def _wrap_task(self, key: str, task: Callable[[], None]) -> Callable[[], None]:
        def wrapped() -> None:
            logger.debug(f"Refreshing prompt cache for key: {key}")
            try:
                task()
            finally:
                with self._lock:
                    self._processing_keys.remove(key)
                logger.debug(f"Refreshed prompt cache for key: {key}")

        return wrapped

    def shutdown(self) -> None:
        logger.debug(
            f"Shutting down prompt refresh task manager, {len(self._consumers)} consumers,..."
        )

        atexit.unregister(self.shutdown)

        for consumer in self._consumers:
            consumer.pause()

        for _ in self._consumers:
            self._queue.put(_SHUTDOWN_SENTINEL)

        for consumer in self._consumers:
            try:
                consumer.join()
            except RuntimeError:
                # consumer thread has not started
                pass

        logger.debug("Shutdown of prompt refresh task manager completed.")


class PromptCache:
    _cache: Dict[str, PromptCacheItem]
    _lock: RLock

    _task_manager: PromptCacheTaskManager
    """Task manager for refreshing cache"""

    def __init__(
        self, max_prompt_refresh_workers: int = DEFAULT_PROMPT_CACHE_REFRESH_WORKERS
    ):
        self._cache = {}
        self._lock = RLock()
        self._task_manager = PromptCacheTaskManager(threads=max_prompt_refresh_workers)
        logger.debug("Prompt cache initialized.")

    def get(self, key: str) -> Optional[PromptCacheItem]:
        with self._lock:
            return self._cache.get(key, None)

    def set(self, key: str, value: PromptClient, ttl_seconds: Optional[int]) -> None:
        if ttl_seconds is None:
            ttl_seconds = DEFAULT_PROMPT_CACHE_TTL_SECONDS

        with self._lock:
            self._cache[key] = PromptCacheItem(value, ttl_seconds)

    def delete(self, key: str) -> None:
        with self._lock:
            self._cache.pop(key, None)

    def invalidate(self, prompt_name: str) -> None:
        """Invalidate all cached prompts with the given prompt name."""
        with self._lock:
            for key in list(self._cache):
                if key.startswith(prompt_name):
                    del self._cache[key]

    def add_refresh_prompt_task(self, key: str, fetch_func: Callable[[], None]) -> None:
        logger.debug(f"Submitting refresh task for key: {key}")
        self._task_manager.add_task(key, fetch_func)

    def add_refresh_prompt_task_if_current(
        self,
        key: str,
        expected_item: PromptCacheItem,
        fetch_func: Callable[[], None],
    ) -> None:
        with self._lock:
            current_item = self._cache.get(key)
            if (
                current_item is not None
                and current_item is not expected_item
                and not current_item.is_expired()
            ):
                logger.debug(
                    f"Skipping refresh task for key: {key} because cache is already fresh."
                )
                return

        self.add_refresh_prompt_task(key, fetch_func)

    def clear(self) -> None:
        """Clear the entire prompt cache, removing all cached prompts."""
        with self._lock:
            self._cache.clear()

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
