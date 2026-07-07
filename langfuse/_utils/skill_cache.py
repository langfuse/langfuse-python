"""@private"""

import atexit
import os
from datetime import datetime
from queue import Queue
from threading import RLock, Thread
from typing import Callable, Dict, List, Optional, Set

from langfuse._client.environment_variables import (
    LANGFUSE_SKILL_CACHE_DEFAULT_TTL_SECONDS,
)
from langfuse.logger import langfuse_logger as logger
from langfuse.model import SkillClient

DEFAULT_SKILL_CACHE_TTL_SECONDS = int(
    os.getenv(LANGFUSE_SKILL_CACHE_DEFAULT_TTL_SECONDS, 60)
)

DEFAULT_SKILL_CACHE_REFRESH_WORKERS = 1
_SHUTDOWN_SENTINEL = object()


class SkillCacheItem:
    def __init__(self, skill: SkillClient, ttl_seconds: int):
        self.value = skill
        self._expiry = ttl_seconds + self.get_epoch_seconds()

    def is_expired(self) -> bool:
        return self.get_epoch_seconds() > self._expiry

    @staticmethod
    def get_epoch_seconds() -> int:
        return int(datetime.now().timestamp())


class SkillCacheRefreshConsumer(Thread):
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
                f"SkillCacheRefreshConsumer processing task, {self._identifier}"
            )
            try:
                task()
            # Task failed, but we still consider it processed
            except Exception as e:
                logger.warning(
                    f"SkillCacheRefreshConsumer encountered an error, cache was not refreshed: {self._identifier}, {e}"
                )

            self._queue.task_done()

    def pause(self) -> None:
        """Pause the consumer."""
        self.running = False


class SkillCacheTaskManager(object):
    _consumers: List[SkillCacheRefreshConsumer]
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
            consumer = SkillCacheRefreshConsumer(self._queue, i)
            consumer.start()
            self._consumers.append(consumer)

        atexit.register(self.shutdown)

    def add_task(self, key: str, task: Callable[[], None]) -> None:
        with self._lock:
            if key not in self._processing_keys:
                logger.debug(f"Adding skill cache refresh task for key: {key}")
                self._processing_keys.add(key)
                wrapped_task = self._wrap_task(key, task)
                self._queue.put((wrapped_task))
            else:
                logger.debug(
                    f"Skill cache refresh task already submitted for key: {key}"
                )

    def active_tasks(self) -> int:
        with self._lock:
            return len(self._processing_keys)

    def wait_for_idle(self) -> None:
        self._queue.join()

    def _wrap_task(self, key: str, task: Callable[[], None]) -> Callable[[], None]:
        def wrapped() -> None:
            logger.debug(f"Refreshing skill cache for key: {key}")
            try:
                task()
            finally:
                with self._lock:
                    self._processing_keys.remove(key)
                logger.debug(f"Refreshed skill cache for key: {key}")

        return wrapped

    def shutdown(self) -> None:
        logger.debug(
            f"Shutting down skill refresh task manager, {len(self._consumers)} consumers,..."
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

        logger.debug("Shutdown of skill refresh task manager completed.")


class SkillCache:
    _cache: Dict[str, SkillCacheItem]
    _lock: RLock

    _task_manager: SkillCacheTaskManager
    """Task manager for refreshing cache"""

    def __init__(
        self, max_skill_refresh_workers: int = DEFAULT_SKILL_CACHE_REFRESH_WORKERS
    ):
        self._cache = {}
        self._lock = RLock()
        self._task_manager = SkillCacheTaskManager(threads=max_skill_refresh_workers)
        logger.debug("Skill cache initialized.")

    def get(self, key: str) -> Optional[SkillCacheItem]:
        with self._lock:
            return self._cache.get(key, None)

    def set(self, key: str, value: SkillClient, ttl_seconds: Optional[int]) -> None:
        if ttl_seconds is None:
            ttl_seconds = DEFAULT_SKILL_CACHE_TTL_SECONDS

        with self._lock:
            self._cache[key] = SkillCacheItem(value, ttl_seconds)

    def delete(self, key: str) -> None:
        with self._lock:
            self._cache.pop(key, None)

    def invalidate(self, skill_name: str) -> None:
        """Invalidate all cached skills with the given skill name."""
        prefix = skill_name + "-"
        with self._lock:
            for key in list(self._cache):
                if key.startswith(prefix):
                    del self._cache[key]

    def add_refresh_skill_task(self, key: str, fetch_func: Callable[[], None]) -> None:
        logger.debug(f"Submitting refresh task for key: {key}")
        self._task_manager.add_task(key, fetch_func)

    def add_refresh_skill_task_if_current(
        self,
        key: str,
        expected_item: SkillCacheItem,
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

        self.add_refresh_skill_task(key, fetch_func)

    def clear(self) -> None:
        """Clear the entire skill cache, removing all cached skills."""
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
            # Default to production labeled skill
            parts.append("label:production")

        return "-".join(parts)
