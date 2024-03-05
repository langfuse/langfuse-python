import threading
from typing import Optional


from langfuse import Langfuse


class LangfuseSingleton:
    _instance = None
    _lock = threading.Lock()
    _langfuse: Optional[Langfuse] = None

    def __new__(cls):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super(LangfuseSingleton, cls).__new__(cls)
        return cls._instance

    def get(self) -> Langfuse:
        if self._langfuse:
            return self._langfuse

        with self._lock:
            if self._langfuse:
                return self._langfuse

            self._langfuse = Langfuse()

            return self._langfuse
