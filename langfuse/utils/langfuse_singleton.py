import os
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

            public_key = os.environ.get("LANGFUSE_PUBLIC_KEY")
            secret_key = os.environ.get("LANGFUSE_SECRET_KEY")
            host = os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com")

            if public_key and secret_key:
                self._langfuse = Langfuse(
                    public_key=public_key, secret_key=secret_key, host=host
                )

                return self._langfuse

            else:
                raise ValueError(
                    "Missing LANGFUSE_SECRET_KEY or LANGFUSE_PUBLIC_KEY environment variables"
                )
