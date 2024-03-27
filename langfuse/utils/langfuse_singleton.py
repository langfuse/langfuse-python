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

    def get(
        self,
        public_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        host: Optional[str] = None,
        debug: bool = False,
        sdk_integration: Optional[str] = None,
    ) -> Langfuse:
        if self._langfuse:
            return self._langfuse

        with self._lock:
            if self._langfuse:
                return self._langfuse

            self._langfuse = Langfuse(
                public_key=public_key,
                secret_key=secret_key,
                host=host,
                debug=debug,
                sdk_integration=sdk_integration,
            )

            return self._langfuse

    def reset(self) -> None:
        with self._lock:
            self._langfuse = None
