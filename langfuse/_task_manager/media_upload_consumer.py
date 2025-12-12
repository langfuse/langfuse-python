import logging
import threading
import time

from .media_manager import MediaManager


class MediaUploadConsumer(threading.Thread):
    _log = logging.getLogger("langfuse")
    _identifier: int
    _max_retries: int
    _media_manager: MediaManager

    def __init__(
        self,
        *,
        identifier: int,
        media_manager: MediaManager,
    ):
        """Create a consumer thread."""
        super().__init__()
        # Make consumer a daemon thread so that it doesn't block program exit
        self.daemon = True
        # It's important to set running in the constructor: if we are asked to
        # pause immediately after construction, we might set running to True in
        # run() *after* we set it to False in pause... and keep running
        # forever.
        self.running = True
        # Track when thread last processed something
        self.last_activity = time.time()
        self._identifier = identifier
        self._media_manager = media_manager

    def run(self) -> None:
        """Run the media upload consumer."""
        self._log.debug(
            f"Thread: Media upload consumer thread #{self._identifier} started and actively processing queue items"
        )
        while self.running:
            try:
                # Update activity timestamp before processing
                self.last_activity = time.time()
                self._media_manager.process_next_media_upload()
                # Update after successful processing
                self.last_activity = time.time()
            except Exception as e:
                self._log.error(
                    f"Thread #{self._identifier}: Unexpected error in consumer loop: {e}"
                )
                # Continue running despite errors
                time.sleep(0.1)

    def pause(self) -> None:
        """Pause the media upload consumer."""
        self._log.debug(
            f"Thread: Pausing media upload consumer thread #{self._identifier}"
        )
        self.running = False

    def is_healthy(self, timeout_seconds: float = 5.0) -> bool:
        """
        Check if thread is healthy and recently active.
        Returns False if thread hasn't processed anything in timeout_seconds.
        """
        if not self.is_alive():
            return False

        time_since_activity = time.time() - self.last_activity
        return time_since_activity < timeout_seconds
