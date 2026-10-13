import json
import os
import threading
import time
from queue import Empty, Full, Queue
from typing import Any, Dict, List, Optional, cast

import backoff
import httpx

from langfuse._utils.parse_error import handle_fern_exception
from langfuse._utils.serializer import EventSerializer
from langfuse.api import LangfuseAPI
from langfuse.api.core import ApiError
from langfuse.api.scores.types.create_score_batch_results import (
    CreateScoreBatchResults,
)
from langfuse.api.scores.types.create_scores_request import CreateScoresRequest
from langfuse.logger import langfuse_logger as logger

# POST /api/public/scores uses the default Next.js body limit of 1 MiB.
_SCORES_BODY_LIMIT_BYTES = 1_000_000
_configured_batch_size_bytes = int(
    os.environ.get("LANGFUSE_MAX_BATCH_SIZE_BYTES", _SCORES_BODY_LIMIT_BYTES)
)
if _configured_batch_size_bytes > _SCORES_BODY_LIMIT_BYTES:
    logger.warning(
        "Configuration: LANGFUSE_MAX_BATCH_SIZE_BYTES=%s exceeds the %s byte body "
        "limit of POST /api/public/scores; score batches are capped at %s bytes.",
        _configured_batch_size_bytes,
        _SCORES_BODY_LIMIT_BYTES,
        _SCORES_BODY_LIMIT_BYTES,
    )
MAX_BATCH_SIZE_BYTES = min(_configured_batch_size_bytes, _SCORES_BODY_LIMIT_BYTES)
_SHUTDOWN_SENTINEL = object()


def _is_retryable(error: Exception) -> bool:
    if isinstance(error, httpx.TransportError):
        return True

    if isinstance(error, ApiError):
        status = error.status_code
        return status is None or status >= 500 or status == 429

    return False


class ScoreConsumer(threading.Thread):
    """Send queued scores to POST /api/public/scores in batches."""

    def __init__(
        self,
        *,
        score_queue: Queue,
        identifier: int,
        api_client: LangfuseAPI,
        flush_at: Optional[int] = None,
        flush_interval: Optional[float] = None,
        max_retries: Optional[int] = None,
    ):
        """Create a consumer thread."""
        super().__init__()
        # It's important to set running in the constructor: if we are asked to
        # pause immediately after construction, we might set running to True in
        # run() *after* we set it to False in pause... and keep running
        # forever.
        self.running = True
        # Make consumer a daemon thread so that it doesn't block program exit
        self.daemon = True
        self._score_queue = score_queue
        self._identifier = identifier
        self._api_client = api_client
        self._flush_at = flush_at or 15
        self._flush_interval = flush_interval or 1
        self._max_retries = max_retries or 3
        # A score taken from the queue that did not fit into the previous batch.
        # It has not been acknowledged yet and starts the next batch.
        self._carry_over: Optional[Dict[str, Any]] = None

    def _next(self) -> List[Dict[str, Any]]:
        """Return the next batch of JSON-ready scores to upload."""
        scores: List[Dict[str, Any]] = []
        total_size = 2  # enclosing brackets of the JSON list

        if self._carry_over is not None:
            scores.append(self._carry_over)
            total_size += self._get_item_size(self._carry_over)
            self._carry_over = None

        start_time = time.monotonic()

        while len(scores) < self._flush_at:
            elapsed = time.monotonic() - start_time
            if elapsed >= self._flush_interval:
                break
            try:
                score = self._score_queue.get(
                    block=True, timeout=self._flush_interval - elapsed
                )

                if score is _SHUTDOWN_SENTINEL:
                    self._score_queue.task_done()
                    break

                try:
                    serialized = json.loads(json.dumps(score, cls=EventSerializer))
                except Exception as e:
                    logger.error(
                        "Data error: Failed to serialize score. Score will be "
                        "dropped. Error: %s",
                        e,
                    )
                    self._score_queue.task_done()

                    continue

                item_size = self._get_item_size(serialized)

                if item_size + 2 > MAX_BATCH_SIZE_BYTES:
                    logger.error(
                        "Data error: Score name=%s is %s bytes, which exceeds the "
                        "request limit of %s bytes. Score will be dropped.",
                        serialized.get("name"),
                        item_size,
                        MAX_BATCH_SIZE_BYTES,
                    )
                    self._score_queue.task_done()

                    continue

                # +1 for the separating comma
                if scores and total_size + item_size + 1 > MAX_BATCH_SIZE_BYTES:
                    logger.debug(
                        "Batch management: Reached maximum batch size limit (%s bytes). "
                        "Processing %s scores now.",
                        total_size,
                        len(scores),
                    )
                    self._carry_over = serialized
                    break

                scores.append(serialized)
                total_size += item_size + (1 if len(scores) > 1 else 0)

            except Empty:
                break

            except Exception as e:
                logger.warning(
                    "Data processing error: Failed to process score in consumer thread "
                    "#%s. Score will be dropped. Error: %s",
                    self._identifier,
                    str(e),
                    exc_info=True,
                )
                self._score_queue.task_done()

        return scores

    def _get_item_size(self, item: Any) -> int:
        """Return the size of the item in bytes."""
        return len(json.dumps(item).encode())

    def run(self) -> None:
        """Run the consumer."""
        logger.debug(
            "Startup: Score consumer thread #%s started with batch size %s "
            "and interval %ss",
            self._identifier,
            self._flush_at,
            self._flush_interval,
        )
        while self.running:
            self.upload()

        while self._carry_over is not None:
            self.upload()

    def upload(self) -> None:
        """Upload the next batch of scores."""
        batch = self._next()
        if len(batch) == 0:
            return

        try:
            self._upload_batch(batch)
        except ApiError as e:
            handle_fern_exception(e)
        except Exception as e:
            logger.error(
                "API error: Failed to send %s scores to Langfuse. Scores will be "
                "dropped. Error: %s",
                len(batch),
                e,
            )
        finally:
            # mark items as acknowledged from queue
            for _ in batch:
                self._score_queue.task_done()

    def pause(self) -> None:
        """Pause the consumer."""
        self.running = False
        try:
            self._score_queue.put(_SHUTDOWN_SENTINEL, block=False)
        except Full:
            # If the queue is full, the consumer will wake up naturally while
            # draining items, so a dedicated shutdown signal is not required.
            pass

    def _upload_batch(self, batch: List[Dict[str, Any]]) -> None:
        logger.debug("API: Sending batch of %s scores to Langfuse API", len(batch))

        @backoff.on_exception(
            backoff.expo,
            Exception,
            max_tries=self._max_retries,
            giveup=lambda e: not _is_retryable(e),
            logger=None,
        )
        def send_with_backoff() -> Any:
            # The items are JSON-ready dicts with API field names. Unlike the
            # generated request model, the API accepts any JSON value as score
            # metadata, so the dicts are sent as they are.
            return self._api_client.scores.create(
                request=cast(CreateScoresRequest, batch),
                # Retries are handled here so that transport errors are retried too.
                request_options={"max_retries": 0},
            )

        response = send_with_backoff()

        if isinstance(response, CreateScoreBatchResults):
            logger.error(
                "API error: Langfuse accepted %s and rejected %s of %s scores. "
                "Rejected scores are not retried. Errors: %s",
                response.accepted,
                response.rejected,
                len(batch),
                "; ".join(error.message for error in response.errors),
            )
            return

        logger.debug("API: Successfully sent %s scores to Langfuse API", len(batch))
