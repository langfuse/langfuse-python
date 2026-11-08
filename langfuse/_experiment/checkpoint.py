"""@private Checkpoint persistence for resumable experiment runs.

Checkpoints record which items already completed (or permanently failed) so
a crashed or interrupted run can resume without re-paying for completed
task calls. The on-disk format is JSON with a sha256 integrity footer.
"""

import json
import os
import re
import tempfile
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from langfuse._client.utils import get_sha256_hash_hex
from langfuse._experiment.errors import (
    CheckpointCorruptedError,
    CheckpointMismatchError,
)

_SCHEMA_VERSION = 1
_INTEGRITY_KEY = "__integrity_sha256__"

# Filenames are built only from [A-Za-z0-9_-]: anything else in the key
# (including dots, which could form ".." traversal sequences) is replaced.
_UNSAFE_KEY_RE = re.compile(r"[^A-Za-z0-9_-]")


def experiment_checkpoint_key(experiment_name: str, run_name: str) -> str:
    """Deterministic checkpoint key for runs whose ``experiment_id`` is random.

    Dataset runs derive a stable ``experiment_id`` from project, dataset and
    run name, so the id itself is the checkpoint key. Local (non-dataset)
    runs get a fresh random id on every call, so the key is derived from the
    experiment name and run name instead; resuming such a run therefore
    requires the caller to pass an explicit ``run_name``.
    """
    payload = json.dumps(
        ["langfuse-experiment-checkpoint-v1", experiment_name, run_name],
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return get_sha256_hash_hex(payload)[:32]


def _sanitize_key(key: str) -> str:
    """Return a filesystem-safe filename for ``key``.

    Every character outside ``[A-Za-z0-9_-]`` is replaced with ``_``, so
    traversal sequences (``..``) and separators can never appear in the
    result. The value is prefixed with a short hash of the raw key so
    distinct keys cannot collide after sanitization. Keys containing NUL
    are rejected outright.
    """
    if "\x00" in key:
        message = "checkpoint key must not contain NUL bytes: %r" % key
        raise ValueError(message)
    safe = _UNSAFE_KEY_RE.sub("_", key)
    digest = get_sha256_hash_hex(key)[:16]
    return "%s--%s.json" % (digest, safe)


@dataclass
class CheckpointItem:
    """One item's persisted state."""

    state: str  # "completed" | "failed"
    output_json: Optional[str] = None
    evaluations: List[Dict[str, Any]] = field(default_factory=list)
    trace_id: Optional[str] = None
    attempts: int = 0
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "state": self.state,
            "output_json": self.output_json,
            "evaluations": list(self.evaluations),
            "trace_id": self.trace_id,
            "attempts": self.attempts,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CheckpointItem":
        return cls(
            state=str(data.get("state", "failed")),
            output_json=data.get("output_json"),
            evaluations=list(data.get("evaluations") or []),
            trace_id=data.get("trace_id"),
            attempts=int(data.get("attempts") or 0),
            error=data.get("error"),
        )


@dataclass
class ExperimentFingerprint:
    """Identity of the task+evaluator set a checkpoint belongs to."""

    task_name: str
    evaluator_names: List[str]
    data_size: int
    data_digest: str  # sha256 over sorted item keys

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_name": self.task_name,
            "evaluator_names": list(self.evaluator_names),
            "data_size": self.data_size,
            "data_digest": self.data_digest,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentFingerprint":
        return cls(
            task_name=str(data.get("task_name") or ""),
            evaluator_names=list(data.get("evaluator_names") or []),
            data_size=int(data.get("data_size") or 0),
            data_digest=str(data.get("data_digest") or ""),
        )

    def matches(self, other: "ExperimentFingerprint") -> bool:
        return self.to_dict() == other.to_dict()


def compute_data_digest(item_keys: List[str]) -> str:
    joined = "\n".join(sorted(item_keys))
    return get_sha256_hash_hex(joined)


@dataclass
class ExperimentCheckpoint:
    """Full persisted state of one experiment run."""

    schema_version: int = _SCHEMA_VERSION
    experiment_id: str = ""
    experiment_name: str = ""
    run_name: str = ""
    fingerprint: Optional[ExperimentFingerprint] = None
    items: Dict[str, CheckpointItem] = field(default_factory=dict)
    created_at: float = 0.0
    updated_at: float = 0.0
    status: str = "running"  # "running" | "completed" | "aborted"

    def to_payload(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "experiment_id": self.experiment_id,
            "experiment_name": self.experiment_name,
            "run_name": self.run_name,
            "fingerprint": (self.fingerprint.to_dict() if self.fingerprint else None),
            "items": {k: v.to_dict() for k, v in self.items.items()},
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "status": self.status,
        }

    def serialize(self) -> str:
        payload = self.to_payload()
        integrity = get_sha256_hash_hex(
            json.dumps(payload, sort_keys=True, ensure_ascii=False)
        )
        payload[_INTEGRITY_KEY] = integrity
        return json.dumps(payload, ensure_ascii=False)

    @classmethod
    def deserialize(cls, raw: str) -> "ExperimentCheckpoint":
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, TypeError) as e:
            raise CheckpointCorruptedError("invalid JSON: %s" % e) from e
        if not isinstance(data, dict):
            raise CheckpointCorruptedError("top-level value is not an object")
        stored = data.pop(_INTEGRITY_KEY, None)
        if stored is None:
            raise CheckpointCorruptedError("missing integrity hash")
        recomputed = get_sha256_hash_hex(
            json.dumps(data, sort_keys=True, ensure_ascii=False)
        )
        if not isinstance(stored, str) or stored != recomputed:
            raise CheckpointCorruptedError("integrity hash mismatch")
        schema_version = int(data.get("schema_version") or 0)
        if schema_version != _SCHEMA_VERSION:
            raise CheckpointCorruptedError(
                "unsupported schema version %s" % schema_version
            )
        fingerprint_data = data.get("fingerprint")
        fingerprint = (
            ExperimentFingerprint.from_dict(fingerprint_data)
            if fingerprint_data
            else None
        )
        items_raw = data.get("items") or {}
        items = {str(k): CheckpointItem.from_dict(v) for k, v in items_raw.items()}
        return cls(
            schema_version=schema_version,
            experiment_id=str(data.get("experiment_id") or ""),
            experiment_name=str(data.get("experiment_name") or ""),
            run_name=str(data.get("run_name") or ""),
            fingerprint=fingerprint,
            items=items,
            created_at=float(data.get("created_at") or 0.0),
            updated_at=float(data.get("updated_at") or 0.0),
            status=str(data.get("status") or "running"),
        )

    def verify_fingerprint(self, expected: ExperimentFingerprint) -> None:
        if self.fingerprint is None:
            raise CheckpointMismatchError("fingerprint", expected.to_dict(), None)
        if not self.fingerprint.matches(expected):
            raise CheckpointMismatchError(
                "fingerprint", expected.to_dict(), self.fingerprint.to_dict()
            )


@runtime_checkable
class CheckpointStore(Protocol):
    """Protocol for pluggable checkpoint storage backends.

    ``experiment_key`` is the same key passed to :meth:`load`, so a store
    never has to guess the lookup key from the checkpoint payload (for local
    data the key differs from ``checkpoint.experiment_id``).
    """

    def load(self, experiment_key: str) -> Optional[ExperimentCheckpoint]: ...

    def save(self, checkpoint: ExperimentCheckpoint, experiment_key: str) -> None: ...

    def delete(self, experiment_key: str) -> None: ...


class InMemoryCheckpointStore:
    """In-process checkpoint store, useful for tests."""

    def __init__(self) -> None:
        self._store: Dict[str, ExperimentCheckpoint] = {}
        self._lock = threading.Lock()

    def load(self, experiment_key: str) -> Optional[ExperimentCheckpoint]:
        with self._lock:
            return self._store.get(experiment_key)

    def save(self, checkpoint: ExperimentCheckpoint, experiment_key: str) -> None:
        with self._lock:
            self._store[experiment_key] = checkpoint

    def delete(self, experiment_key: str) -> None:
        with self._lock:
            self._store.pop(experiment_key, None)


class FileCheckpointStore:
    """On-disk checkpoint store with atomic writes.

    Writes are debounced to at most one write per ``flush_interval_s`` seconds
    (plus a final write on close/abort). Writes run on a worker thread via
    ``asyncio.to_thread`` so the event loop is never blocked.
    """

    def __init__(self, directory: str, flush_interval_s: float = 1.0) -> None:
        self._directory = directory
        self._flush_interval_s = flush_interval_s
        self._lock = threading.Lock()
        self._last_flush: float = 0.0
        os.makedirs(directory, exist_ok=True)

    def _path_for(self, experiment_key: str) -> str:
        return os.path.join(self._directory, _sanitize_key(experiment_key))

    def load(self, experiment_key: str) -> Optional[ExperimentCheckpoint]:
        path = self._path_for(experiment_key)
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw = f.read()
        except OSError:
            return None
        return ExperimentCheckpoint.deserialize(raw)

    def save(self, checkpoint: ExperimentCheckpoint, experiment_key: str) -> None:
        """Write ``checkpoint`` atomically to disk under ``experiment_key``.

        ``save`` performs a full write every call; debouncing is the caller's
        responsibility (the executor calls it at most once per
        ``flush_interval_s``). ``experiment_key`` must be the same key the
        store is later loaded with.
        """
        path = self._path_for(experiment_key)
        with self._lock:
            self._write_atomic(path, checkpoint.serialize())

    @staticmethod
    def _write_atomic(path: str, data: str) -> None:
        directory = os.path.dirname(path) or "."
        fd, tmp_path = tempfile.mkstemp(prefix=".cp-", suffix=".tmp", dir=directory)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, path)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    def delete(self, experiment_key: str) -> None:
        path = self._path_for(experiment_key)
        with self._lock:
            if os.path.exists(path):
                try:
                    os.unlink(path)
                except OSError:
                    pass

    def should_flush(self, now: float) -> bool:
        return (now - self._last_flush) >= self._flush_interval_s

    def mark_flushed(self, now: float) -> None:
        self._last_flush = now


__all__ = [
    "CheckpointItem",
    "ExperimentFingerprint",
    "ExperimentCheckpoint",
    "CheckpointStore",
    "InMemoryCheckpointStore",
    "FileCheckpointStore",
    "compute_data_digest",
    "experiment_checkpoint_key",
]
