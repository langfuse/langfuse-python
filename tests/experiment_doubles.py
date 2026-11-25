"""Test doubles for the resumable experiment runner (langfuse.experiment_resume).

NOT shipped with the SDK — these live under tests/ so the runtime module carries
only the real store (HttpResultStore). Imported by test_experiment_v2.py.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional

from langfuse.experiment_resume import ResultStore, ServerError, derive_run_id
from langfuse.experiment_store import summarize


class InMemoryResultStore:
    def __init__(self) -> None:
        self.runs: Dict[str, Dict[str, Any]] = {}
        self.results: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self.run_evaluations: Dict[str, List[Dict[str, Any]]] = {}

    async def create_or_resume_run(self, *, name, run_name, dataset_ref, fingerprint, config):
        run_id = derive_run_id(name=name, run_name=run_name, fingerprint=fingerprint)
        if run_id in self.runs:
            return run_id, "resumed"
        self.runs[run_id] = {"name": name, "run_name": run_name, "fingerprint": fingerprint,
                             "dataset_ref": dataset_ref, "config": config, "status": "open"}
        self.results[run_id] = {}
        self.run_evaluations[run_id] = []
        return run_id, "created"

    async def gap_query(self, run_id, item_ids):
        have = self.results.get(run_id, {})
        return {iid: have[iid]["state"] for iid in item_ids if iid in have}

    async def upsert_items(self, run_id, results):
        bucket = self.results.setdefault(run_id, {})
        applied: List[Dict[str, str]] = []
        for res in results:
            wire = res.to_wire()
            existing = bucket.get(res.item_id)
            if existing == wire:
                applied.append({"item_id": res.item_id, "applied": "noop_duplicate"})
                continue
            verb = "updated" if existing is not None else "created"
            bucket[res.item_id] = wire
            applied.append({"item_id": res.item_id, "applied": verb})
        return applied

    async def upsert_run_evaluations(self, run_id, evaluations):
        self.run_evaluations[run_id] = list(evaluations)

    async def complete_run(self, run_id, status):
        self.runs[run_id]["status"] = status
        summary = summarize(list(self.results.get(run_id, {}).values()))
        summary["status"] = status
        summary["run_evaluations"] = self.run_evaluations.get(run_id, [])
        return summary


class FlakyStore:
    """Wraps a store and raises on writes with probability fail_rate, writing
    through first — mimics at-least-once so the retry must be idempotent."""

    def __init__(self, inner: ResultStore, *, fail_rate: float = 0.5, seed: Optional[int] = None):
        self.inner = inner
        self.fail_rate = fail_rate
        self._rng = random.Random(seed)
        self.injected = 0

    async def create_or_resume_run(self, **kwargs):
        return await self.inner.create_or_resume_run(**kwargs)

    async def gap_query(self, run_id, item_ids):
        return await self.inner.gap_query(run_id, item_ids)

    async def upsert_items(self, run_id, results):
        if self._rng.random() < self.fail_rate:
            self.injected += 1
            await self.inner.upsert_items(run_id, results)
            raise ServerError("injected transport failure on upsert")
        return await self.inner.upsert_items(run_id, results)

    async def upsert_run_evaluations(self, run_id, evaluations):
        return await self.inner.upsert_run_evaluations(run_id, evaluations)

    async def complete_run(self, run_id, status):
        return await self.inner.complete_run(run_id, status)

