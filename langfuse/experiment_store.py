"""Storage / transport layer for resumable experiment runs."""

from __future__ import annotations

from typing import (
    Any,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

from langfuse.experiment_resume import ClientError, ItemResult, ServerError


# --- HTTP client ------------------------------------
class HttpResultStore:
    """Client for the experiment-run endpoints. Classifies responses into
    retryable (ServerError) / non-retryable (ClientError); the runner owns retry.
    Uses HTTP Basic auth (public_key / secret_key), like the rest of the SDK."""

    def __init__(self, base_url: str, public_key: str, secret_key: str, *,
                 project: str = "default", timeout: float = 30.0):
        import httpx

        self._httpx = httpx
        self._base = base_url.rstrip("/") + "/api/public/v2/experiment-runs"
        self._project = project
        self._client = httpx.AsyncClient(timeout=timeout, auth=(public_key, secret_key))

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _post(self, path: str, body: Dict[str, Any]) -> Any:
        """One request, no retry — the runner owns retry. Classify the outcome:
        429/5xx and transport errors -> ServerError (retryable); any other 4xx
        -> ClientError (fail fast, since retrying it will never succeed)."""
        try:
            resp = await self._client.post(self._base + path, json=body)
        except self._httpx.HTTPError as e:  # connection/transport failure
            raise ServerError(f"transport error on {path}: {e}") from e
        if resp.status_code == 429 or resp.status_code >= 500:
            raise ServerError(f"{resp.status_code} from {path}: {resp.text[:200]}")
        if resp.status_code >= 400:
            raise ClientError(f"{resp.status_code} from {path}: {resp.text[:200]}")
        return resp.json()

    async def create_or_resume_run(self, *, name: str, run_name: Optional[str],
                                   dataset_ref: Optional[Dict[str, Any]], fingerprint: str,
                                   config: Optional[Dict[str, Any]]) -> Tuple[str, str]:
        out = await self._post("", {"project": self._project, "name": name, "run_name": run_name,
                                    "dataset_ref": dataset_ref, "fingerprint": fingerprint,
                                    "config_snapshot": config})
        return out["run_id"], out["status"]

    async def gap_query(self, run_id: str, item_ids: Sequence[str]) -> Dict[str, str]:
        out = await self._post(f"/{run_id}/items:gap", {"item_ids": list(item_ids)})
        return {k["item_id"]: k["state"] for k in out.get("known", [])}

    async def upsert_items(self, run_id: str, results: Sequence[ItemResult]) -> List[Dict[str, str]]:
        out = await self._post(f"/{run_id}/items:batch", {"items": [r.to_wire() for r in results]})
        return list(out.get("results", []))

    async def upsert_run_evaluations(self, run_id: str, evaluations: List[Dict[str, Any]]) -> None:
        await self._post(f"/{run_id}/run-evaluations", {"evaluations": evaluations})

    async def complete_run(self, run_id: str, status: str) -> Dict[str, Any]:
        out = await self._post(f"/{run_id}:complete", {"status": status})
        summary: Dict[str, Any] = out["summary"]
        return summary



# --- aggregation (computed from item rows, never stored as truth) ---
def summarize(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Counts + per-metric aggregates + failure list, computed from item rows
    (never stored as truth)."""
    succeeded = failed = timed_out = 0
    sums: Dict[str, float] = {}
    ns: Dict[str, int] = {}
    failures: List[Dict[str, Any]] = []
    for r in rows:
        state = r["state"]
        if state == "succeeded":
            succeeded += 1
            for ev in r.get("evaluations") or []:
                v = ev.get("value")
                if isinstance(v, bool):  # pass/fail booleans aggregate as 1.0/0.0
                    v = 1.0 if v else 0.0
                if isinstance(v, (int, float)):
                    sums[ev["name"]] = sums.get(ev["name"], 0.0) + float(v)
                    ns[ev["name"]] = ns.get(ev["name"], 0) + 1
        elif state == "timed_out":
            timed_out += 1
            failures.append({"item_id": r["item_id"], "reason": r.get("error") or "timeout", "attempts": r.get("attempts", 0)})
        else:
            failed += 1
            failures.append({"item_id": r["item_id"], "reason": r.get("error") or "error", "attempts": r.get("attempts", 0)})
    scores = {name: {"mean": sums[name] / ns[name], "n": ns[name]} for name in sums}
    return {"succeeded": succeeded, "failed": failed, "timed_out": timed_out,
            "rows": len(rows), "scores": scores, "failures": failures}
