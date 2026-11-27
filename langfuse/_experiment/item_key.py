"""@private Stable per-item identity used by experiment checkpoints.

The key derivation is shared with the ``experiment_item_id`` expression in
``langfuse._client.client`` so the two never diverge.
"""

from typing import Any, Dict, List, Optional

from langfuse._client.attributes import _serialize
from langfuse._client.utils import get_sha256_hash_hex
from langfuse.api import DatasetItem


def _hash_serialized(value: Any) -> str:
    """sha256 hex of the serialized value, treating None as the JSON null."""
    serialized = _serialize(value)
    if serialized is None:
        serialized = "null"
    return get_sha256_hash_hex(serialized)[:16]


def local_item_input_hash(input_data: Any) -> str:
    """First 16 hex chars of ``sha256(_serialize(input))``.

    This is the same value used as ``experiment_item_id`` for local dict items
    in ``Langfuse._process_experiment_item``. Centralizing it here keeps the
    checkpoint key and the span attribute in sync.
    """
    return _hash_serialized(input_data)


def _is_dataset_item(item: object) -> bool:
    return isinstance(item, DatasetItem) or (
        not isinstance(item, dict)
        and hasattr(item, "dataset_id")
        and hasattr(item, "id")
    )


def _dataset_item_id(item: object) -> Optional[str]:
    return getattr(item, "id", None)


def _item_input(item: object) -> Any:
    if isinstance(item, dict):
        return item.get("input")
    return getattr(item, "input", None)


def _item_expected_output(item: object) -> Any:
    if isinstance(item, dict):
        return item.get("expected_output")
    return getattr(item, "expected_output", None)


def _base_key(item: object, dataset_version: Optional[str]) -> str:
    """Compute the base (pre-disambiguation) key for one item."""
    if _is_dataset_item(item):
        item_id = _dataset_item_id(item) or ""
        if dataset_version is not None:
            return "%s@v%s" % (item_id, dataset_version)
        return item_id

    input_data = _item_input(item)
    base = local_item_input_hash(input_data)
    expected_output = _item_expected_output(item)
    # Always fold the expected-output hash in so two items with identical
    # inputs but different ground truth do not collide in the checkpoint.
    expected_hash = _hash_serialized(expected_output)
    return "%s:%s" % (base, expected_hash)


def make_item_keys(data: List[Any], dataset_version: Optional[str] = None) -> List[str]:
    """Return stable keys for every item in ``data``.

    Duplicate keys within the same ``data`` list are disambiguated by
    appending a deterministic ``#<occurrence>`` suffix (0-based) so the order
    of items fixes the suffix assignment.
    """
    base_keys: List[str] = [_base_key(item, dataset_version) for item in data]

    seen: Dict[str, int] = {}
    keys: List[str] = []
    for base in base_keys:
        count = seen.get(base, 0)
        seen[base] = count + 1
        if count == 0 and base not in keys:
            keys.append(base)
        else:
            keys.append("%s#%d" % (base, count))
    return keys


def experiment_item_id_for(item: object) -> str:
    """Return the ``experiment_item_id`` for an item.

    For dataset items this is the dataset item id; for local dict items this
    is the input hash. Mirrors the expression in
    ``Langfuse._process_experiment_item``.
    """
    if _is_dataset_item(item):
        return _dataset_item_id(item) or ""
    return local_item_input_hash(_item_input(item))


__all__ = [
    "local_item_input_hash",
    "make_item_keys",
    "experiment_item_id_for",
]
