"""Regression tests for issue #1655 — cycle detection on the dict / list /
Sequence / __slots__ branches of EventSerializer. Pre-fix these branches
recursed forever, the RecursionError was swallowed inside GC, and the
asyncio loop was GIL-starved for minutes."""

import json
import time
from collections.abc import Sequence

from langfuse._utils.serializer import EventSerializer


def test_dict_self_cycle_does_not_recurse():
    d = {"a": 1}
    d["self"] = d

    parsed = json.loads(EventSerializer().encode(d))

    assert parsed["a"] == 1
    assert "cycle" in parsed["self"]
    assert "dict" in parsed["self"]


def test_list_self_cycle_does_not_recurse():
    lst = [1, 2, 3]
    lst.append(lst)

    parsed = json.loads(EventSerializer().encode(lst))

    assert parsed[:3] == [1, 2, 3]
    assert "cycle" in parsed[3]
    assert "list" in parsed[3]


def test_slots_self_cycle_does_not_recurse():
    """__slots__-only object (no __dict__) with a self-referential slot."""

    class SlotsObj:
        __slots__ = ("name", "ref")

        def __init__(self):
            self.name = "outer"
            self.ref = None

    obj = SlotsObj()
    obj.ref = obj

    parsed = json.loads(EventSerializer().encode(obj))

    assert parsed["name"] == "outer"
    assert "cycle" in parsed["ref"]


def test_cycle_encode_completes_quickly():
    """Pre-fix this hung the interpreter at the recursion limit while the
    asyncio loop sat GIL-starved. With the guard it returns instantly."""
    d = {"a": 1}
    d["self"] = d

    start = time.monotonic()
    EventSerializer().encode(d)
    elapsed = time.monotonic() - start

    assert elapsed < 1.0, f"encode took {elapsed:.3f}s — cycle detection regressed"


def test_mutual_dict_cycle():
    """a -> b -> a — cycle straddles two distinct dict ids."""
    a = {}
    b = {}
    a["b"] = b
    b["a"] = a

    encoded = EventSerializer().encode(a)
    json.loads(encoded)
    assert "cycle" in encoded


def test_mixed_container_cycle_dict_through_list():
    """dict -> list -> dict_self — both the dict and list guards must fire."""
    d = {}
    d["children"] = [d]

    encoded = EventSerializer().encode(d)
    json.loads(encoded)
    assert "cycle" in encoded


def test_object_cycle_through_dict_attribute():
    """__dict__ object whose attribute is a cyclic dict — the __dict__
    branch hands off to the new dict branch's guard."""

    class Holder:
        def __init__(self):
            self.cfg = {"holder": None}
            self.cfg["holder"] = self.cfg

    parsed = json.loads(EventSerializer().encode(Holder()))

    assert "holder" in parsed["cfg"]
    assert "cycle" in parsed["cfg"]["holder"]


def test_shared_dict_in_siblings_is_not_marked_as_cycle():
    """DAG, not cycle. try/finally + discard is what makes this pass —
    a stay-forever `seen` set would falsely flag the second visit."""
    shared = {"value": 42}
    container = {"left": shared, "right": shared}

    encoded = EventSerializer().encode(container)
    parsed = json.loads(encoded)

    assert parsed["left"] == {"value": 42}
    assert parsed["right"] == {"value": 42}
    assert "cycle" not in encoded


def test_shared_list_in_siblings_is_not_marked_as_cycle():
    shared = [1, 2, 3]
    container = {"first": shared, "second": shared}

    encoded = EventSerializer().encode(container)
    parsed = json.loads(encoded)

    assert parsed["first"] == [1, 2, 3]
    assert parsed["second"] == [1, 2, 3]
    assert "cycle" not in encoded


def test_deeply_nested_non_cyclic_serialises_fully():
    """50-level linear chain — depth alone must not trip the guard."""
    cur = {"depth": 50}
    for i in range(49, 0, -1):
        cur = {"depth": i, "child": cur}

    parsed = json.loads(EventSerializer().encode(cur))

    walked = parsed
    for expected in range(1, 51):
        assert walked["depth"] == expected
        walked = walked.get("child", walked)


def test_dict_attribute_self_cycle_preserves_existing_marker():
    """Back-compat: the existing __dict__ branch marker is the bare class
    name, not the new <cycle:Type> form. Left untouched — the fix is
    purely additive on the previously unprotected branches."""

    class Node:
        def __init__(self, name):
            self.name = name
            self.parent = None

    node = Node("root")
    node.parent = node

    parsed = json.loads(EventSerializer().encode(node))

    assert parsed["name"] == "root"
    assert parsed["parent"] == "Node"


def test_encode_called_twice_does_not_leak_seen_state():
    """encode() clears `seen` at the top; the new try/finally also
    discards mid-walk. Two back-to-back cyclic encodes on the same
    instance must both succeed independently."""
    serializer = EventSerializer()

    first = {"x": 1}
    first["self"] = first
    second = {"y": 2}
    second["self"] = second

    parsed_first = json.loads(serializer.encode(first))
    parsed_second = json.loads(serializer.encode(second))

    assert parsed_first["x"] == 1 and "cycle" in parsed_first["self"]
    assert parsed_second["y"] == 2 and "cycle" in parsed_second["self"]


class _CycleSequence(Sequence):
    """Minimal user-defined Sequence (not a list) — exercises the
    Sequence-only branch's new guard."""

    def __init__(self):
        self._items = []

    def __getitem__(self, index):
        return self._items[index]

    def __len__(self):
        return len(self._items)

    def append(self, item):
        self._items.append(item)


def test_custom_sequence_self_cycle_does_not_recurse():
    seq = _CycleSequence()
    seq.append("x")
    seq.append(seq)

    # _CycleSequence has __dict__, so the __dict__ branch is hit first;
    # either way the encode must terminate and produce valid JSON.
    json.loads(EventSerializer().encode(seq))


def test_empty_containers_unchanged():
    assert EventSerializer().encode({}) == "{}"
    assert EventSerializer().encode([]) == "[]"


def test_non_cyclic_dict_passthrough():
    data = {"name": "alice", "age": 30, "tags": ["a", "b"]}
    assert json.loads(EventSerializer().encode(data)) == data
