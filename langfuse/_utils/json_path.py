"""Resolve the JSONPaths the Langfuse API attaches to dataset item media references.

The backend's ``findMediaReferences`` reads ``node.path`` from jsonpath-plus'
``JSONPath(..., resultType="all")``, which returns a bracket-normalized path. So we
only ever see ``$``, ``['<key>']`` (single-quoted, with no escaping — keys may
contain literal quotes, brackets, dots, etc.), and ``[<int>]`` (also emitted for
all-digit object keys). It is always bracket notation — never dot notation like
``$.x.y`` — i.e. an RFC 9535 normalized path. We parse exactly this restricted
grammar rather than depend on a full JSONPath engine. ``findMediaReferences`` is the
format of record; anything outside the grammar raises.
"""

from typing import Any, List, Union


def parse_path(json_path: str) -> List[Union[str, int]]:
    """Parse a jsonpath-plus normalized path into ordered segments.

    Object keys become ``str`` segments, array indices become ``int`` segments.
    Returns an empty list for the root ``$``.
    """
    if not json_path.startswith("$"):
        raise ValueError(json_path)

    segments: List[Union[str, int]] = []
    i, n = 1, len(json_path)
    while i < n:
        if json_path[i] != "[":
            raise ValueError(json_path)
        i += 1

        if i < n and json_path[i] == "'":  # object key: ['<key>'] (single-quoted)
            i += 1
            # No escaping, so the key ends at the closing "']".
            close = json_path.find("']", i)
            if close == -1:
                raise ValueError(json_path)
            segments.append(json_path[i:close])
            i = close + 2
        else:  # array index: [<int>]
            start = i
            while i < n and json_path[i].isdigit():
                i += 1
            if i == start or i >= n or json_path[i] != "]":
                raise ValueError(json_path)
            segments.append(int(json_path[start:i]))
            i += 1

    return segments


def set_value_at_path(value: Any, json_path: str, replacement: Any) -> Any:
    """Replace the node at ``json_path`` within ``value`` with ``replacement``.

    Mutates ``value`` in place and returns it; for the root path ``$`` it returns
    ``replacement`` directly. Raises if the path can't be parsed or navigated.
    """
    segments = parse_path(json_path)
    if not segments:  # "$": the whole value is the reference
        return replacement

    target = value
    for segment in segments[:-1]:
        target = target[segment]

    leaf = segments[-1]
    # JSON object keys are always strings, so an int leaf on a dict is an
    # all-digit key that jsonpath-plus rendered ambiguously as "[0]". We can't
    # tell it from a list index, so raise rather than add a bogus int key.
    if isinstance(leaf, int) and isinstance(target, dict):
        raise KeyError(json_path)
    target[leaf] = replacement

    return value
