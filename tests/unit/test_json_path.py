import pytest

from langfuse._utils.json_path import parse_path, set_value_at_path

_REF = "@@@langfuseMedia:type=image/png|id=abc|source=bytes@@@"

# (description, value, json_path, expected) generated from jsonpath-plus 10.x — the
# same library the backend uses to produce the jsonPath. `json_path` is what
# jsonpath-plus emits for the reference's location; `expected` is the value after
# jsonpath-plus sets a sentinel there, so this verifies our setter matches the
# library byte-for-byte.
_JSONPATH_PLUS_CASES = [
    ("root", _REF, "$", "__MEDIA__"),
    ("simple key", {"image": _REF}, "$['image']", {"image": "__MEDIA__"}),
    ("key with space", {"my image": _REF}, "$['my image']", {"my image": "__MEDIA__"}),
    ("apostrophe key", {"O'connor": _REF}, "$['O'connor']", {"O'connor": "__MEDIA__"}),
    ("double-quote key", {'a"b': _REF}, "$['a\"b']", {'a"b': "__MEDIA__"}),
    ("bracket key", {"arr[0]": _REF}, "$['arr[0]']", {"arr[0]": "__MEDIA__"}),
    ("dot key", {"a.b": _REF}, "$['a.b']", {"a.b": "__MEDIA__"}),
    ("list root", [_REF], "$[0]", ["__MEDIA__"]),
    (
        "list element",
        {"items": [0, _REF, 2]},
        "$['items'][1]",
        {"items": [0, "__MEDIA__", 2]},
    ),
    (
        "nested obj",
        {"a": {"b": {"c": _REF}}},
        "$['a']['b']['c']",
        {"a": {"b": {"c": "__MEDIA__"}}},
    ),
    ("obj in list", [{"x": _REF}], "$[0]['x']", [{"x": "__MEDIA__"}]),
    ("two indices", [[_REF]], "$[0][0]", [["__MEDIA__"]]),
    ("three indices", [[[_REF]]], "$[0][0][0]", [[["__MEDIA__"]]]),
    (
        "key then two indices",
        {"matrix": [[0, _REF]]},
        "$['matrix'][0][1]",
        {"matrix": [[0, "__MEDIA__"]]},
    ),
    (
        "index key index",
        [{"rows": [_REF]}],
        "$[0]['rows'][0]",
        [{"rows": ["__MEDIA__"]}],
    ),
    (
        "deep mixed",
        {"messages": [{"content": [{"image_url": _REF}]}]},
        "$['messages'][0]['content'][0]['image_url']",
        {"messages": [{"content": [{"image_url": "__MEDIA__"}]}]},
    ),
    (
        "sibling untouched",
        {"keep": "txt", "img": _REF},
        "$['img']",
        {"keep": "txt", "img": "__MEDIA__"},
    ),
]


@pytest.mark.parametrize(
    ("value", "json_path", "expected"),
    [(c[1], c[2], c[3]) for c in _JSONPATH_PLUS_CASES],
    ids=[c[0] for c in _JSONPATH_PLUS_CASES],
)
def test_set_value_at_path_matches_jsonpath_plus(value, json_path, expected):
    assert set_value_at_path(value, json_path, "__MEDIA__") == expected


@pytest.mark.parametrize(
    ("value", "json_path"),
    [
        # All-digit keys and keys containing "']" are indistinguishable / broken in
        # jsonpath-plus' output, so they cannot be resolved and must raise (the
        # caller leaves the value unchanged rather than guessing).
        ({"0": _REF}, "$[0]"),
        ({"a']b": _REF}, "$['a']b']"),
        # Malformed paths the API should never emit.
        ("x", "image"),
        ({"a": _REF}, "$['a'"),
        ({"a": _REF}, "$[a]"),
    ],
)
def test_set_value_at_path_raises_on_unresolvable(value, json_path):
    with pytest.raises(Exception):
        set_value_at_path(value, json_path, "__MEDIA__")


def test_parse_path_segments():
    assert parse_path("$") == []
    assert parse_path("$['image']") == ["image"]
    assert parse_path("$[0]") == [0]
    assert parse_path("$['a']['b'][2]") == ["a", "b", 2]
    assert parse_path("$[0][1][2]") == [0, 1, 2]
    assert parse_path("$['O'connor']") == ["O'connor"]
