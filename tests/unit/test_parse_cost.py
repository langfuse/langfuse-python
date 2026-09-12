from langfuse.openai import _parse_cost


class _Usage:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def test_none_usage():
    assert _parse_cost(None) is None


def test_object_with_float_cost():
    assert _parse_cost(_Usage(cost=0.0021)) == {"total": 0.0021}


def test_object_without_cost():
    assert _parse_cost(_Usage(prompt_tokens=10)) is None


def test_dict_with_float_cost():
    """OpenRouter usage arrives as a dict on the streaming path.

    _parse_usage already branches on isinstance(usage, dict) and both are called
    with the same object, so cost has to handle a dict too.
    """
    assert _parse_cost({"cost": 0.0021, "prompt_tokens": 10}) == {"total": 0.0021}


def test_dict_without_cost():
    assert _parse_cost({"prompt_tokens": 10}) is None


def test_integer_cost_is_not_dropped():
    """A free or fully cached model reports "cost": 0, which is an int, not a float."""
    assert _parse_cost({"cost": 0}) == {"total": 0.0}
    assert _parse_cost(_Usage(cost=0)) == {"total": 0.0}
    assert _parse_cost(_Usage(cost=2)) == {"total": 2.0}


def test_bool_cost_is_rejected():
    assert _parse_cost({"cost": True}) is None
    assert _parse_cost(_Usage(cost=False)) is None


def test_non_numeric_cost_is_rejected():
    assert _parse_cost({"cost": "0.0021"}) is None
    assert _parse_cost({"cost": None}) is None
