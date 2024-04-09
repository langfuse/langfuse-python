import pytest

from langfuse.utils import compile_template_string


def test_basic_replacement():
    template = "Hello, {{ name }}!"
    expected = "Hello, John!"

    assert compile_template_string(template, name="John") == expected


def test_multiple_replacements():
    template = "{{greeting}}, {{name}}! Your balance is {{balance}}."
    expected = "Hello, John! Your balance is $100."

    assert (
        compile_template_string(template, greeting="Hello", name="John", balance="$100")
        == expected
    )


def test_no_replacements():
    template = "This is a test."
    expected = "This is a test."

    assert compile_template_string(template) == expected


def test_unmatched_opening_tag():
    template = "Hello, {{name! Your balance is $100."
    expected = "Hello, {{name! Your balance is $100."

    assert compile_template_string(template, name="John") == expected


def test_unmatched_closing_tag():
    template = "Hello, {{name}}! Your balance is $100}}"
    expected = "Hello, John! Your balance is $100}}"

    assert compile_template_string(template, name="John") == expected


def test_missing_variable():
    template = "Hello, {{name}}!"
    expected = "Hello, {{name}}!"

    assert compile_template_string(template) == expected


def test_strip_whitespace():
    template = "Hello, {{    name }}!"
    expected = "Hello, John!"

    assert compile_template_string(template, name="John") == expected


def test_special_characters():
    template = "Symbols: {{symbol}}."
    expected = "Symbols: @$%^&*."

    assert compile_template_string(template, symbol="@$%^&*") == expected


def test_multiple_templates_one_var():
    template = "{{a}} + {{a}} = {{b}}"
    expected = "1 + 1 = 2"

    assert compile_template_string(template, a=1, b=2) == expected


def test_unused_variable():
    template = "{{a}} + {{a}}"
    expected = "1 + 1"

    assert compile_template_string(template, a=1, b=2) == expected


def test_single_curly_braces():
    template = "{{a}} + {a} = {{b}"
    expected = "1 + {a} = {{b}"

    assert compile_template_string(template, a=1, b=2) == expected


def test_complex_json():
    template = """{{a}} + {{
    "key1": "val1",
    "key2": "val2",
    }}"""
    expected = """1 + {{
    "key1": "val1",
    "key2": "val2",
    }}"""

    assert compile_template_string(template, a=1, b=2) == expected


def test_replacement_with_empty_string():
    template = "Hello, {{name}}!"
    expected = "Hello, !"

    assert compile_template_string(template, name="") == expected


def test_variable_case_sensitivity():
    template = "{{Name}} != {{name}}"
    expected = "John != john"

    assert compile_template_string(template, Name="John", name="john") == expected


@pytest.mark.parametrize(
    "template,kwargs,expected",
    [
        ("{{a}} + {{b}} = {{result}}", {"a": 1, "b": 2, "result": 3}, "1 + 2 = 3"),
        ("{{x}}, {{y}}", {"x": "X", "y": "Y"}, "X, Y"),
        ("No variables", {}, "No variables"),
    ],
)
def test_various_templates(template, kwargs, expected):
    assert compile_template_string(template, **kwargs) == expected
