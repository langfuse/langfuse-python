import pytest

from langfuse.model import TemplateParser


def test_basic_replacement():
    template = "Hello, {{ name }}!"
    expected = "Hello, John!"

    assert TemplateParser.compile_template(template, {"name": "John"}) == expected


def test_multiple_replacements():
    template = "{{greeting}}, {{name}}! Your balance is {{balance}}."
    expected = "Hello, John! Your balance is $100."

    assert (
        TemplateParser.compile_template(
            template, {"greeting": "Hello", "name": "John", "balance": "$100"}
        )
        == expected
    )


def test_no_replacements():
    template = "This is a test."
    expected = "This is a test."

    assert TemplateParser.compile_template(template) == expected


def test_content_as_variable_name():
    template = "This is a {{content}}."
    expected = "This is a dog."

    assert TemplateParser.compile_template(template, {"content": "dog"}) == expected


def test_unmatched_opening_tag():
    template = "Hello, {{name! Your balance is $100."
    expected = "Hello, {{name! Your balance is $100."

    assert TemplateParser.compile_template(template, {"name": "John"}) == expected


def test_unmatched_closing_tag():
    template = "Hello, {{name}}! Your balance is $100}}"
    expected = "Hello, John! Your balance is $100}}"

    assert TemplateParser.compile_template(template, {"name": "John"}) == expected


def test_missing_variable():
    template = "Hello, {{name}}!"
    expected = "Hello, {{name}}!"

    assert TemplateParser.compile_template(template) == expected


def test_none_variable():
    template = "Hello, {{name}}!"
    expected = "Hello, !"

    assert TemplateParser.compile_template(template, {"name": None}) == expected


def test_strip_whitespace():
    template = "Hello, {{    name }}!"
    expected = "Hello, John!"

    assert TemplateParser.compile_template(template, {"name": "John"}) == expected


def test_special_characters():
    template = "Symbols: {{symbol}}."
    expected = "Symbols: @$%^&*."

    assert TemplateParser.compile_template(template, {"symbol": "@$%^&*"}) == expected


def test_multiple_templates_one_var():
    template = "{{a}} + {{a}} = {{b}}"
    expected = "1 + 1 = 2"

    assert TemplateParser.compile_template(template, {"a": 1, "b": 2}) == expected


def test_unused_variable():
    template = "{{a}} + {{a}}"
    expected = "1 + 1"

    assert TemplateParser.compile_template(template, {"a": 1, "b": 2}) == expected


def test_single_curly_braces():
    template = "{{a}} + {a} = {{b}"
    expected = "1 + {a} = {{b}"

    assert TemplateParser.compile_template(template, {"a": 1, "b": 2}) == expected


def test_complex_json():
    template = """{{a}} + {{
    "key1": "val1",
    "key2": "val2",
    }}"""
    expected = """1 + {{
    "key1": "val1",
    "key2": "val2",
    }}"""

    assert TemplateParser.compile_template(template, {"a": 1, "b": 2}) == expected


def test_replacement_with_empty_string():
    template = "Hello, {{name}}!"
    expected = "Hello, !"

    assert TemplateParser.compile_template(template, {"name": ""}) == expected


def test_variable_case_sensitivity():
    template = "{{Name}} != {{name}}"
    expected = "John != john"

    assert (
        TemplateParser.compile_template(template, {"Name": "John", "name": "john"})
        == expected
    )


def test_start_with_closing_braces():
    template = "}}"
    expected = "}}"

    assert TemplateParser.compile_template(template, {"name": "john"}) == expected


def test_unescaped_JSON_variable_value():
    template = "{{some_json}}"
    some_json = """
{
  "user": {
    "id": 12345,
    "name": "John Doe",
    "email": "john.doe@example.com",
    "isActive": true,
    "accountCreated": "2024-01-15T08:00:00Z",
    "roles": [
      "user",
      "admin"
    ],
    "preferences": {
      "language": "en",
      "notifications": {
        "email": true,
        "sms": false
      }
    },
    "address": {
      "street": "123 Elm Street",
      "city": "Anytown",
      "state": "Anystate",
      "zipCode": "12345",
      "country": "USA"
    }
  }
}"""

    compiled = TemplateParser.compile_template(template, {"some_json": some_json})
    assert compiled == some_json


@pytest.mark.parametrize(
    "template,data,expected",
    [
        ("{{a}} + {{b}} = {{result}}", {"a": 1, "b": 2, "result": 3}, "1 + 2 = 3"),
        ("{{x}}, {{y}}", {"x": "X", "y": "Y"}, "X, Y"),
        ("No variables", {}, "No variables"),
    ],
)
def test_various_templates(template, data, expected):
    assert TemplateParser.compile_template(template, data) == expected
