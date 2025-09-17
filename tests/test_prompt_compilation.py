import pytest
from langchain.prompts import ChatPromptTemplate, PromptTemplate

from langfuse.api.resources.prompts import ChatMessage, Prompt_Chat
from langfuse.model import (
    ChatPromptClient,
    Prompt_Text,
    TemplateParser,
    TextPromptClient,
)


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


class TestLangchainPromptCompilation:
    """Test cases for Langchain prompt compilation with JSON handling."""

    def test_normal_variables_with_nested_json(self):
        """Test normal variables (double braces) alongside complex, nested JSON."""
        prompt_string = """This is a prompt with {{animal}} and {{location}}.

{{
    "metadata": {{
        "context": "test",
        "nested": {{
            "animal": {{animal}},
            "properties": {{
                "location": "{{location}}",
                "count": 42
            }}
        }}
    }},
    "data": [
        {{
            "type": "primary",
            "value": {{animal}}
        }}
    ]
}}"""

        prompt = TextPromptClient(
            Prompt_Text(
                type="text",
                name="nested_json_test",
                version=1,
                config={},
                tags=[],
                labels=[],
                prompt=prompt_string,
            )
        )

        langchain_prompt_string = prompt.get_langchain_prompt()
        langchain_prompt = PromptTemplate.from_template(langchain_prompt_string)
        formatted_prompt = langchain_prompt.format(animal="cat", location="Paris")

        expected = """This is a prompt with cat and Paris.

{
    "metadata": {
        "context": "test",
        "nested": {
            "animal": cat,
            "properties": {
                "location": "Paris",
                "count": 42
            }
        }
    },
    "data": [
        {
            "type": "primary",
            "value": cat
        }
    ]
}"""

        assert formatted_prompt == expected

    def test_mixed_variables_with_nested_json(self):
        """Test normal variables (double braces) and Langchain variables (single braces) with nested JSON."""
        prompt_string = """Normal variable: {{user_name}}
Langchain variable: {user_age}

{{
    "user": {{
        "name": {{user_name}},
        "age": {user_age},
        "profile": {{
            "settings": {{
                "theme": "dark",
                "notifications": true
            }}
        }}
    }},
    "system": {{
        "version": "1.0",
        "active": true
    }}
}}"""

        prompt = TextPromptClient(
            Prompt_Text(
                type="text",
                name="mixed_variables_test",
                version=1,
                config={},
                tags=[],
                labels=[],
                prompt=prompt_string,
            )
        )

        langchain_prompt_string = prompt.get_langchain_prompt()
        langchain_prompt = PromptTemplate.from_template(langchain_prompt_string)
        formatted_prompt = langchain_prompt.format(user_name="Alice", user_age=25)

        expected = """Normal variable: Alice
Langchain variable: 25

{
    "user": {
        "name": Alice,
        "age": 25,
        "profile": {
            "settings": {
                "theme": "dark",
                "notifications": true
            }
        }
    },
    "system": {
        "version": "1.0",
        "active": true
    }
}"""

        assert formatted_prompt == expected

    def test_variables_inside_and_alongside_json(self):
        """Test variables both alongside AND INSIDE complex nested JSON."""
        prompt_string = """System message: {{system_msg}}
User input: {user_input}

{{
    "request": {{
        "system": {{system_msg}},
        "user": {user_input},
        "config": {{
            "model": "gpt-4",
            "temperature": 0.7,
            "metadata": {{
                "session": {{session_id}},
                "timestamp": {timestamp},
                "nested_data": {{
                    "level1": {{
                        "level2": {{
                            "user_var": {{user_name}},
                            "system_var": {system_status}
                        }}
                    }}
                }}
            }}
        }}
    }},
    "context": {{context_data}}
}}

Final note: {{system_msg}} and {user_input}"""

        prompt = TextPromptClient(
            Prompt_Text(
                type="text",
                name="variables_inside_json_test",
                version=1,
                config={},
                tags=[],
                labels=[],
                prompt=prompt_string,
            )
        )

        langchain_prompt_string = prompt.get_langchain_prompt()
        langchain_prompt = PromptTemplate.from_template(langchain_prompt_string)
        formatted_prompt = langchain_prompt.format(
            system_msg="Hello",
            user_input="Test input",
            session_id="sess123",
            timestamp=1234567890,
            user_name="Bob",
            system_status="active",
            context_data="context_info",
        )

        expected = """System message: Hello
User input: Test input

{
    "request": {
        "system": Hello,
        "user": Test input,
        "config": {
            "model": "gpt-4",
            "temperature": 0.7,
            "metadata": {
                "session": sess123,
                "timestamp": 1234567890,
                "nested_data": {
                    "level1": {
                        "level2": {
                            "user_var": Bob,
                            "system_var": active
                        }
                    }
                }
            }
        }
    },
    "context": context_info
}

Final note: Hello and Test input"""

        assert formatted_prompt == expected

    def test_edge_case_empty_json_objects(self):
        """Test edge case with empty JSON objects and arrays."""
        prompt_string = """Variable: {{test_var}}

{{
    "empty_object": {{}},
    "empty_array": [],
    "mixed": {{
        "data": {{test_var}},
        "empty": {{}},
        "nested_empty": {{
            "inner": {{}}
        }}
    }}
}}"""

        prompt = TextPromptClient(
            Prompt_Text(
                type="text",
                name="empty_json_test",
                version=1,
                config={},
                tags=[],
                labels=[],
                prompt=prompt_string,
            )
        )

        langchain_prompt_string = prompt.get_langchain_prompt()
        langchain_prompt = PromptTemplate.from_template(langchain_prompt_string)
        formatted_prompt = langchain_prompt.format(test_var="value")

        expected = """Variable: value

{
    "empty_object": {},
    "empty_array": [],
    "mixed": {
        "data": value,
        "empty": {},
        "nested_empty": {
            "inner": {}
        }
    }
}"""

        assert formatted_prompt == expected

    def test_edge_case_nested_quotes_in_json(self):
        """Test edge case with nested quotes and escaped characters in JSON."""
        prompt_string = """Message: {{message}}

{{
    "text": "This is a \\"quoted\\" string",
    "user_message": {{message}},
    "escaped": "Line 1\\\\nLine 2",
    "complex": {{
        "description": "Contains 'single' and \\"double\\" quotes",
        "dynamic": {{message}}
    }}
}}"""

        prompt = TextPromptClient(
            Prompt_Text(
                type="text",
                name="nested_quotes_test",
                version=1,
                config={},
                tags=[],
                labels=[],
                prompt=prompt_string,
            )
        )

        langchain_prompt_string = prompt.get_langchain_prompt()
        langchain_prompt = PromptTemplate.from_template(langchain_prompt_string)
        formatted_prompt = langchain_prompt.format(message="Hello world")

        expected = """Message: Hello world

{
    "text": "This is a \\"quoted\\" string",
    "user_message": Hello world,
    "escaped": "Line 1\\\\nLine 2",
    "complex": {
        "description": "Contains 'single' and \\"double\\" quotes",
        "dynamic": Hello world
    }
}"""

        assert formatted_prompt == expected

    def test_edge_case_json_with_variables_in_strings(self):
        """Test that double braces inside JSON strings are treated as normal variables."""
        prompt_string = """Variable: {{test_var}}

{{
    "text_with_braces": "This has {{connector}} characters",
    "also_braces": "Format: {{key}} = {{value}}",
    "user_data": {{test_var}}
}}"""

        prompt = TextPromptClient(
            Prompt_Text(
                type="text",
                name="variables_in_strings_test",
                version=1,
                config={},
                tags=[],
                labels=[],
                prompt=prompt_string,
            )
        )

        langchain_prompt_string = prompt.get_langchain_prompt()
        langchain_prompt = PromptTemplate.from_template(langchain_prompt_string)
        formatted_prompt = langchain_prompt.format(
            test_var="test_value", key="name", value="John", connector="special"
        )

        expected = """Variable: test_value

{
    "text_with_braces": "This has special characters",
    "also_braces": "Format: name = John",
    "user_data": test_value
}"""

        assert formatted_prompt == expected

    def test_complex_real_world_scenario(self):
        """Test a complex real-world scenario combining all features."""
        prompt_string = """System: {{system_prompt}}
User query: {user_query}
Context: {{context}}

{{
    "request": {{
        "system_instruction": {{system_prompt}},
        "user_input": {user_query},
        "context": {{context}},
        "settings": {{
            "model": "gpt-4",
            "temperature": 0.7,
            "max_tokens": 1000,
            "functions": [
                {{
                    "name": "search",
                    "description": "Search for information",
                    "parameters": {{
                        "query": {user_query},
                        "context": {{context}}
                    }}
                }}
            ]
        }},
        "metadata": {{
            "session_id": {{session_id}},
            "timestamp": {timestamp},
            "user_info": {{
                "id": {user_id},
                "preferences": {{
                    "language": "en",
                    "format": "json"
                }}
            }}
        }}
    }},
    "response_format": {{
        "type": "structured",
        "schema": {{
            "answer": "string",
            "confidence": "number",
            "sources": "array"
        }}
    }}
}}

Instructions: Use {{system_prompt}} to process {user_query} with context {{context}}."""

        prompt = TextPromptClient(
            Prompt_Text(
                type="text",
                name="complex_scenario_test",
                version=1,
                config={},
                tags=[],
                labels=[],
                prompt=prompt_string,
            )
        )

        langchain_prompt_string = prompt.get_langchain_prompt()
        langchain_prompt = PromptTemplate.from_template(langchain_prompt_string)
        formatted_prompt = langchain_prompt.format(
            system_prompt="You are a helpful assistant",
            user_query="What is the weather?",
            context="Weather inquiry",
            session_id="sess_123",
            timestamp=1234567890,
            user_id="user_456",
        )

        expected = """System: You are a helpful assistant
User query: What is the weather?
Context: Weather inquiry

{
    "request": {
        "system_instruction": You are a helpful assistant,
        "user_input": What is the weather?,
        "context": Weather inquiry,
        "settings": {
            "model": "gpt-4",
            "temperature": 0.7,
            "max_tokens": 1000,
            "functions": [
                {
                    "name": "search",
                    "description": "Search for information",
                    "parameters": {
                        "query": What is the weather?,
                        "context": Weather inquiry
                    }
                }
            ]
        },
        "metadata": {
            "session_id": sess_123,
            "timestamp": 1234567890,
            "user_info": {
                "id": user_456,
                "preferences": {
                    "language": "en",
                    "format": "json"
                }
            }
        }
    },
    "response_format": {
        "type": "structured",
        "schema": {
            "answer": "string",
            "confidence": "number",
            "sources": "array"
        }
    }
}

Instructions: Use You are a helpful assistant to process What is the weather? with context Weather inquiry."""

        assert formatted_prompt == expected

    def test_chat_prompt_with_json_variables(self):
        """Test that chat prompts work correctly with JSON handling and variables."""
        chat_messages = [
            ChatMessage(
                role="system",
                content="""You are {{assistant_type}} assistant.

Configuration:
{{
    "settings": {{
        "model": "{{model_name}}",
        "temperature": {temperature},
        "capabilities": [
            {{
                "name": "search",
                "enabled": {{search_enabled}},
                "params": {{
                    "provider": "{{search_provider}}"
                }}
            }}
        ]
    }}
}}""",
            ),
            ChatMessage(
                role="user",
                content="Hello {{user_name}}! I need help with: {{user_request}}",
            ),
        ]

        prompt = ChatPromptClient(
            Prompt_Chat(
                type="chat",
                name="chat_json_test",
                version=1,
                config={},
                tags=[],
                labels=[],
                prompt=chat_messages,
            )
        )

        langchain_messages = prompt.get_langchain_prompt()
        langchain_prompt = ChatPromptTemplate.from_messages(langchain_messages)
        formatted_messages = langchain_prompt.format_messages(
            assistant_type="helpful",
            model_name="gpt-4",
            temperature=0.7,
            search_enabled="true",
            search_provider="google",
            user_name="Alice",
            user_request="data analysis",
        )

        expected_system = """You are helpful assistant.

Configuration:
{
    "settings": {
        "model": "gpt-4",
        "temperature": 0.7,
        "capabilities": [
            {
                "name": "search",
                "enabled": true,
                "params": {
                    "provider": "google"
                }
            }
        ]
    }
}"""

        expected_user = "Hello Alice! I need help with: data analysis"

        assert len(formatted_messages) == 2
        assert formatted_messages[0].content == expected_system
        assert formatted_messages[1].content == expected_user

    def test_chat_prompt_with_placeholders_langchain(self):
        """Test that chat prompts with placeholders work correctly with Langchain."""
        from langfuse.api.resources.prompts import Prompt_Chat

        chat_messages = [
            ChatMessage(
                role="system",
                content="You are a {{role}} assistant with {{capability}} capabilities.",
            ),
            {"type": "placeholder", "name": "examples"},
            ChatMessage(
                role="user",
                content="Help me with {{task}}.",
            ),
        ]

        prompt_client = ChatPromptClient(
            Prompt_Chat(
                type="chat",
                name="chat_placeholder_langchain_test",
                version=1,
                config={},
                tags=[],
                labels=[],
                prompt=chat_messages,
            ),
        )

        # Test compile with placeholders and variables
        compiled_messages = prompt_client.compile(
            role="helpful",
            capability="math",
            task="addition",
            examples=[
                {"role": "user", "content": "Example: What is 2+2?"},
                {"role": "assistant", "content": "2+2 equals 4."},
            ],
        )

        assert len(compiled_messages) == 4
        assert (
            compiled_messages[0]["content"]
            == "You are a helpful assistant with math capabilities."
        )
        assert compiled_messages[1]["content"] == "Example: What is 2+2?"
        assert compiled_messages[2]["content"] == "2+2 equals 4."
        assert compiled_messages[3]["content"] == "Help me with addition."

        langchain_messages = prompt_client.get_langchain_prompt(
            role="helpful",
            capability="math",
            task="addition",
            examples=[
                {"role": "user", "content": "Example: What is 2+2?"},
                {"role": "assistant", "content": "2+2 equals 4."},
            ],
        )
        langchain_prompt = ChatPromptTemplate.from_messages(langchain_messages)
        formatted_messages = langchain_prompt.format_messages()

        assert len(formatted_messages) == 4
        assert (
            formatted_messages[0].content
            == "You are a helpful assistant with math capabilities."
        )
        assert formatted_messages[1].content == "Example: What is 2+2?"
        assert formatted_messages[2].content == "2+2 equals 4."
        assert formatted_messages[3].content == "Help me with addition."

    def test_get_langchain_prompt_with_unresolved_placeholders(self):
        """Test that unresolved placeholders become MessagesPlaceholder objects."""
        from langfuse.api.resources.prompts import Prompt_Chat
        from langfuse.model import ChatPromptClient

        chat_messages = [
            {"role": "system", "content": "You are a {{role}} assistant"},
            {"type": "placeholder", "name": "examples"},
            {"role": "user", "content": "Help me with {{task}}"},
        ]

        prompt_client = ChatPromptClient(
            Prompt_Chat(
                type="chat",
                name="test_unresolved_placeholder",
                version=1,
                config={},
                tags=[],
                labels=[],
                prompt=chat_messages,
            ),
        )

        # Call get_langchain_prompt without resolving placeholder
        langchain_messages = prompt_client.get_langchain_prompt(
            role="helpful",
            task="coding",
        )

        # Should have 3 items: system message, MessagesPlaceholder, user message
        assert len(langchain_messages) == 3

        # First message should be the system message
        assert langchain_messages[0] == ("system", "You are a helpful assistant")

        # Second should be a MessagesPlaceholder for the unresolved placeholder
        placeholder_msg = langchain_messages[1]
        try:
            from langchain_core.prompts.chat import MessagesPlaceholder

            assert isinstance(placeholder_msg, MessagesPlaceholder)
            assert placeholder_msg.variable_name == "examples"
        except ImportError:
            # Fallback case when langchain_core is not available
            assert placeholder_msg == ("system", "{examples}")

        # Third message should be the user message
        assert langchain_messages[2] == ("user", "Help me with coding")


def test_tool_calls_preservation_in_message_placeholder():
    """Test that tool calls are preserved when compiling message placeholders."""
    from langfuse.api.resources.prompts import Prompt_Chat

    chat_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"type": "placeholder", "name": "message_history"},
        {"role": "user", "content": "Help me with {{task}}"},
    ]

    prompt_client = ChatPromptClient(
        Prompt_Chat(
            type="chat",
            name="tool_calls_test",
            version=1,
            config={},
            tags=[],
            labels=[],
            prompt=chat_messages,
        )
    )

    # Message history with tool calls - exactly like the bug report describes
    message_history_with_tool_calls = [
        {"role": "user", "content": "What's the weather like?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_123",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"location": "San Francisco"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "content": "It's sunny, 72°F",
            "tool_call_id": "call_123",
            "name": "get_weather",
        },
    ]

    # Compile with message history and variables
    compiled_messages = prompt_client.compile(
        task="weather inquiry", message_history=message_history_with_tool_calls
    )

    # Should have 5 messages: system + 3 from history + user
    assert len(compiled_messages) == 5

    # System message
    assert compiled_messages[0]["role"] == "system"
    assert compiled_messages[0]["content"] == "You are a helpful assistant."

    # User message from history
    assert compiled_messages[1]["role"] == "user"
    assert compiled_messages[1]["content"] == "What's the weather like?"

    # Assistant message with TOOL CALLS
    assert compiled_messages[2]["role"] == "assistant"
    assert compiled_messages[2]["content"] == ""
    assert "tool_calls" in compiled_messages[2]
    assert len(compiled_messages[2]["tool_calls"]) == 1
    assert compiled_messages[2]["tool_calls"][0]["id"] == "call_123"
    assert compiled_messages[2]["tool_calls"][0]["function"]["name"] == "get_weather"

    # TOOL CALL results message
    assert compiled_messages[3]["role"] == "tool"
    assert compiled_messages[3]["content"] == "It's sunny, 72°F"
    assert compiled_messages[3]["tool_call_id"] == "call_123"
    assert compiled_messages[3]["name"] == "get_weather"

    # Final user message with compiled variable
    assert compiled_messages[4]["role"] == "user"
    assert compiled_messages[4]["content"] == "Help me with weather inquiry"
