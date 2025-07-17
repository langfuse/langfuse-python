from time import sleep
from unittest.mock import Mock, patch

import openai
import pytest

from langfuse._client.client import Langfuse
from langfuse._utils.prompt_cache import (
    DEFAULT_PROMPT_CACHE_TTL_SECONDS,
    PromptCacheItem,
)
from langfuse.api.resources.prompts import Prompt_Chat, Prompt_Text
from langfuse.model import ChatPromptClient, TextPromptClient
from tests.utils import create_uuid, get_api

OVERRIDE_DEFAULT_PROMPT_CACHE_TTL_SECONDS = 120

def test_create_prompt():
    langfuse = Langfuse()
    prompt_name = create_uuid()
    prompt_client = langfuse.create_prompt(
        name=prompt_name,
        prompt="test prompt",
        labels=["production"],
        commit_message="initial commit",
    )

    second_prompt_client = langfuse.get_prompt(prompt_name)

    assert prompt_client.name == second_prompt_client.name
    assert prompt_client.version == second_prompt_client.version
    assert prompt_client.prompt == second_prompt_client.prompt
    assert prompt_client.config == second_prompt_client.config
    assert prompt_client.commit_message == second_prompt_client.commit_message
    assert prompt_client.config == {}


def test_create_prompt_with_special_chars_in_name():
    langfuse = Langfuse()
    prompt_name = create_uuid() + "special chars !@#$%^&*() +"
    prompt_client = langfuse.create_prompt(
        name=prompt_name,
        prompt="test prompt",
        labels=["production"],
        tags=["test"],
    )

    second_prompt_client = langfuse.get_prompt(prompt_name)

    assert prompt_client.name == second_prompt_client.name
    assert prompt_client.version == second_prompt_client.version
    assert prompt_client.prompt == second_prompt_client.prompt
    assert prompt_client.tags == second_prompt_client.tags
    assert prompt_client.config == second_prompt_client.config
    assert prompt_client.config == {}


def test_create_chat_prompt():
    langfuse = Langfuse()
    prompt_name = create_uuid()

    prompt_client = langfuse.create_prompt(
        name=prompt_name,
        prompt=[
            {"role": "system", "content": "test prompt 1 with {{animal}}"},
            {"role": "user", "content": "test prompt 2 with {{occupation}}"},
        ],
        labels=["production"],
        tags=["test"],
        type="chat",
        commit_message="initial commit",
    )

    second_prompt_client = langfuse.get_prompt(prompt_name, type="chat")

    # Create a test generation
    completion = openai.OpenAI().chat.completions.create(
        model="gpt-4",
        messages=prompt_client.compile(animal="dog", occupation="doctor"),
    )

    assert len(completion.choices) > 0

    assert prompt_client.name == second_prompt_client.name
    assert prompt_client.version == second_prompt_client.version
    assert prompt_client.prompt == second_prompt_client.prompt
    assert prompt_client.config == second_prompt_client.config
    assert prompt_client.labels == ["production", "latest"]
    assert prompt_client.tags == second_prompt_client.tags
    assert prompt_client.commit_message == second_prompt_client.commit_message
    assert prompt_client.config == {}


def test_create_chat_prompt_with_placeholders():
    langfuse = Langfuse()
    prompt_name = create_uuid()

    prompt_client = langfuse.create_prompt(
        name=prompt_name,
        prompt=[
            {"role": "system", "content": "You are a {{role}} assistant"},
            {"type": "placeholder", "name": "history"},
            {"role": "user", "content": "Help me with {{task}}"},
        ],
        labels=["production"],
        tags=["test"],
        type="chat",
        commit_message="initial commit",
    )

    second_prompt_client = langfuse.get_prompt(prompt_name, type="chat")
    messages = second_prompt_client.compile(
        role="helpful",
        task="coding",
        history=[
            {"role": "user", "content": "Example: {{task}}"},
            {"role": "assistant", "content": "Example response"},
        ],
    )

    # Create a test generation using compiled messages
    completion = openai.OpenAI().chat.completions.create(
        model="gpt-4",
        messages=messages,
    )

    assert len(completion.choices) > 0
    assert len(messages) == 4
    assert messages[0]["content"] == "You are a helpful assistant"
    assert messages[1]["content"] == "Example: coding"
    assert messages[2]["content"] == "Example response"
    assert messages[3]["content"] == "Help me with coding"

    assert prompt_client.name == second_prompt_client.name
    assert prompt_client.version == second_prompt_client.version
    assert prompt_client.config == second_prompt_client.config
    assert prompt_client.labels == ["production", "latest"]
    assert prompt_client.tags == second_prompt_client.tags
    assert prompt_client.commit_message == second_prompt_client.commit_message
    assert prompt_client.config == {}


def test_create_prompt_with_placeholders():
    """Test creating a prompt with placeholder messages."""
    langfuse = Langfuse()
    prompt_name = create_uuid()
    prompt_client = langfuse.create_prompt(
        name=prompt_name,
        prompt=[
            {"role": "system", "content": "System message"},
            {"type": "placeholder", "name": "context"},
            {"role": "user", "content": "User message"},
        ],
        type="chat",
    )

    # Verify the full prompt structure with placeholders
    assert len(prompt_client.prompt) == 3

    # First message - system
    assert prompt_client.prompt[0]["type"] == "message"
    assert prompt_client.prompt[0]["role"] == "system"
    assert prompt_client.prompt[0]["content"] == "System message"
    # Placeholder
    assert prompt_client.prompt[1]["type"] == "placeholder"
    assert prompt_client.prompt[1]["name"] == "context"
    # Third message - user
    assert prompt_client.prompt[2]["type"] == "message"
    assert prompt_client.prompt[2]["role"] == "user"
    assert prompt_client.prompt[2]["content"] == "User message"


def test_get_prompt_with_placeholders():
    """Test retrieving a prompt with placeholders."""
    langfuse = Langfuse()
    prompt_name = create_uuid()

    langfuse.create_prompt(
        name=prompt_name,
        prompt=[
            {"role": "system", "content": "You are {{name}}"},
            {"type": "placeholder", "name": "history"},
            {"role": "user", "content": "{{question}}"},
        ],
        type="chat",
    )

    prompt_client = langfuse.get_prompt(prompt_name, type="chat", version=1)

    # Verify placeholder structure is preserved
    assert len(prompt_client.prompt) == 3

    # First message - system with variable
    assert prompt_client.prompt[0]["type"] == "message"
    assert prompt_client.prompt[0]["role"] == "system"
    assert prompt_client.prompt[0]["content"] == "You are {{name}}"
    # Placeholder
    assert prompt_client.prompt[1]["type"] == "placeholder"
    assert prompt_client.prompt[1]["name"] == "history"
    # Third message - user with variable
    assert prompt_client.prompt[2]["type"] == "message"
    assert prompt_client.prompt[2]["role"] == "user"
    assert prompt_client.prompt[2]["content"] == "{{question}}"


@pytest.mark.parametrize(
    ("variables", "placeholders", "expected_len", "expected_contents"),
    [
        # 0. Variables only, no placeholders. Unresolved placeholders kept in output
        (
            {"role": "helpful", "task": "coding"},
            {},
            3,
            [
                "You are a helpful assistant",
                None,
                "Help me with coding",
            ],  # None = placeholder
        ),
        # 1. No variables, no placeholders. Expect verbatim message+placeholder output
        (
            {},
            {},
            3,
            ["You are a {{role}} assistant", None, "Help me with {{task}}"],
        ),  # None = placeholder
        # 2. Placeholders only, empty variables. Expect output with placeholders filled in
        (
            {},
            {
                "examples": [
                    {"role": "user", "content": "Example question"},
                    {"role": "assistant", "content": "Example answer"},
                ],
            },
            4,
            [
                "You are a {{role}} assistant",
                "Example question",
                "Example answer",
                "Help me with {{task}}",
            ],
        ),
        # 3. Both variables and placeholders. Expect fully compiled output
        (
            {"role": "helpful", "task": "coding"},
            {
                "examples": [
                    {"role": "user", "content": "Show me {{task}}"},
                    {"role": "assistant", "content": "Here's {{task}}"},
                ],
            },
            4,
            [
                "You are a helpful assistant",
                "Show me coding",
                "Here's coding",
                "Help me with coding",
            ],
        ),
        # # Empty placeholder array
        # This is expected to fail! If the user provides a placeholder, it should contain an array
        # (
        #     {"role": "helpful", "task": "coding"},
        #     {"examples": []},
        #     2,
        #     ["You are a helpful assistant", "Help me with coding"],
        # ),
        # 4. Unused placeholder fill ins. Unresolved placeholders kept in output
        (
            {"role": "helpful", "task": "coding"},
            {"unused": [{"role": "user", "content": "Won't appear"}]},
            3,
            [
                "You are a helpful assistant",
                None,
                "Help me with coding",
            ],  # None = placeholder
        ),
        # 5. Placeholder with non-list value (should log warning and append as string)
        (
            {"role": "helpful", "task": "coding"},
            {"examples": "not a list"},
            3,
            [
                "You are a helpful assistant",
                "not a list",  # String value appended directly
                "Help me with coding",
            ],
        ),
        # 6. Placeholder with invalid message structure (should log warning and include both)
        (
            {"role": "helpful", "task": "coding"},
            {
                "examples": [
                    "invalid message",
                    {"role": "user", "content": "valid message"},
                ]
            },
            4,
            [
                "You are a helpful assistant",
                "['invalid message', {'role': 'user', 'content': 'valid message'}]",  # Invalid structure becomes string
                "valid message",  # Valid message processed normally
                "Help me with coding",
            ],
        ),
    ],
)
def test_compile_with_placeholders(
    variables, placeholders, expected_len, expected_contents
) -> None:
    """Test compile_with_placeholders with different variable/placeholder combinations."""
    from langfuse.api.resources.prompts import Prompt_Chat
    from langfuse.model import ChatPromptClient

    mock_prompt = Prompt_Chat(
        name="test_prompt",
        version=1,
        type="chat",
        config={},
        tags=[],
        labels=[],
        prompt=[
            {"role": "system", "content": "You are a {{role}} assistant"},
            {"type": "placeholder", "name": "examples"},
            {"role": "user", "content": "Help me with {{task}}"},
        ],
    )

    compile_kwargs = {**placeholders, **variables}
    result = ChatPromptClient(mock_prompt).compile(**compile_kwargs)

    assert len(result) == expected_len
    for i, expected_content in enumerate(expected_contents):
        if expected_content is None:
            # This should be an unresolved placeholder
            assert "type" in result[i] and result[i]["type"] == "placeholder"
        elif isinstance(result[i], str):
            # This is a string value from invalid placeholder
            assert result[i] == expected_content
        else:
            # This should be a regular message
            assert "content" in result[i]
            assert result[i]["content"] == expected_content


def test_warning_on_unresolved_placeholders():
    """Test that a warning is emitted when compiling with unresolved placeholders."""
    from unittest.mock import patch

    langfuse = Langfuse()
    prompt_name = create_uuid()

    langfuse.create_prompt(
        name=prompt_name,
        prompt=[
            {"role": "system", "content": "You are {{name}}"},
            {"type": "placeholder", "name": "history"},
            {"role": "user", "content": "{{question}}"},
        ],
        type="chat",
    )

    prompt_client = langfuse.get_prompt(prompt_name, type="chat", version=1)

    # Test that warning is emitted when compiling with unresolved placeholders
    with patch("langfuse.logger.langfuse_logger.warning") as mock_warning:
        # Compile without providing the 'history' placeholder
        result = prompt_client.compile(name="Assistant", question="What is 2+2?")

        # Verify the warning was called with the expected message
        mock_warning.assert_called_once()
        warning_message = mock_warning.call_args[0][0]
        assert "Placeholders ['history'] have not been resolved" in warning_message

        # Verify the result only contains the resolved messages
        assert len(result) == 3
        assert result[0]["content"] == "You are Assistant"
        assert result[1]["name"] == "history"
        assert result[2]["content"] == "What is 2+2?"


def test_compiling_chat_prompt():
    langfuse = Langfuse()
    prompt_name = create_uuid()

    prompt_client = langfuse.create_prompt(
        name=prompt_name,
        prompt=[
            {
                "role": "system",
                "content": "test prompt 1 with {{state}} {{target}} {{state}}",
            },
            {"role": "user", "content": "test prompt 2 with {{state}}"},
        ],
        labels=["production"],
        type="chat",
    )

    second_prompt_client = langfuse.get_prompt(prompt_name, type="chat")

    assert prompt_client.name == second_prompt_client.name
    assert prompt_client.version == second_prompt_client.version
    assert prompt_client.prompt == second_prompt_client.prompt
    assert prompt_client.labels == ["production", "latest"]

    assert second_prompt_client.compile(target="world", state="great") == [
        {"role": "system", "content": "test prompt 1 with great world great"},
        {"role": "user", "content": "test prompt 2 with great"},
    ]


def test_compiling_prompt():
    langfuse = Langfuse()
    prompt_name = "test_compiling_prompt"

    prompt_client = langfuse.create_prompt(
        name=prompt_name,
        prompt='Hello, {{target}}! I hope you are {{state}}. {{undefined_variable}}. And here is some JSON that should not be compiled: {{ "key": "value" }} \
            Here is a custom var for users using str.format instead of the mustache-style double curly braces: {custom_var}',
        labels=["production"],
    )

    second_prompt_client = langfuse.get_prompt(prompt_name)

    assert prompt_client.name == second_prompt_client.name
    assert prompt_client.version == second_prompt_client.version
    assert prompt_client.prompt == second_prompt_client.prompt
    assert prompt_client.labels == ["production", "latest"]

    compiled = second_prompt_client.compile(target="world", state="great")

    assert (
        compiled
        == 'Hello, world! I hope you are great. {{undefined_variable}}. And here is some JSON that should not be compiled: {{ "key": "value" }} \
            Here is a custom var for users using str.format instead of the mustache-style double curly braces: {custom_var}'
    )


def test_compiling_prompt_without_character_escaping():
    langfuse = Langfuse()
    prompt_name = "test_compiling_prompt_without_character_escaping"

    prompt_client = langfuse.create_prompt(
        name=prompt_name, prompt="Hello, {{ some_json }}", labels=["production"]
    )

    second_prompt_client = langfuse.get_prompt(prompt_name)

    assert prompt_client.name == second_prompt_client.name
    assert prompt_client.version == second_prompt_client.version
    assert prompt_client.prompt == second_prompt_client.prompt
    assert prompt_client.labels == ["production", "latest"]

    some_json = '{"key": "value"}'
    compiled = second_prompt_client.compile(some_json=some_json)

    assert compiled == 'Hello, {"key": "value"}'


def test_compiling_prompt_with_content_as_variable_name():
    langfuse = Langfuse()
    prompt_name = "test_compiling_prompt_with_content_as_variable_name"

    prompt_client = langfuse.create_prompt(
        name=prompt_name,
        prompt="Hello, {{ content }}!",
        labels=["production"],
    )

    second_prompt_client = langfuse.get_prompt(prompt_name)

    assert prompt_client.name == second_prompt_client.name
    assert prompt_client.version == second_prompt_client.version
    assert prompt_client.prompt == second_prompt_client.prompt
    assert prompt_client.labels == ["production", "latest"]

    compiled = second_prompt_client.compile(content="Jane")

    assert compiled == "Hello, Jane!"


def test_create_prompt_with_null_config():
    langfuse = Langfuse(debug=False)

    langfuse.create_prompt(
        name="test_null_config",
        prompt="Hello, world! I hope you are great",
        labels=["production"],
        config=None,
    )

    prompt = langfuse.get_prompt("test_null_config")

    assert prompt.config == {}


def test_create_prompt_with_tags():
    langfuse = Langfuse(debug=False)
    prompt_name = create_uuid()

    langfuse.create_prompt(
        name=prompt_name,
        prompt="Hello, world! I hope you are great",
        tags=["tag1", "tag2"],
    )

    prompt = langfuse.get_prompt(prompt_name, version=1)

    assert prompt.tags == ["tag1", "tag2"]


def test_create_prompt_with_empty_tags():
    langfuse = Langfuse(debug=False)
    prompt_name = create_uuid()

    langfuse.create_prompt(
        name=prompt_name,
        prompt="Hello, world! I hope you are great",
        tags=[],
    )

    prompt = langfuse.get_prompt(prompt_name, version=1)

    assert prompt.tags == []


def test_create_prompt_with_previous_tags():
    langfuse = Langfuse(debug=False)
    prompt_name = create_uuid()

    langfuse.create_prompt(
        name=prompt_name,
        prompt="Hello, world! I hope you are great",
    )

    prompt = langfuse.get_prompt(prompt_name, version=1)

    assert prompt.tags == []

    langfuse.create_prompt(
        name=prompt_name,
        prompt="Hello, world! I hope you are great",
        tags=["tag1", "tag2"],
    )

    prompt_v2 = langfuse.get_prompt(prompt_name, version=2)

    assert prompt_v2.tags == ["tag1", "tag2"]

    langfuse.create_prompt(
        name=prompt_name,
        prompt="Hello, world! I hope you are great",
    )

    prompt_v3 = langfuse.get_prompt(prompt_name, version=3)

    assert prompt_v3.tags == ["tag1", "tag2"]


def test_remove_prompt_tags():
    langfuse = Langfuse(debug=False)
    prompt_name = create_uuid()

    langfuse.create_prompt(
        name=prompt_name,
        prompt="Hello, world! I hope you are great",
        tags=["tag1", "tag2"],
    )

    langfuse.create_prompt(
        name=prompt_name,
        prompt="Hello, world! I hope you are great",
        tags=[],
    )

    prompt_v1 = langfuse.get_prompt(prompt_name, version=1)
    prompt_v2 = langfuse.get_prompt(prompt_name, version=2)

    assert prompt_v1.tags == []
    assert prompt_v2.tags == []


def test_update_prompt_tags():
    langfuse = Langfuse(debug=False)
    prompt_name = create_uuid()

    langfuse.create_prompt(
        name=prompt_name,
        prompt="Hello, world! I hope you are great",
        tags=["tag1", "tag2"],
    )

    prompt_v1 = langfuse.get_prompt(prompt_name, version=1)

    assert prompt_v1.tags == ["tag1", "tag2"]

    langfuse.create_prompt(
        name=prompt_name,
        prompt="Hello, world! I hope you are great",
        tags=["tag3", "tag4"],
    )

    prompt_v2 = langfuse.get_prompt(prompt_name, version=2)

    assert prompt_v2.tags == ["tag3", "tag4"]


def test_get_prompt_by_version_or_label():
    langfuse = Langfuse()
    prompt_name = create_uuid()

    for i in range(3):
        langfuse.create_prompt(
            name=prompt_name,
            prompt="test prompt " + str(i + 1),
            labels=["production"] if i == 1 else [],
        )

    default_prompt_client = langfuse.get_prompt(prompt_name)
    assert default_prompt_client.version == 2
    assert default_prompt_client.prompt == "test prompt 2"
    assert default_prompt_client.labels == ["production"]

    first_prompt_client = langfuse.get_prompt(prompt_name, version=1)
    assert first_prompt_client.version == 1
    assert first_prompt_client.prompt == "test prompt 1"
    assert first_prompt_client.labels == []

    second_prompt_client = langfuse.get_prompt(prompt_name, version=2)
    assert second_prompt_client.version == 2
    assert second_prompt_client.prompt == "test prompt 2"
    assert second_prompt_client.labels == ["production"]

    third_prompt_client = langfuse.get_prompt(prompt_name, label="latest")
    assert third_prompt_client.version == 3
    assert third_prompt_client.prompt == "test prompt 3"
    assert third_prompt_client.labels == ["latest"]


def test_prompt_end_to_end():
    langfuse = Langfuse(debug=False)

    langfuse.create_prompt(
        name="test",
        prompt="Hello, {{target}}! I hope you are {{state}}.",
        labels=["production"],
        config={"temperature": 0.5},
    )

    prompt = langfuse.get_prompt("test")

    prompt_str = prompt.compile(target="world", state="great")
    assert prompt_str == "Hello, world! I hope you are great."
    assert prompt.config == {"temperature": 0.5}

    generation = langfuse.start_generation(
        name="mygen", input=prompt_str, prompt=prompt
    ).end()

    # to check that these do not error
    generation.update(prompt=prompt)

    langfuse.flush()

    api = get_api()

    trace = api.trace.get(generation.trace_id)

    assert len(trace.observations) == 1

    generation = trace.observations[0]
    assert generation.prompt_id is not None

    observation = api.observations.get(generation.id)

    assert observation.prompt_id is not None


@pytest.fixture
def langfuse():
    langfuse_instance = Langfuse(
        public_key="test-public-key",
        secret_key="test-secret-key",
        host="https://mock-host.com",
    )
    langfuse_instance.api = Mock()

    return langfuse_instance

@pytest.fixture
def langfuse_with_override_default_cache():
    langfuse_instance = Langfuse(
        public_key="test-public-key",
        secret_key="test-secret-key",
        host="https://mock-host.com",
        default_cache_ttl_seconds=OVERRIDE_DEFAULT_PROMPT_CACHE_TTL_SECONDS,
    )
    langfuse_instance.api = Mock()
    return langfuse_instance

# Fetching a new prompt when nothing in cache
def test_get_fresh_prompt(langfuse):
    prompt_name = "test_get_fresh_prompt"
    prompt = Prompt_Text(
        name=prompt_name,
        version=1,
        prompt="Make me laugh",
        type="text",
        labels=[],
        config={},
        tags=[],
    )

    mock_server_call = langfuse.api.prompts.get
    mock_server_call.return_value = prompt

    result = langfuse.get_prompt(prompt_name, fallback="fallback")
    mock_server_call.assert_called_once_with(
        prompt_name,
        version=None,
        label=None,
        request_options=None,
    )

    assert result == TextPromptClient(prompt)


# Should throw an error if prompt name is unspecified
def test_throw_if_name_unspecified(langfuse):
    prompt_name = ""

    with pytest.raises(ValueError) as exc_info:
        langfuse.get_prompt(prompt_name)

    assert "Prompt name cannot be empty" in str(exc_info.value)


# Should throw an error if nothing in cache and fetch fails
def test_throw_when_failing_fetch_and_no_cache(langfuse):
    prompt_name = "failing_fetch_and_no_cache"

    mock_server_call = langfuse.api.prompts.get
    mock_server_call.side_effect = Exception("Prompt not found")

    with pytest.raises(Exception) as exc_info:
        langfuse.get_prompt(prompt_name)

    assert "Prompt not found" in str(exc_info.value)


def test_using_custom_prompt_timeouts(langfuse):
    prompt_name = "test_using_custom_prompt_timeouts"
    prompt = Prompt_Text(
        name=prompt_name,
        version=1,
        prompt="Make me laugh",
        type="text",
        labels=[],
        config={},
        tags=[],
    )

    mock_server_call = langfuse.api.prompts.get
    mock_server_call.return_value = prompt

    result = langfuse.get_prompt(
        prompt_name, fallback="fallback", fetch_timeout_seconds=1000
    )
    mock_server_call.assert_called_once_with(
        prompt_name,
        version=None,
        label=None,
        request_options={"timeout_in_seconds": 1000},
    )

    assert result == TextPromptClient(prompt)


# Should throw an error if cache_ttl_seconds is passed as positional rather than keyword argument
def test_throw_if_cache_ttl_seconds_positional_argument(langfuse):
    prompt_name = "test ttl seconds in positional arg"
    ttl_seconds = 20

    with pytest.raises(TypeError) as exc_info:
        langfuse.get_prompt(prompt_name, ttl_seconds)

    assert "positional arguments" in str(exc_info.value)


# Should return cached prompt if not expired
def test_get_valid_cached_prompt(langfuse):
    prompt_name = "test_get_valid_cached_prompt"
    prompt = Prompt_Text(
        name=prompt_name,
        version=1,
        prompt="Make me laugh",
        type="text",
        labels=[],
        config={},
        tags=[],
    )
    prompt_client = TextPromptClient(prompt)

    mock_server_call = langfuse.api.prompts.get
    mock_server_call.return_value = prompt

    result_call_1 = langfuse.get_prompt(prompt_name, fallback="fallback")
    assert mock_server_call.call_count == 1
    assert result_call_1 == prompt_client

    result_call_2 = langfuse.get_prompt(prompt_name)
    assert mock_server_call.call_count == 1
    assert result_call_2 == prompt_client


# Should return cached chat prompt if not expired when fetching by label
def test_get_valid_cached_chat_prompt_by_label(langfuse):
    prompt_name = "test_get_valid_cached_chat_prompt_by_label"
    prompt = Prompt_Chat(
        name=prompt_name,
        version=1,
        prompt=[{"role": "system", "content": "Make me laugh"}],
        labels=["test"],
        type="chat",
        config={},
        tags=[],
    )
    prompt_client = ChatPromptClient(prompt)

    mock_server_call = langfuse.api.prompts.get
    mock_server_call.return_value = prompt

    result_call_1 = langfuse.get_prompt(prompt_name, label="test")
    assert mock_server_call.call_count == 1
    assert result_call_1 == prompt_client

    result_call_2 = langfuse.get_prompt(prompt_name, label="test")
    assert mock_server_call.call_count == 1
    assert result_call_2 == prompt_client


# Should return cached chat prompt if not expired when fetching by version
def test_get_valid_cached_chat_prompt_by_version(langfuse):
    prompt_name = "test_get_valid_cached_chat_prompt_by_version"
    prompt = Prompt_Chat(
        name=prompt_name,
        version=1,
        prompt=[{"role": "system", "content": "Make me laugh"}],
        labels=["test"],
        type="chat",
        config={},
        tags=[],
    )
    prompt_client = ChatPromptClient(prompt)

    mock_server_call = langfuse.api.prompts.get
    mock_server_call.return_value = prompt

    result_call_1 = langfuse.get_prompt(prompt_name, version=1)
    assert mock_server_call.call_count == 1
    assert result_call_1 == prompt_client

    result_call_2 = langfuse.get_prompt(prompt_name, version=1)
    assert mock_server_call.call_count == 1
    assert result_call_2 == prompt_client


# Should return cached chat prompt if fetching the default prompt or the 'production' labeled one
def test_get_valid_cached_production_chat_prompt(langfuse):
    prompt_name = "test_get_valid_cached_production_chat_prompt"
    prompt = Prompt_Chat(
        name=prompt_name,
        version=1,
        prompt=[{"role": "system", "content": "Make me laugh"}],
        labels=["test"],
        type="chat",
        config={},
        tags=[],
    )
    prompt_client = ChatPromptClient(prompt)

    mock_server_call = langfuse.api.prompts.get
    mock_server_call.return_value = prompt

    result_call_1 = langfuse.get_prompt(prompt_name)
    assert mock_server_call.call_count == 1
    assert result_call_1 == prompt_client

    result_call_2 = langfuse.get_prompt(prompt_name, label="production")
    assert mock_server_call.call_count == 1
    assert result_call_2 == prompt_client


# Should return cached chat prompt if not expired
def test_get_valid_cached_chat_prompt(langfuse):
    prompt_name = "test_get_valid_cached_chat_prompt"
    prompt = Prompt_Chat(
        name=prompt_name,
        version=1,
        prompt=[{"role": "system", "content": "Make me laugh"}],
        labels=[],
        type="chat",
        config={},
        tags=[],
    )
    prompt_client = ChatPromptClient(prompt)

    mock_server_call = langfuse.api.prompts.get
    mock_server_call.return_value = prompt

    result_call_1 = langfuse.get_prompt(prompt_name)
    assert mock_server_call.call_count == 1
    assert result_call_1 == prompt_client

    result_call_2 = langfuse.get_prompt(prompt_name)
    assert mock_server_call.call_count == 1
    assert result_call_2 == prompt_client


# Should refetch and return new prompt if cached one is expired according to custom TTL
@patch.object(PromptCacheItem, "get_epoch_seconds")
def test_get_fresh_prompt_when_expired_cache_custom_ttl(mock_time, langfuse: Langfuse):
    mock_time.return_value = 0
    ttl_seconds = 20

    prompt_name = "test_get_fresh_prompt_when_expired_cache_custom_ttl"
    prompt = Prompt_Text(
        name=prompt_name,
        version=1,
        prompt="Make me laugh",
        config={"temperature": 0.9},
        labels=[],
        type="text",
        tags=[],
    )
    prompt_client = TextPromptClient(prompt)

    mock_server_call = langfuse.api.prompts.get
    mock_server_call.return_value = prompt

    result_call_1 = langfuse.get_prompt(prompt_name, cache_ttl_seconds=ttl_seconds)
    assert mock_server_call.call_count == 1
    assert result_call_1 == prompt_client

    # Set time to just BEFORE cache expiry
    mock_time.return_value = ttl_seconds - 1

    result_call_2 = langfuse.get_prompt(prompt_name)
    assert mock_server_call.call_count == 1  # No new call
    assert result_call_2 == prompt_client

    # Set time to just AFTER cache expiry
    mock_time.return_value = ttl_seconds + 1

    result_call_3 = langfuse.get_prompt(prompt_name)

    while True:
        if langfuse._resources.prompt_cache._task_manager.active_tasks() == 0:
            break
        sleep(0.1)

    assert mock_server_call.call_count == 2  # New call
    assert result_call_3 == prompt_client


# Should disable caching when cache_ttl_seconds is set to 0
@patch.object(PromptCacheItem, "get_epoch_seconds")
def test_disable_caching_when_ttl_zero(mock_time, langfuse: Langfuse):
    mock_time.return_value = 0
    prompt_name = "test_disable_caching_when_ttl_zero"

    # Initial prompt
    prompt1 = Prompt_Text(
        name=prompt_name,
        version=1,
        prompt="Make me laugh",
        labels=[],
        type="text",
        config={},
        tags=[],
    )

    # Updated prompts
    prompt2 = Prompt_Text(
        name=prompt_name,
        version=2,
        prompt="Tell me a joke",
        labels=[],
        type="text",
        config={},
        tags=[],
    )
    prompt3 = Prompt_Text(
        name=prompt_name,
        version=3,
        prompt="Share a funny story",
        labels=[],
        type="text",
        config={},
        tags=[],
    )

    mock_server_call = langfuse.api.prompts.get
    mock_server_call.side_effect = [prompt1, prompt2, prompt3]

    # First call
    result1 = langfuse.get_prompt(prompt_name, cache_ttl_seconds=0)
    assert mock_server_call.call_count == 1
    assert result1 == TextPromptClient(prompt1)

    # Second call
    result2 = langfuse.get_prompt(prompt_name, cache_ttl_seconds=0)
    assert mock_server_call.call_count == 2
    assert result2 == TextPromptClient(prompt2)

    # Third call
    result3 = langfuse.get_prompt(prompt_name, cache_ttl_seconds=0)
    assert mock_server_call.call_count == 3
    assert result3 == TextPromptClient(prompt3)

    # Verify that all results are different
    assert result1 != result2 != result3


# Should return stale prompt immediately if cached one is expired according to default TTL and add to refresh promise map
@patch.object(PromptCacheItem, "get_epoch_seconds")
def test_get_stale_prompt_when_expired_cache_default_ttl(mock_time, langfuse: Langfuse):
    import logging

    logging.basicConfig(level=logging.DEBUG)
    mock_time.return_value = 0

    prompt_name = "test_get_stale_prompt_when_expired_cache_default_ttl"
    prompt = Prompt_Text(
        name=prompt_name,
        version=1,
        prompt="Make me laugh",
        labels=[],
        type="text",
        config={},
        tags=[],
    )
    prompt_client = TextPromptClient(prompt)

    mock_server_call = langfuse.api.prompts.get
    mock_server_call.return_value = prompt

    result_call_1 = langfuse.get_prompt(prompt_name)
    assert mock_server_call.call_count == 1
    assert result_call_1 == prompt_client

    # Update the version of the returned mocked prompt
    updated_prompt = Prompt_Text(
        name=prompt_name,
        version=2,
        prompt="Make me laugh",
        labels=[],
        type="text",
        config={},
        tags=[],
    )
    mock_server_call.return_value = updated_prompt

    # Set time to just AFTER cache expiry
    mock_time.return_value = DEFAULT_PROMPT_CACHE_TTL_SECONDS + 1

    stale_result = langfuse.get_prompt(prompt_name)
    assert stale_result == prompt_client

    # Ensure that only one refresh is triggered despite multiple calls
    # Cannot check for value as the prompt might have already been updated
    langfuse.get_prompt(prompt_name)
    langfuse.get_prompt(prompt_name)
    langfuse.get_prompt(prompt_name)
    langfuse.get_prompt(prompt_name)

    while True:
        if langfuse._resources.prompt_cache._task_manager.active_tasks() == 0:
            break
        sleep(0.1)

    assert mock_server_call.call_count == 2  # Only one new call to server

    # Check that the prompt has been updated after refresh
    updated_result = langfuse.get_prompt(prompt_name)
    assert updated_result.version == 2
    assert updated_result == TextPromptClient(updated_prompt)


# Should refetch and return new prompt if cached one is expired according to default TTL
@patch.object(PromptCacheItem, "get_epoch_seconds")
def test_get_fresh_prompt_when_expired_cache_default_ttl(mock_time, langfuse: Langfuse):
    mock_time.return_value = 0

    prompt_name = "test_get_fresh_prompt_when_expired_cache_default_ttl"
    prompt = Prompt_Text(
        name=prompt_name,
        version=1,
        prompt="Make me laugh",
        labels=[],
        type="text",
        config={},
        tags=[],
    )
    prompt_client = TextPromptClient(prompt)

    mock_server_call = langfuse.api.prompts.get
    mock_server_call.return_value = prompt

    result_call_1 = langfuse.get_prompt(prompt_name)
    assert mock_server_call.call_count == 1
    assert result_call_1 == prompt_client

    # Set time to just BEFORE cache expiry
    mock_time.return_value = DEFAULT_PROMPT_CACHE_TTL_SECONDS - 1

    result_call_2 = langfuse.get_prompt(prompt_name)
    assert mock_server_call.call_count == 1  # No new call
    assert result_call_2 == prompt_client

    # Set time to just AFTER cache expiry
    mock_time.return_value = DEFAULT_PROMPT_CACHE_TTL_SECONDS + 1

    result_call_3 = langfuse.get_prompt(prompt_name)
    while True:
        if langfuse._resources.prompt_cache._task_manager.active_tasks() == 0:
            break
        sleep(0.1)

    assert mock_server_call.call_count == 2  # New call
    assert result_call_3 == prompt_client

# Should refetch and return new prompt if cached one is expired according overridden to default TTL
@patch.object(PromptCacheItem, "get_epoch_seconds")
def test_get_fresh_prompt_when_expired_cache_overridden_default_ttl(mock_time, langfuse_with_override_default_cache: Langfuse):
    langfuse = langfuse_with_override_default_cache

    mock_time.return_value = 0

    prompt_name = "test_get_fresh_prompt_when_expired_cache_overridden_default_ttl"
    prompt = Prompt_Text(
        name=prompt_name,
        version=1,
        prompt="Make me laugh",
        labels=[],
        type="text",
        config={},
        tags=[],
    )
    prompt_client = TextPromptClient(prompt)

    mock_server_call = langfuse.api.prompts.get
    mock_server_call.return_value = prompt

    result_call_1 = langfuse.get_prompt(prompt_name)
    assert mock_server_call.call_count == 1
    assert result_call_1 == prompt_client

    # Set time to just BEFORE cache expiry using DEFAULT TTL (should NOT expire)
    mock_time.return_value = DEFAULT_PROMPT_CACHE_TTL_SECONDS - 1

    result_call_2 = langfuse.get_prompt(prompt_name)
    assert mock_server_call.call_count == 1  # No new call - cache still valid
    assert result_call_2 == prompt_client

    # Set time to just AFTER cache expiry using DEFAULT TTL (should NOT expire with overridden TTL)
    mock_time.return_value = DEFAULT_PROMPT_CACHE_TTL_SECONDS + 1

    result_call_3 = langfuse.get_prompt(prompt_name)
    assert mock_server_call.call_count == 1  # Still no new call - overridden TTL is longer
    assert result_call_3 == prompt_client

    # Set time to just BEFORE cache expiry using OVERRIDDEN TTL (should NOT expire)
    mock_time.return_value = OVERRIDE_DEFAULT_PROMPT_CACHE_TTL_SECONDS - 1

    result_call_4 = langfuse.get_prompt(prompt_name)
    assert mock_server_call.call_count == 1  # No new call
    assert result_call_4 == prompt_client

    # Set time to just AFTER cache expiry using OVERRIDDEN TTL (should expire)
    mock_time.return_value = OVERRIDE_DEFAULT_PROMPT_CACHE_TTL_SECONDS + 1

    result_call_5 = langfuse.get_prompt(prompt_name)
    while True:
        if langfuse._resources.prompt_cache._task_manager.active_tasks() == 0:
            break
        sleep(0.1)

    assert mock_server_call.call_count == 2  # New call - cache expired at overridden TTL
    assert result_call_5 == prompt_client

    # Verify that the overridden TTL is actually being used by checking the cache configuration
    assert langfuse._resources.prompt_cache._default_cache_ttl_seconds == OVERRIDE_DEFAULT_PROMPT_CACHE_TTL_SECONDS
    assert langfuse._resources.prompt_cache._default_cache_ttl_seconds != DEFAULT_PROMPT_CACHE_TTL_SECONDS


# Should return expired prompt if refetch fails
@patch.object(PromptCacheItem, "get_epoch_seconds")
def test_get_expired_prompt_when_failing_fetch(mock_time, langfuse: Langfuse):
    mock_time.return_value = 0

    prompt_name = "test_get_expired_prompt_when_failing_fetch"
    prompt = Prompt_Text(
        name=prompt_name,
        version=1,
        prompt="Make me laugh",
        labels=[],
        type="text",
        config={},
        tags=[],
    )
    prompt_client = TextPromptClient(prompt)

    mock_server_call = langfuse.api.prompts.get
    mock_server_call.return_value = prompt

    result_call_1 = langfuse.get_prompt(prompt_name)
    assert mock_server_call.call_count == 1
    assert result_call_1 == prompt_client

    # Set time to just AFTER cache expiry
    mock_time.return_value = DEFAULT_PROMPT_CACHE_TTL_SECONDS + 1

    mock_server_call.side_effect = Exception("Server error")

    result_call_2 = langfuse.get_prompt(prompt_name, max_retries=1)
    while True:
        if langfuse._resources.prompt_cache._task_manager.active_tasks() == 0:
            break
        sleep(0.1)

    assert mock_server_call.call_count == 3
    assert result_call_2 == prompt_client


# Should fetch new prompt if version changes
def test_get_fresh_prompt_when_version_changes(langfuse: Langfuse):
    prompt_name = "test_get_fresh_prompt_when_version_changes"
    prompt = Prompt_Text(
        name=prompt_name,
        version=1,
        prompt="Make me laugh",
        labels=[],
        type="text",
        config={},
        tags=[],
    )
    prompt_client = TextPromptClient(prompt)

    mock_server_call = langfuse.api.prompts.get
    mock_server_call.return_value = prompt

    result_call_1 = langfuse.get_prompt(prompt_name, version=1)
    assert mock_server_call.call_count == 1
    assert result_call_1 == prompt_client

    version_changed_prompt = Prompt_Text(
        name=prompt_name,
        version=2,
        labels=[],
        prompt="Make me laugh",
        type="text",
        config={},
        tags=[],
    )
    version_changed_prompt_client = TextPromptClient(version_changed_prompt)
    mock_server_call.return_value = version_changed_prompt

    result_call_2 = langfuse.get_prompt(prompt_name, version=2)
    assert mock_server_call.call_count == 2
    assert result_call_2 == version_changed_prompt_client


def test_do_not_return_fallback_if_fetch_success():
    langfuse = Langfuse()
    prompt_name = create_uuid()
    prompt_client = langfuse.create_prompt(
        name=prompt_name,
        prompt="test prompt",
        labels=["production"],
    )

    second_prompt_client = langfuse.get_prompt(prompt_name, fallback="fallback")

    assert prompt_client.name == second_prompt_client.name
    assert prompt_client.version == second_prompt_client.version
    assert prompt_client.prompt == second_prompt_client.prompt
    assert prompt_client.config == second_prompt_client.config
    assert prompt_client.config == {}


def test_fallback_text_prompt():
    langfuse = Langfuse()

    fallback_text_prompt = "this is a fallback text prompt with {{variable}}"

    # Should throw an error if prompt not found and no fallback provided
    with pytest.raises(Exception):
        langfuse.get_prompt("nonexistent_prompt")

    prompt = langfuse.get_prompt("nonexistent_prompt", fallback=fallback_text_prompt)

    assert prompt.prompt == fallback_text_prompt
    assert (
        prompt.compile(variable="value") == "this is a fallback text prompt with value"
    )


def test_fallback_chat_prompt():
    langfuse = Langfuse()
    fallback_chat_prompt = [
        {"role": "system", "content": "fallback system"},
        {"role": "user", "content": "fallback user name {{name}}"},
    ]

    # Should throw an error if prompt not found and no fallback provided
    with pytest.raises(Exception):
        langfuse.get_prompt("nonexistent_chat_prompt", type="chat")

    prompt = langfuse.get_prompt(
        "nonexistent_chat_prompt", type="chat", fallback=fallback_chat_prompt
    )

    # Check that the prompt structure contains the fallback data (allowing for internal formatting)
    assert len(prompt.prompt) == len(fallback_chat_prompt)
    assert all(msg["type"] == "message" for msg in prompt.prompt)
    assert prompt.prompt[0]["role"] == "system"
    assert prompt.prompt[0]["content"] == "fallback system"
    assert prompt.prompt[1]["role"] == "user"
    assert prompt.prompt[1]["content"] == "fallback user name {{name}}"
    assert prompt.compile(name="Jane") == [
        {"role": "system", "content": "fallback system"},
        {"role": "user", "content": "fallback user name Jane"},
    ]


def test_do_not_link_observation_if_fallback():
    langfuse = Langfuse()

    fallback_text_prompt = "this is a fallback text prompt with {{variable}}"

    # Should throw an error if prompt not found and no fallback provided
    with pytest.raises(Exception):
        langfuse.get_prompt("nonexistent_prompt")

    prompt = langfuse.get_prompt("nonexistent_prompt", fallback=fallback_text_prompt)

    generation = langfuse.start_generation(
        name="mygen", prompt=prompt, input="this is a test input"
    ).end()
    langfuse.flush()

    api = get_api()
    trace = api.trace.get(generation.trace_id)

    assert len(trace.observations) == 1
    assert trace.observations[0].prompt_id is None


def test_variable_names_on_content_with_variable_names():
    langfuse = Langfuse()

    prompt_client = langfuse.create_prompt(
        name="test_variable_names_1",
        prompt="test prompt with var names {{ var1 }} {{ var2 }}",
        labels=["production"],
        type="text",
    )

    second_prompt_client = langfuse.get_prompt("test_variable_names_1")

    assert prompt_client.name == second_prompt_client.name
    assert prompt_client.version == second_prompt_client.version
    assert prompt_client.prompt == second_prompt_client.prompt
    assert prompt_client.labels == ["production", "latest"]

    var_names = second_prompt_client.variables

    assert var_names == ["var1", "var2"]


def test_variable_names_on_content_with_no_variable_names():
    langfuse = Langfuse()

    prompt_client = langfuse.create_prompt(
        name="test_variable_names_2",
        prompt="test prompt with no var names",
        labels=["production"],
        type="text",
    )

    second_prompt_client = langfuse.get_prompt("test_variable_names_2")

    assert prompt_client.name == second_prompt_client.name
    assert prompt_client.version == second_prompt_client.version
    assert prompt_client.prompt == second_prompt_client.prompt
    assert prompt_client.labels == ["production", "latest"]

    var_names = second_prompt_client.variables

    assert var_names == []


def test_variable_names_on_content_with_variable_names_chat_messages():
    langfuse = Langfuse()

    prompt_client = langfuse.create_prompt(
        name="test_variable_names_3",
        prompt=[
            {
                "role": "system",
                "content": "test prompt with template vars {{ var1 }} {{ var2 }}",
            },
            {"role": "user", "content": "test prompt 2 with template vars {{ var3 }}"},
        ],
        labels=["production"],
        type="chat",
    )

    second_prompt_client = langfuse.get_prompt("test_variable_names_3")

    assert prompt_client.name == second_prompt_client.name
    assert prompt_client.version == second_prompt_client.version
    assert prompt_client.prompt == second_prompt_client.prompt
    assert prompt_client.labels == ["production", "latest"]

    var_names = second_prompt_client.variables

    assert var_names == ["var1", "var2", "var3"]


def test_variable_names_on_content_with_no_variable_names_chat_messages():
    langfuse = Langfuse()
    prompt_name = "test_variable_names_on_content_with_no_variable_names_chat_messages"

    prompt_client = langfuse.create_prompt(
        name=prompt_name,
        prompt=[
            {"role": "system", "content": "test prompt with no template vars"},
            {"role": "user", "content": "test prompt 2 with no template vars"},
        ],
        labels=["production"],
        type="chat",
    )

    second_prompt_client = langfuse.get_prompt(prompt_name)

    assert prompt_client.name == second_prompt_client.name
    assert prompt_client.version == second_prompt_client.version
    assert prompt_client.prompt == second_prompt_client.prompt
    assert prompt_client.labels == ["production", "latest"]

    var_names = second_prompt_client.variables

    assert var_names == []


def test_update_prompt():
    langfuse = Langfuse()
    prompt_name = create_uuid()

    # Create initial prompt
    langfuse.create_prompt(
        name=prompt_name,
        prompt="test prompt",
        labels=["production"],
    )

    # Update prompt labels
    updated_prompt = langfuse.update_prompt(
        name=prompt_name,
        version=1,
        new_labels=["john", "doe"],
    )

    # Fetch prompt after update (should be invalidated)
    fetched_prompt = langfuse.get_prompt(prompt_name)

    # Verify the fetched prompt matches the updated values
    assert fetched_prompt.name == prompt_name
    assert fetched_prompt.version == 1
    print(f"Fetched prompt labels: {fetched_prompt.labels}")
    print(f"Updated prompt labels: {updated_prompt.labels}")

    # production was set by the first call, latest is managed and set by Langfuse
    expected_labels = sorted(["latest", "doe", "production", "john"])
    assert sorted(fetched_prompt.labels) == expected_labels
    assert sorted(updated_prompt.labels) == expected_labels
