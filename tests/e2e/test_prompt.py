import pytest

from langfuse._client.client import Langfuse
from tests.support.utils import create_uuid, get_api


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

    generation = langfuse.start_observation(
        as_type="generation",
        name="mygen",
        input=prompt_str,
        prompt=prompt,
    ).end()

    # to check that these do not error
    generation.update(prompt=prompt)

    langfuse.flush()

    api = get_api()

    trace = api.trace.get(generation.trace_id)

    assert len(trace.observations) == 1

    generation = trace.observations[0]
    assert generation.prompt_id is not None

    observation = api.legacy.observations_v1.get(generation.id)

    assert observation.prompt_id is not None


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

    generation = langfuse.start_observation(
        as_type="generation",
        name="mygen",
        prompt=prompt,
        input="this is a test input",
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


def test_update_prompt_in_folder():
    langfuse = Langfuse()
    prompt_name = f"some-folder/{create_uuid()}"

    # Create initial prompt
    langfuse.create_prompt(
        name=prompt_name,
        prompt="test prompt",
        labels=["production"],
    )

    old_prompt_obj = langfuse.get_prompt(prompt_name)

    updated_prompt = langfuse.update_prompt(
        name=old_prompt_obj.name,
        version=old_prompt_obj.version,
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
