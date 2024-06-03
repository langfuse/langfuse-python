import pytest
from unittest.mock import Mock, patch

import openai
from langfuse.client import Langfuse
from langfuse.prompt_cache import PromptCacheItem, DEFAULT_PROMPT_CACHE_TTL_SECONDS
from tests.utils import create_uuid, get_api
from langfuse.api.resources.prompts import Prompt_Text, Prompt_Chat
from langfuse.model import TextPromptClient, ChatPromptClient


def test_create_prompt():
    langfuse = Langfuse()
    prompt_name = create_uuid()
    prompt_client = langfuse.create_prompt(
        name=prompt_name,
        prompt="test prompt",
        labels=["production"],
    )

    second_prompt_client = langfuse.get_prompt(prompt_name)

    assert prompt_client.name == second_prompt_client.name
    assert prompt_client.version == second_prompt_client.version
    assert prompt_client.prompt == second_prompt_client.prompt
    assert prompt_client.config == second_prompt_client.config
    assert prompt_client.config == {}


def test_create_prompt_with_is_active():
    # Backward compatibility test for is_active
    langfuse = Langfuse()
    prompt_name = create_uuid()
    prompt_client = langfuse.create_prompt(
        name=prompt_name, prompt="test prompt", is_active=True
    )

    second_prompt_client = langfuse.get_prompt(prompt_name)

    assert prompt_client.name == second_prompt_client.name
    assert prompt_client.version == second_prompt_client.version
    assert prompt_client.prompt == second_prompt_client.prompt
    assert prompt_client.config == second_prompt_client.config
    assert prompt_client.labels == ["production", "latest"]
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
    )

    second_prompt_client = langfuse.get_prompt(prompt_name, type="chat")

    # Create a test generation
    completion = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=prompt_client.compile(animal="dog", occupation="doctor"),
    )

    assert len(completion.choices) > 0

    assert prompt_client.name == second_prompt_client.name
    assert prompt_client.version == second_prompt_client.version
    assert prompt_client.prompt == second_prompt_client.prompt
    assert prompt_client.config == second_prompt_client.config
    assert prompt_client.labels == ["production", "latest"]
    assert prompt_client.tags == second_prompt_client.tags
    assert prompt_client.config == {}


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

    prompt_client = langfuse.create_prompt(
        name="test",
        prompt='Hello, {{target}}! I hope you are {{state}}. {{undefined_variable}}. And here is some JSON that should not be compiled: {{ "key": "value" }} \
            Here is a custom var for users using str.format instead of the mustache-style double curly braces: {custom_var}',
        is_active=True,
    )

    second_prompt_client = langfuse.get_prompt("test")

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

    prompt_client = langfuse.create_prompt(
        name="test",
        prompt="Hello, {{ some_json }}",
        is_active=True,
    )

    second_prompt_client = langfuse.get_prompt("test")

    assert prompt_client.name == second_prompt_client.name
    assert prompt_client.version == second_prompt_client.version
    assert prompt_client.prompt == second_prompt_client.prompt
    assert prompt_client.labels == ["production", "latest"]

    some_json = '{"key": "value"}'
    compiled = second_prompt_client.compile(some_json=some_json)

    assert compiled == 'Hello, {"key": "value"}'


def test_compiling_prompt_with_content_as_variable_name():
    langfuse = Langfuse()

    prompt_client = langfuse.create_prompt(
        name="test",
        prompt="Hello, {{ content }}!",
        is_active=True,
    )

    second_prompt_client = langfuse.get_prompt("test")

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
        is_active=True,
        config=None,
    )

    prompt = langfuse.get_prompt("test_null_config")

    assert prompt.config == {}


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

    first_prompt_client = langfuse.get_prompt(prompt_name, 1)
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
        is_active=True,
        config={"temperature": 0.5},
    )

    prompt = langfuse.get_prompt("test")

    prompt_str = prompt.compile(target="world", state="great")
    assert prompt_str == "Hello, world! I hope you are great."
    assert prompt.config == {"temperature": 0.5}

    generation = langfuse.generation(input=prompt_str, prompt=prompt)

    # to check that these do not error
    generation.update(prompt=prompt)
    generation.end(prompt=prompt)

    langfuse.flush()

    api = get_api()

    trace_id = langfuse.get_trace_id()

    trace = api.trace.get(trace_id)

    assert len(trace.observations) == 1

    generation = trace.observations[0]
    assert generation.prompt_id is not None

    observation = api.observations.get(generation.id)

    assert observation.prompt_id is not None


@pytest.fixture
def langfuse():
    langfuse_instance = Langfuse()
    langfuse_instance.client = Mock()
    langfuse_instance.log = Mock()

    return langfuse_instance


# Fetching a new prompt when nothing in cache
def test_get_fresh_prompt(langfuse):
    prompt_name = "test"
    prompt = Prompt_Text(
        name=prompt_name,
        version=1,
        prompt="Make me laugh",
        type="text",
        labels=[],
        config={},
    )

    mock_server_call = langfuse.client.prompts.get
    mock_server_call.return_value = prompt

    result = langfuse.get_prompt(prompt_name)
    mock_server_call.assert_called_once_with(prompt_name, version=None, label=None)

    assert result == TextPromptClient(prompt)


# Should throw an error if prompt name is unspecified
def test_throw_if_name_unspecified(langfuse):
    prompt_name = ""

    with pytest.raises(ValueError) as exc_info:
        langfuse.get_prompt(prompt_name)

    assert "Prompt name cannot be empty" in str(exc_info.value)


# Should throw an error if nothing in cache and fetch fails
def test_throw_when_failing_fetch_and_no_cache(langfuse):
    prompt_name = "test"

    mock_server_call = langfuse.client.prompts.get
    mock_server_call.side_effect = Exception("Prompt not found")

    with pytest.raises(Exception) as exc_info:
        langfuse.get_prompt(prompt_name)

    assert "Prompt not found" in str(exc_info.value)
    langfuse.log.exception.assert_called_once()


# Should throw an error if cache_ttl_seconds is passed as positional rather than keyword argument
def test_throw_if_cache_ttl_seconds_positional_argument(langfuse):
    prompt_name = "test"
    version = 1
    ttl_seconds = 20

    with pytest.raises(TypeError) as exc_info:
        langfuse.get_prompt(prompt_name, version, ttl_seconds)

    assert "positional arguments" in str(exc_info.value)


# Should return cached prompt if not expired
def test_get_valid_cached_prompt(langfuse):
    prompt_name = "test"
    prompt = Prompt_Text(
        name=prompt_name,
        version=1,
        prompt="Make me laugh",
        type="text",
        labels=[],
        config={},
    )
    prompt_client = TextPromptClient(prompt)

    mock_server_call = langfuse.client.prompts.get
    mock_server_call.return_value = prompt

    result_call_1 = langfuse.get_prompt(prompt_name)
    assert mock_server_call.call_count == 1
    assert result_call_1 == prompt_client

    result_call_2 = langfuse.get_prompt(prompt_name)
    assert mock_server_call.call_count == 1
    assert result_call_2 == prompt_client


# Should return cached chat prompt if not expired when fetching by label
def test_get_valid_cached_chat_prompt_by_label(langfuse):
    prompt_name = "test"
    prompt = Prompt_Chat(
        name=prompt_name,
        version=1,
        prompt=[{"role": "system", "content": "Make me laugh"}],
        labels=["test"],
        type="chat",
        config={},
    )
    prompt_client = ChatPromptClient(prompt)

    mock_server_call = langfuse.client.prompts.get
    mock_server_call.return_value = prompt

    result_call_1 = langfuse.get_prompt(prompt_name, label="test")
    assert mock_server_call.call_count == 1
    assert result_call_1 == prompt_client

    result_call_2 = langfuse.get_prompt(prompt_name, label="test")
    assert mock_server_call.call_count == 1
    assert result_call_2 == prompt_client


# Should return cached chat prompt if not expired when fetching by version
def test_get_valid_cached_chat_prompt_by_version(langfuse):
    prompt_name = "test"
    prompt = Prompt_Chat(
        name=prompt_name,
        version=1,
        prompt=[{"role": "system", "content": "Make me laugh"}],
        labels=["test"],
        type="chat",
        config={},
    )
    prompt_client = ChatPromptClient(prompt)

    mock_server_call = langfuse.client.prompts.get
    mock_server_call.return_value = prompt

    result_call_1 = langfuse.get_prompt(prompt_name, version=1)
    assert mock_server_call.call_count == 1
    assert result_call_1 == prompt_client

    result_call_2 = langfuse.get_prompt(prompt_name, version=1)
    assert mock_server_call.call_count == 1
    assert result_call_2 == prompt_client


# Should return cached chat prompt if fetching the default prompt or the 'production' labeled one
def test_get_valid_cached_production_chat_prompt(langfuse):
    prompt_name = "test"
    prompt = Prompt_Chat(
        name=prompt_name,
        version=1,
        prompt=[{"role": "system", "content": "Make me laugh"}],
        labels=["test"],
        type="chat",
        config={},
    )
    prompt_client = ChatPromptClient(prompt)

    mock_server_call = langfuse.client.prompts.get
    mock_server_call.return_value = prompt

    result_call_1 = langfuse.get_prompt(prompt_name)
    assert mock_server_call.call_count == 1
    assert result_call_1 == prompt_client

    result_call_2 = langfuse.get_prompt(prompt_name, label="production")
    assert mock_server_call.call_count == 1
    assert result_call_2 == prompt_client


# Should return cached chat prompt if not expired
def test_get_valid_cached_chat_prompt(langfuse):
    prompt_name = "test"
    prompt = Prompt_Chat(
        name=prompt_name,
        version=1,
        prompt=[{"role": "system", "content": "Make me laugh"}],
        labels=[],
        type="chat",
        config={},
    )
    prompt_client = ChatPromptClient(prompt)

    mock_server_call = langfuse.client.prompts.get
    mock_server_call.return_value = prompt

    result_call_1 = langfuse.get_prompt(prompt_name)
    assert mock_server_call.call_count == 1
    assert result_call_1 == prompt_client

    result_call_2 = langfuse.get_prompt(prompt_name)
    assert mock_server_call.call_count == 1
    assert result_call_2 == prompt_client


# Should refetch and return new prompt if cached one is expired according to custom TTL
@patch.object(PromptCacheItem, "get_epoch_seconds")
def test_get_fresh_prompt_when_expired_cache_custom_ttl(mock_time, langfuse):
    mock_time.return_value = 0
    ttl_seconds = 20

    prompt_name = "test"
    prompt = Prompt_Text(
        name=prompt_name,
        version=1,
        prompt="Make me laugh",
        config={"temperature": 0.9},
        labels=[],
        type="text",
    )
    prompt_client = TextPromptClient(prompt)

    mock_server_call = langfuse.client.prompts.get
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
    assert mock_server_call.call_count == 2  # New call
    assert result_call_3 == prompt_client


# Should refetch and return new prompt if cached one is expired according to default TTL
@patch.object(PromptCacheItem, "get_epoch_seconds")
def test_get_fresh_prompt_when_expired_cache_default_ttl(mock_time, langfuse):
    mock_time.return_value = 0

    prompt_name = "test"
    prompt = Prompt_Text(
        name=prompt_name,
        version=1,
        prompt="Make me laugh",
        labels=[],
        type="text",
        config={},
    )
    prompt_client = TextPromptClient(prompt)

    mock_server_call = langfuse.client.prompts.get
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
    assert mock_server_call.call_count == 2  # New call
    assert result_call_3 == prompt_client


# Should return expired prompt if refetch fails
@patch.object(PromptCacheItem, "get_epoch_seconds")
def test_get_expired_prompt_when_failing_fetch(mock_time, langfuse):
    mock_time.return_value = 0

    prompt_name = "test"
    prompt = Prompt_Text(
        name=prompt_name,
        version=1,
        prompt="Make me laugh",
        labels=[],
        type="text",
        config={},
    )
    prompt_client = TextPromptClient(prompt)

    mock_server_call = langfuse.client.prompts.get
    mock_server_call.return_value = prompt

    result_call_1 = langfuse.get_prompt(prompt_name)
    assert mock_server_call.call_count == 1
    assert result_call_1 == prompt_client

    # Set time to just AFTER cache expiry
    mock_time.return_value = DEFAULT_PROMPT_CACHE_TTL_SECONDS + 1

    mock_server_call.side_effect = Exception("Server error")

    result_call_2 = langfuse.get_prompt(prompt_name)
    assert mock_server_call.call_count == 2
    assert result_call_2 == prompt_client


# Should fetch new prompt if version changes
def test_get_fresh_prompt_when_version_changes(langfuse):
    prompt_name = "test"
    prompt = Prompt_Text(
        name=prompt_name,
        version=1,
        prompt="Make me laugh",
        labels=[],
        type="text",
        config={},
    )
    prompt_client = TextPromptClient(prompt)

    mock_server_call = langfuse.client.prompts.get
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
    )
    version_changed_prompt_client = TextPromptClient(version_changed_prompt)
    mock_server_call.return_value = version_changed_prompt

    result_call_2 = langfuse.get_prompt(prompt_name, version=2)
    assert mock_server_call.call_count == 2
    assert result_call_2 == version_changed_prompt_client


def test_tags_feature():
    langfuse = Langfuse()
    # Test creating a prompt without tags
    prompt_name = create_uuid()
    langfuse.create_prompt(
        name=prompt_name,
        prompt="test prompt without tags",
        labels=["production"],
    )

    v1_prompt = langfuse.get_prompt(prompt_name)
    assert v1_prompt.tags == []

    # Test updating a prompt to add tags
    langfuse.create_prompt(
        name=prompt_name,
        prompt="test prompt to update with tags",
        labels=["production"],
        tags=["new_tag1", "new_tag2"],
    )

    v2_prompt = langfuse.get_prompt(prompt_name)
    assert v2_prompt.tags == ["new_tag1", "new_tag2"]
    v1_prompt = langfuse.get_prompt(prompt_name, version=1)
    assert v1_prompt.tags == ["new_tag1", "new_tag2"]

    # Test update to add tags
    langfuse.create_prompt(
        name=prompt_name,
        prompt="test prompt to update with additional tags",
        labels=["production"],
        tags=["new_tag2", "new_tag3", "new_tag4"],
    )
    v3_prompt = langfuse.get_prompt(prompt_name)
    assert v3_prompt.tags == [
        "new_tag1",
        "new_tag2",
        "new_tag3",
        "new_tag4",
    ]
    v2_prompt = langfuse.get_prompt(prompt_name, version=2)
    assert v2_prompt.tags == [
        "new_tag1",
        "new_tag2",
        "new_tag3",
        "new_tag4",
    ]
    v1_prompt = langfuse.get_prompt(prompt_name, version=1)
    assert v1_prompt.tags == [
        "new_tag1",
        "new_tag2",
        "new_tag3",
        "new_tag4",
    ]

    # create new version without tags
    langfuse.create_prompt(
        name=prompt_name,
        prompt="test prompt without tags",
        labels=["production"],
    )
    v4_prompt = langfuse.get_prompt(prompt_name)
    assert v4_prompt.tags == [
        "new_tag1",
        "new_tag2",
        "new_tag3",
        "new_tag4",
    ]

    # remove tags
    langfuse.create_prompt(
        name=prompt_name,
        prompt="test prompt to update with no tags",
        labels=["production"],
        tags=[],
    )
    v5_prompt = langfuse.get_prompt(prompt_name)
    assert v5_prompt.tags == []
    v4_prompt = langfuse.get_prompt(prompt_name, version=4)
    assert v4_prompt.tags == []
    v3_prompt = langfuse.get_prompt(prompt_name, version=3)
    assert v3_prompt.tags == []
    v2_prompt = langfuse.get_prompt(prompt_name, version=2)
    assert v2_prompt.tags == []
    v1_prompt = langfuse.get_prompt(prompt_name, version=1)
    assert v1_prompt.tags == []
