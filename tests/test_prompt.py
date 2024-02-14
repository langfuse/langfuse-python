import pytest
from unittest.mock import Mock, patch


from langfuse.api.resources.prompts.types.prompt import Prompt
from langfuse.client import Langfuse, PromptClient
from langfuse.prompt_cache import PromptCacheItem, DEFAULT_PROMPT_CACHE_TTL_SECONDS
from tests.utils import create_uuid, get_api


def test_create_prompt():
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
    assert prompt_client.config == {}


def test_compiling_prompt():
    langfuse = Langfuse()

    prompt_client = langfuse.create_prompt(
        name="test",
        prompt="Hello, {{target}}! I hope you are {{state}}.",
        is_active=True,
    )

    second_prompt_client = langfuse.get_prompt("test")

    assert prompt_client.name == second_prompt_client.name
    assert prompt_client.version == second_prompt_client.version
    assert prompt_client.prompt == second_prompt_client.prompt

    assert (
        second_prompt_client.compile(target="world", state="great")
        == "Hello, world! I hope you are great."
    )

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

    langfuse.generation(input=prompt_str, prompt=prompt)

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
    prompt = Prompt(name=prompt_name, version=1, prompt="Make me laugh")

    mock_server_call = langfuse.client.prompts.get
    mock_server_call.return_value = prompt

    result = langfuse.get_prompt(prompt_name)
    mock_server_call.assert_called_once_with(name=prompt_name, version=None)

    assert result == PromptClient(prompt)


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
    prompt = Prompt(name=prompt_name, version=1, prompt="Make me laugh")
    prompt_client = PromptClient(prompt)

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
    prompt = Prompt(
        name=prompt_name, version=1, prompt="Make me laugh", config={"temperature": 0.9}
    )
    prompt_client = PromptClient(prompt)

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
    prompt = Prompt(name=prompt_name, version=1, prompt="Make me laugh")
    prompt_client = PromptClient(prompt)

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
    prompt = Prompt(name=prompt_name, version=1, prompt="Make me laugh")
    prompt_client = PromptClient(prompt)

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
    prompt = Prompt(name=prompt_name, version=1, prompt="Make me laugh")
    prompt_client = PromptClient(prompt)

    mock_server_call = langfuse.client.prompts.get
    mock_server_call.return_value = prompt

    result_call_1 = langfuse.get_prompt(prompt_name, version=1)
    assert mock_server_call.call_count == 1
    assert result_call_1 == prompt_client

    version_changed_prompt = Prompt(name=prompt_name, version=2, prompt="Make me laugh")
    version_changed_prompt_client = PromptClient(version_changed_prompt)
    mock_server_call.return_value = version_changed_prompt

    result_call_2 = langfuse.get_prompt(prompt_name, version=2)
    assert mock_server_call.call_count == 2
    assert result_call_2 == version_changed_prompt_client
