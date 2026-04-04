from time import sleep
from unittest.mock import Mock, patch

import pytest

from langfuse._client.client import Langfuse
from langfuse._utils.prompt_cache import (
    DEFAULT_PROMPT_CACHE_TTL_SECONDS,
    PromptCache,
    PromptCacheItem,
)
from langfuse.api import NotFoundError, Prompt_Chat, Prompt_Text
from langfuse.model import ChatPromptClient, TextPromptClient


@pytest.mark.parametrize(
    ("variables", "placeholders", "expected_len", "expected_contents"),
    [
        (
            {"role": "helpful", "task": "coding"},
            {},
            3,
            ["You are a helpful assistant", None, "Help me with coding"],
        ),
        (
            {},
            {},
            3,
            ["You are a {{role}} assistant", None, "Help me with {{task}}"],
        ),
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
        (
            {"role": "helpful", "task": "coding"},
            {"unused": [{"role": "user", "content": "Won't appear"}]},
            3,
            ["You are a helpful assistant", None, "Help me with coding"],
        ),
        (
            {"role": "helpful", "task": "coding"},
            {"examples": "not a list"},
            3,
            [
                "You are a helpful assistant",
                "not a list",
                "Help me with coding",
            ],
        ),
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
                "['invalid message', {'role': 'user', 'content': 'valid message'}]",
                "valid message",
                "Help me with coding",
            ],
        ),
    ],
)
def test_compile_with_placeholders(
    variables, placeholders, expected_len, expected_contents
) -> None:
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
            assert "type" in result[i] and result[i]["type"] == "placeholder"
        elif isinstance(result[i], str):
            assert result[i] == expected_content
        else:
            assert "content" in result[i]
            assert result[i]["content"] == expected_content


@pytest.fixture
def langfuse():
    from langfuse._client.resource_manager import LangfuseResourceManager

    langfuse_instance = Langfuse()
    langfuse_instance.api = Mock()

    if langfuse_instance._resources is None:
        langfuse_instance._resources = Mock(spec=LangfuseResourceManager)
        langfuse_instance._resources.prompt_cache = PromptCache()

    return langfuse_instance


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


def test_throw_if_name_unspecified(langfuse):
    with pytest.raises(ValueError) as exc_info:
        langfuse.get_prompt("")

    assert "Prompt name cannot be empty" in str(exc_info.value)


def test_throw_when_failing_fetch_and_no_cache(langfuse):
    mock_server_call = langfuse.api.prompts.get
    mock_server_call.side_effect = Exception("Prompt not found")

    with pytest.raises(Exception) as exc_info:
        langfuse.get_prompt("failing_fetch_and_no_cache")

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


def test_throw_if_cache_ttl_seconds_positional_argument(langfuse):
    with pytest.raises(TypeError) as exc_info:
        langfuse.get_prompt("test ttl seconds in positional arg", 20)

    assert "positional arguments" in str(exc_info.value)


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

    mock_time.return_value = ttl_seconds - 1

    result_call_2 = langfuse.get_prompt(prompt_name)
    assert mock_server_call.call_count == 1
    assert result_call_2 == prompt_client

    mock_time.return_value = ttl_seconds + 1

    result_call_3 = langfuse.get_prompt(prompt_name)

    while True:
        if langfuse._resources.prompt_cache._task_manager.active_tasks() == 0:
            break
        sleep(0.1)

    assert mock_server_call.call_count == 2
    assert result_call_3 == prompt_client


@patch.object(PromptCacheItem, "get_epoch_seconds")
def test_disable_caching_when_ttl_zero(mock_time, langfuse: Langfuse):
    mock_time.return_value = 0
    prompt_name = "test_disable_caching_when_ttl_zero"

    prompt1 = Prompt_Text(
        name=prompt_name,
        version=1,
        prompt="Make me laugh",
        labels=[],
        type="text",
        config={},
        tags=[],
    )
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

    result1 = langfuse.get_prompt(prompt_name, cache_ttl_seconds=0)
    assert mock_server_call.call_count == 1
    assert result1 == TextPromptClient(prompt1)

    result2 = langfuse.get_prompt(prompt_name, cache_ttl_seconds=0)
    assert mock_server_call.call_count == 2
    assert result2 == TextPromptClient(prompt2)

    result3 = langfuse.get_prompt(prompt_name, cache_ttl_seconds=0)
    assert mock_server_call.call_count == 3
    assert result3 == TextPromptClient(prompt3)

    assert result1 != result2 != result3


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

    mock_time.return_value = DEFAULT_PROMPT_CACHE_TTL_SECONDS + 1

    stale_result = langfuse.get_prompt(prompt_name)
    assert stale_result == prompt_client

    langfuse.get_prompt(prompt_name)
    langfuse.get_prompt(prompt_name)
    langfuse.get_prompt(prompt_name)
    langfuse.get_prompt(prompt_name)

    while True:
        if langfuse._resources.prompt_cache._task_manager.active_tasks() == 0:
            break
        sleep(0.1)

    assert mock_server_call.call_count == 2

    updated_result = langfuse.get_prompt(prompt_name)
    assert updated_result.version == 2
    assert updated_result == TextPromptClient(updated_prompt)


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

    mock_time.return_value = DEFAULT_PROMPT_CACHE_TTL_SECONDS - 1

    result_call_2 = langfuse.get_prompt(prompt_name)
    assert mock_server_call.call_count == 1
    assert result_call_2 == prompt_client

    mock_time.return_value = DEFAULT_PROMPT_CACHE_TTL_SECONDS + 1

    result_call_3 = langfuse.get_prompt(prompt_name)
    while True:
        if langfuse._resources.prompt_cache._task_manager.active_tasks() == 0:
            break
        sleep(0.1)

    assert mock_server_call.call_count == 2
    assert result_call_3 == prompt_client


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

    mock_time.return_value = DEFAULT_PROMPT_CACHE_TTL_SECONDS + 1
    mock_server_call.side_effect = Exception("Server error")

    result_call_2 = langfuse.get_prompt(prompt_name, max_retries=1)
    while True:
        if langfuse._resources.prompt_cache._task_manager.active_tasks() == 0:
            break
        sleep(0.1)

    assert mock_server_call.call_count == 3
    assert result_call_2 == prompt_client


@patch.object(PromptCacheItem, "get_epoch_seconds")
def test_evict_prompt_cache_entry_when_refresh_returns_not_found(
    mock_time, langfuse: Langfuse
) -> None:
    mock_time.return_value = 0

    prompt_name = "test_evict_prompt_cache_entry_when_refresh_returns_not_found"
    ttl_seconds = 5
    fallback_prompt = "fallback text prompt"

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
    cache_key = PromptCache.generate_cache_key(prompt_name, version=None, label=None)

    mock_server_call = langfuse.api.prompts.get
    mock_server_call.return_value = prompt

    initial_result = langfuse.get_prompt(
        prompt_name,
        cache_ttl_seconds=ttl_seconds,
        max_retries=0,
    )
    assert initial_result == prompt_client
    assert langfuse._resources.prompt_cache.get(cache_key) is not None

    mock_time.return_value = ttl_seconds + 1

    def raise_not_found(*_args: object, **_kwargs: object) -> None:
        raise NotFoundError({"message": "Prompt not found"})

    mock_server_call.side_effect = raise_not_found

    stale_result = langfuse.get_prompt(
        prompt_name,
        cache_ttl_seconds=ttl_seconds,
        max_retries=0,
    )
    assert stale_result == prompt_client

    while True:
        if langfuse._resources.prompt_cache._task_manager.active_tasks() == 0:
            break
        sleep(0.1)

    assert langfuse._resources.prompt_cache.get(cache_key) is None

    fallback_result = langfuse.get_prompt(
        prompt_name,
        cache_ttl_seconds=ttl_seconds,
        fallback=fallback_prompt,
        max_retries=0,
    )
    assert fallback_result.is_fallback
    assert fallback_result.prompt == fallback_prompt


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
