from threading import RLock

from langfuse._utils.prompt_cache import PromptCache


def _cache_with(keys):
    cache = object.__new__(PromptCache)
    cache._lock = RLock()
    cache._cache = {key: object() for key in keys}
    return cache


def test_invalidate_does_not_remove_prompts_with_name_prefix():
    cache = _cache_with(
        ["foo-label:production", "foobar-label:production", "unrelated-version:2"]
    )

    cache.invalidate("foo")

    assert "foo-label:production" not in cache._cache
    assert "foobar-label:production" in cache._cache
    assert "unrelated-version:2" in cache._cache


def test_invalidate_does_not_remove_hyphenated_name_prefix():
    k = PromptCache.generate_cache_key
    keys = [
        k("my-prompt", version=None, label=None),
        k("my-prompt", version=3, label=None),
        k("my-prompt-extra", version=None, label=None),
    ]
    cache = _cache_with(keys)

    cache.invalidate("my-prompt")

    assert k("my-prompt", version=None, label=None) not in cache._cache
    assert k("my-prompt", version=3, label=None) not in cache._cache
    assert k("my-prompt-extra", version=None, label=None) in cache._cache
