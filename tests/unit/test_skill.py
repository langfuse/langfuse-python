from types import SimpleNamespace
from unittest.mock import patch

from langfuse._utils.skill_cache import (
    DEFAULT_SKILL_CACHE_TTL_SECONDS,
    SkillCache,
    SkillCacheItem,
)
from langfuse.model import SkillClient


def _make_skill(
    *,
    name="my-skill",
    version=1,
    description="A test skill",
    instructions="Help the user with {{task}}.",
    metadata=None,
    allowed_tools=None,
    labels=None,
    tags=None,
    commit_message=None,
):
    """Build a lightweight stand-in for the generated ``Skill`` model.

    ``SkillClient`` only reads attributes off the passed object, so a
    ``SimpleNamespace`` avoids depending on the auto-generated ``Skill``
    constructor for these unit tests.
    """
    return SimpleNamespace(
        name=name,
        version=version,
        description=description,
        instructions=instructions,
        metadata=metadata,
        allowed_tools=allowed_tools if allowed_tools is not None else [],
        labels=labels if labels is not None else [],
        tags=tags if tags is not None else [],
        commit_message=commit_message,
    )


# ---------------------------------------------------------------------------
# SkillClient.compile
# ---------------------------------------------------------------------------


def test_compile_substitutes_variables():
    client = SkillClient(_make_skill(instructions="Help the user with {{task}}."))

    assert client.compile(task="coding") == "Help the user with coding."


def test_compile_multiple_variables():
    client = SkillClient(
        _make_skill(instructions="{{greeting}}, complete the {{task}}.")
    )

    assert client.compile(greeting="Hi", task="report") == "Hi, complete the report."


def test_compile_missing_variable_left_untouched():
    client = SkillClient(_make_skill(instructions="Help with {{task}}."))

    assert client.compile() == "Help with {{task}}."


def test_compile_none_variable_becomes_empty_string():
    client = SkillClient(_make_skill(instructions="Help with {{task}}."))

    assert client.compile(task=None) == "Help with ."


def test_variables_property():
    client = SkillClient(_make_skill(instructions="{{greeting}} {{task}} {{greeting}}"))

    assert client.variables == ["greeting", "task", "greeting"]


def test_client_exposes_skill_fields():
    client = SkillClient(
        _make_skill(
            name="planner",
            version=3,
            description="Plans things",
            metadata={"team": "core"},
            allowed_tools=["search", "write"],
            labels=["production"],
            tags=["planning"],
            commit_message="init",
        )
    )

    assert client.name == "planner"
    assert client.version == 3
    assert client.description == "Plans things"
    assert client.metadata == {"team": "core"}
    assert client.allowed_tools == ["search", "write"]
    assert client.labels == ["production"]
    assert client.tags == ["planning"]
    assert client.commit_message == "init"
    assert client.is_fallback is False


def test_client_dict_serialization():
    client = SkillClient(_make_skill(name="s", version=2))
    data = client.dict()

    assert data["name"] == "s"
    assert data["version"] == 2
    assert data["instructions"] == "Help the user with {{task}}."
    assert set(data.keys()) == {
        "name",
        "version",
        "description",
        "instructions",
        "metadata",
        "allowed_tools",
        "labels",
        "tags",
        "commit_message",
    }


def test_client_equality():
    a = SkillClient(_make_skill())
    b = SkillClient(_make_skill())
    c = SkillClient(_make_skill(instructions="different"))

    assert a == b
    assert a != c
    assert a != "not-a-skill"


def test_none_list_fields_default_to_empty():
    client = SkillClient(_make_skill(allowed_tools=None, labels=None, tags=None))

    assert client.allowed_tools == []
    assert client.labels == []
    assert client.tags == []


# ---------------------------------------------------------------------------
# SkillCache TTL / expiry
# ---------------------------------------------------------------------------


@patch.object(SkillCacheItem, "get_epoch_seconds")
def test_cache_item_expiry(mock_time):
    mock_time.return_value = 0
    item = SkillCacheItem(SkillClient(_make_skill()), ttl_seconds=60)

    assert item.is_expired() is False

    mock_time.return_value = 59
    assert item.is_expired() is False

    mock_time.return_value = 61
    assert item.is_expired() is True


@patch.object(SkillCacheItem, "get_epoch_seconds")
def test_cache_set_get_and_default_ttl(mock_time):
    mock_time.return_value = 0
    cache = SkillCache()
    value = SkillClient(_make_skill())

    cache.set("my-skill", value, ttl_seconds=None)
    cached = cache.get("my-skill")

    assert cached is not None
    assert cached.value is value
    assert cached.is_expired() is False

    mock_time.return_value = DEFAULT_SKILL_CACHE_TTL_SECONDS + 1
    assert cache.get("my-skill").is_expired() is True


def test_cache_delete_and_clear():
    cache = SkillCache()
    cache.set("a", SkillClient(_make_skill(name="a")), ttl_seconds=60)
    cache.set("b", SkillClient(_make_skill(name="b")), ttl_seconds=60)

    cache.delete("a")
    assert cache.get("a") is None
    assert cache.get("b") is not None

    cache.clear()
    assert cache.get("b") is None


def test_cache_invalidate_by_name_prefix():
    cache = SkillCache()
    cache.set("my-skill-version:1", SkillClient(_make_skill()), ttl_seconds=60)
    cache.set("my-skill-label:production", SkillClient(_make_skill()), ttl_seconds=60)
    cache.set("other-skill", SkillClient(_make_skill(name="other")), ttl_seconds=60)

    cache.invalidate("my-skill")

    assert cache.get("my-skill-version:1") is None
    assert cache.get("my-skill-label:production") is None
    assert cache.get("other-skill") is not None


def test_generate_cache_key():
    assert SkillCache.generate_cache_key("s", version=2, label=None) == "s-version:2"
    assert (
        SkillCache.generate_cache_key("s", version=None, label="staging")
        == "s-label:staging"
    )
    # Defaults to the production label when neither version nor label given
    assert (
        SkillCache.generate_cache_key("s", version=None, label=None)
        == "s-label:production"
    )
