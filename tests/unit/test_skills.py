import json

import pytest

from langfuse._client.skills import (
    discover_skills,
    serialize_available_skills,
)


def _install(root, name):
    directory = root / name
    directory.mkdir(parents=True)
    (directory / "SKILL.md").write_text("# Skill", encoding="utf-8")
    return directory


def _write_lock(root, skills):
    (root / "langfuse-skills.lock").write_text(
        json.dumps({"lockVersion": 1, "skills": skills}), encoding="utf-8"
    )


def test_discovery_only_includes_immediate_regular_skill_directories(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "skills"
    _install(root, "plain")
    _install(root, ".hidden")
    _install(root, "nested/deeper")
    (root / "directory-link").symlink_to(root / "plain", target_is_directory=True)
    (root / "file-link").mkdir()
    (root / "file-link" / "SKILL.md").symlink_to(root / "plain" / "SKILL.md")
    (root / "directory-marker" / "SKILL.md").mkdir(parents=True)

    assert discover_skills(dir=str(root)) == [{"name": "plain"}]
    assert discover_skills(dir=str(tmp_path / "missing")) == []


def test_configured_lock_precedes_cwd_and_resolves_paths_from_lock(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "skills"
    _install(root, "customer")
    _write_lock(tmp_path, [{"path": "skills/customer", "name": "cwd", "version": 1}])
    _write_lock(root, [{"path": "customer", "name": "configured", "version": 2}])

    assert discover_skills(dir="skills") == [{"name": "configured", "version": 2}]
    (root / "langfuse-skills.lock").unlink()
    assert discover_skills(dir="skills") == [{"name": "cwd", "version": 1}]


def test_invalid_entries_duplicates_and_long_names_preserve_other_skills(
    tmp_path, monkeypatch, caplog
):
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "skills"
    for name in ("first", "second", "invalid-version", "x" * 129):
        _install(root, name)
    _write_lock(
        root,
        [
            {"path": "first", "name": "same", "version": 2_147_483_647},
            {"path": "second", "name": "same", "version": 2},
            {"path": "invalid-version", "name": "bad", "version": True},
            {"path": "first", "name": "bad", "version": 2_147_483_648},
            {"path": "first", "name": "bad", "version": 0},
            {"path": "first", "name": "bad", "version": 1.5},
        ],
    )

    assert discover_skills(dir=str(root)) == [
        {"name": "same", "version": 2_147_483_647},
        {"name": "invalid-version"},
    ]
    assert "Skipping invalid skill entry" in caplog.text
    assert "Skipping duplicate skill name same" in caplog.text
    assert "name exceeds 128 characters" in caplog.text


@pytest.mark.parametrize(
    "content",
    ["not json", '{"lockVersion":2,"skills":[]}', '{"lockVersion":true,"skills":[]}'],
)
def test_malformed_lock_fails_explicit_discovery(tmp_path, monkeypatch, content):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "langfuse-skills.lock").write_text(content, encoding="utf-8")
    with pytest.raises(ValueError):
        discover_skills(dir=str(tmp_path))


def test_serialization_preserves_extra_fields():
    skills = [{"name": "customer", "version": 2, "futureField": {"nested": True}}]
    assert json.loads(serialize_available_skills(skills)) == skills
    assert serialize_available_skills([]) == "[]"


def test_invalid_serialization_does_not_break_tracing(caplog):
    cycle = []
    cycle.append(cycle)
    assert serialize_available_skills(cycle) is None
    assert "Failed to serialize available skills" in caplog.text
