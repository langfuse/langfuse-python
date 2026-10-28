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


def _write_lock(root, skills, lock_version=1):
    (root / "langfuse-skills.lock").write_text(
        json.dumps({"lockVersion": lock_version, "skills": skills}), encoding="utf-8"
    )


def test_discovery_follows_symlinks_and_only_includes_immediate_skills(
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

    assert discover_skills(dir=str(root)) == [
        {"name": "directory-link"},
        {"name": "file-link"},
        {"name": "plain"},
    ]
    assert discover_skills(dir=str(tmp_path / "missing")) == []


@pytest.mark.parametrize("lock_version", [1, 2])
def test_configured_lock_precedes_cwd_and_resolves_paths_from_lock(
    tmp_path, monkeypatch, lock_version
):
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "skills"
    _install(root, "customer")
    _write_lock(tmp_path, [{"path": "skills/customer", "name": "cwd", "version": 1}])
    _write_lock(
        root,
        [{"path": "customer", "name": "configured", "version": 2}],
        lock_version=lock_version,
    )

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
    ["not json", '{"lockVersion":1,"skills":{}}', '{"lockVersion":true,"skills":[]}'],
)
def test_malformed_lock_falls_back_to_directory_names(
    tmp_path, monkeypatch, content, caplog
):
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "skills"
    _install(root, "customer")
    _write_lock(tmp_path, [{"path": "skills/customer", "name": "cwd", "version": 1}])
    (root / "langfuse-skills.lock").write_text(content, encoding="utf-8")

    assert discover_skills(dir=str(root)) == [{"name": "customer"}]
    assert "Could not read skill lockfile" in caplog.text


def test_discovery_follows_lock_symlink(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "skills"
    _install(root, "customer")
    _write_lock(tmp_path, [{"path": "customer", "name": "locked", "version": 2}])
    (root / "langfuse-skills.lock").symlink_to(tmp_path / "langfuse-skills.lock")

    assert discover_skills(dir=str(root)) == [{"name": "locked", "version": 2}]


@pytest.mark.parametrize("kind", ["directory", "symlink-loop"])
def test_unreadable_lock_falls_back_to_directory_names(
    tmp_path, monkeypatch, kind, caplog
):
    monkeypatch.chdir(tmp_path)
    _install(tmp_path, "customer")
    lock = tmp_path / "langfuse-skills.lock"
    if kind == "directory":
        lock.mkdir()
    else:
        lock.symlink_to(lock)

    assert discover_skills(dir=str(tmp_path)) == [{"name": "customer"}]
    assert "Could not read skill lockfile" in caplog.text


@pytest.mark.parametrize("broken_path", ["directory", "SKILL.md"])
def test_discovery_skips_entries_with_filesystem_errors(
    tmp_path, monkeypatch, broken_path, caplog
):
    monkeypatch.chdir(tmp_path)
    _install(tmp_path, "valid")
    broken = tmp_path / "broken"
    if broken_path == "SKILL.md":
        broken.mkdir()
        broken = broken / "SKILL.md"
    broken.symlink_to(broken)
    (tmp_path / "missing-target").symlink_to(tmp_path / "missing")

    assert discover_skills(dir=str(tmp_path)) == [{"name": "valid"}]
    assert "Skipping skill directory" in caplog.text
    assert "missing-target" not in caplog.text


def test_serialization_preserves_extra_fields():
    skills = [{"name": "customer", "version": 2, "futureField": {"nested": True}}]
    assert json.loads(serialize_available_skills(skills)) == skills
    assert serialize_available_skills([]) == "[]"


def test_invalid_serialization_does_not_break_tracing(caplog):
    cycle = []
    cycle.append(cycle)
    assert serialize_available_skills(cycle) is None
    assert "Failed to serialize available skills" in caplog.text
