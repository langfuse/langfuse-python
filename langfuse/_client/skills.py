"""Local skill discovery and availability configuration."""

import json
import os
import stat
from pathlib import Path
from typing import Any, Optional

from langfuse.logger import langfuse_logger
from langfuse.types import SkillReference

_CLI_LOCK = "langfuse-skills.lock"


def discover_skills(*, dir: str) -> list[SkillReference]:
    """Read immediate skill directories and optional CLI installation metadata.

    Directories must contain a regular SKILL.md file. Hidden directories are
    skipped and symlinks are followed. The configured directory's
    langfuse-skills.lock takes precedence over the working directory's lock.
    Unlocked skills use their directory name without a version. Lock records
    do not verify local edits.

    Missing directories return an empty list; unreadable ones warn and return
    an empty list. Unreadable or malformed lockfiles warn and fall back to
    directory names. Entries that cannot be inspected, invalid lock entries,
    and duplicate names are skipped with a warning.
    """
    root = Path(os.path.abspath(dir))
    try:
        if _stat_if_exists(root) is None:
            return []
        with os.scandir(root) as entries:
            candidates = sorted(entries, key=lambda entry: entry.name)
    except OSError as error:
        langfuse_logger.warning(
            "Could not read skills directory %s. No skills discovered: %s",
            root,
            error,
        )
        return []

    locked: dict[Path, SkillReference] = {}
    try:
        locked = _read_cli_lock(_find_cli_lock(root))
    except (OSError, ValueError) as error:
        langfuse_logger.warning(
            "Could not read skill lockfile for %s. Using directory names without versions: %s",
            root,
            error,
        )
    available: dict[str, SkillReference] = {}
    for candidate in candidates:
        if candidate.name.startswith("."):
            continue
        directory = Path(candidate.path)
        try:
            directory_stat = _stat_if_exists(directory)
            if directory_stat is None or not stat.S_ISDIR(directory_stat.st_mode):
                continue
            skill_file = _stat_if_exists(directory / "SKILL.md")
            if skill_file is None or not stat.S_ISREG(skill_file.st_mode):
                continue
        except OSError as error:
            langfuse_logger.warning(
                "Skipping skill directory %s: could not inspect files: %s",
                directory,
                error,
            )
            continue
        skill = locked.get(directory, {"name": candidate.name})
        name = skill["name"]
        if len(name) > 128:
            langfuse_logger.warning(
                "Skipping skill %s: name exceeds 128 characters.", name
            )
            continue
        if name in available:
            langfuse_logger.warning("Skipping duplicate skill name %s.", name)
            continue
        available[name] = skill
    return list(available.values())


def _stat_if_exists(path: Path) -> Optional[os.stat_result]:
    try:
        return path.stat()
    except FileNotFoundError:
        return None


def _find_cli_lock(directory: Path) -> Optional[Path]:
    for path in (directory / _CLI_LOCK, Path.cwd() / _CLI_LOCK):
        if _stat_if_exists(path) is not None:
            return path
    return None


def _read_cli_lock(file: Optional[Path]) -> dict[Path, SkillReference]:
    if file is None:
        return {}
    file_stat = _stat_if_exists(file)
    if file_stat is None:
        return {}
    if not stat.S_ISREG(file_stat.st_mode):
        message = f"Invalid skill lockfile {file}."
        raise ValueError(message)
    value = json.loads(file.read_text(encoding="utf-8"))
    if (
        not isinstance(value, dict)
        or type(value.get("lockVersion")) is not int
        or not isinstance(value.get("skills"), list)
    ):
        message = f"Unsupported or invalid skill lockfile {file}."
        raise ValueError(message)
    locked: dict[Path, SkillReference] = {}
    for skill in value["skills"]:
        if not _is_valid_locked_skill(skill):
            langfuse_logger.warning("Skipping invalid skill entry in %s.", file)
            continue
        path = Path(os.path.abspath(file.parent / skill["path"]))
        locked[path] = {"name": skill["name"], "version": skill["version"]}
    return locked


def _is_valid_locked_skill(skill: Any) -> bool:
    return (
        isinstance(skill, dict)
        and isinstance(skill.get("path"), str)
        and bool(skill["path"])
        and isinstance(skill.get("name"), str)
        and 0 < len(skill["name"]) <= 128
        and type(skill.get("version")) is int
        and 1 <= skill["version"] <= 2_147_483_647
    )


def serialize_available_skills(value: Any) -> Optional[str]:
    """Serialize skill attributes without projecting away unknown fields."""
    if value is None:
        return None
    try:
        return json.dumps(value, separators=(",", ":"), allow_nan=False)
    except Exception:
        langfuse_logger.warning("Failed to serialize available skills. Dropping value.")
        return None
