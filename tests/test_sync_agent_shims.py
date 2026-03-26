from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "agents" / "sync-agent-shims.py"


def write_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def make_fixture_repo(tmp_path: Path) -> Path:
    repo_root = tmp_path / "repo"
    write_file(repo_root / ".agents" / "AGENTS.md", "# Fixture\n")
    write_file(repo_root / ".agents" / "skills" / "example" / "SKILL.md", "# Example\n")
    write_file(
        repo_root / ".agents" / "skills" / "README.md",
        "# Skills\n",
    )
    write_file(
        repo_root / ".agents" / "config.json",
        json.dumps(
            {
                "shared": {
                    "setupScript": "bash scripts/codex/setup.sh",
                    "devCommand": "poetry run bash",
                    "devTerminalDescription": "Interactive development shell",
                },
                "mcpServers": {
                    "docs": {
                        "transport": "http",
                        "url": "https://langfuse.com/api/mcp",
                    },
                    "stdio-example": {
                        "transport": "stdio",
                        "command": "npx",
                        "args": ["-y", "example"],
                        "env": {"EXAMPLE_TOKEN": "test"},
                    },
                },
                "claude": {
                    "settings": {
                        "permissions": {"allow": ["Bash(rg:*)"], "deny": []},
                        "enableAllProjectMcpServers": True,
                    }
                },
                "codex": {"environment": {"version": 1, "name": "fixture"}},
                "cursor": {"environment": {"agentCanUpdateSnapshot": False}},
            },
            indent=2,
        )
        + "\n",
    )
    write_file(repo_root / ".claude" / "skills" / "stale" / "SKILL.md", "# Stale\n")
    write_file(repo_root / ".vscode" / "settings.json", "{\n}\n")
    return repo_root


def run_script(repo_root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--repo-root", str(repo_root), *args],
        capture_output=True,
        text=True,
        check=False,
    )


def test_sync_agent_shims_generates_expected_outputs(tmp_path: Path) -> None:
    repo_root = make_fixture_repo(tmp_path)

    result = run_script(repo_root)

    assert result.returncode == 0
    assert json.loads((repo_root / ".mcp.json").read_text(encoding="utf-8")) == {
        "mcpServers": {
            "docs": {"type": "http", "url": "https://langfuse.com/api/mcp"},
            "stdio-example": {
                "command": "npx",
                "args": ["-y", "example"],
                "env": {"EXAMPLE_TOKEN": "test"},
            },
        }
    }
    assert json.loads(
        (repo_root / ".cursor" / "environment.json").read_text(encoding="utf-8")
    ) == {
        "agentCanUpdateSnapshot": False,
        "install": "bash scripts/codex/setup.sh",
        "terminals": [
            {
                "name": "Development Terminal",
                "command": "poetry run bash",
                "description": "Interactive development shell",
            }
        ],
    }
    assert (repo_root / "AGENTS.md").is_symlink()
    assert (repo_root / "AGENTS.md").resolve() == (
        repo_root / ".agents" / "AGENTS.md"
    ).resolve()
    assert (repo_root / "CLAUDE.md").is_symlink()
    assert (repo_root / "CLAUDE.md").resolve() == (repo_root / "AGENTS.md").resolve()
    assert (repo_root / ".claude" / "skills" / "example").is_symlink()
    assert not (repo_root / ".claude" / "skills" / "stale").exists()
    assert (repo_root / ".codex" / "config.toml").exists()
    assert (repo_root / ".codex" / "environments" / "environment.toml").exists()


def test_sync_agent_shims_check_mode_detects_drift(tmp_path: Path) -> None:
    repo_root = make_fixture_repo(tmp_path)
    run_script(repo_root)
    write_file(repo_root / ".cursor" / "environment.json", "{}\n")

    result = run_script(repo_root, "--check")

    assert result.returncode == 1
    assert "Out of sync" in result.stderr
