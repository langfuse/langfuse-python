# Shared Agent Setup

This directory is the neutral, repo-owned source of truth for agent behavior in
Langfuse Python.

Use `.agents/` for configuration and guidance that should apply across tools.
Do not put durable shared guidance only in `.claude/`, `.codex/`, `.cursor/`,
or `.vscode/`.

## Layout

- `AGENTS.md`: canonical shared root instructions
- `config.json`: shared bootstrap and MCP configuration used to generate
  tool-specific shims
- `skills/`: shared, tool-neutral implementation guidance for recurring
  workflows

## `config.json`

`.agents/config.json` contains five kinds of data:

- `shared`: defaults used across tools
- `mcpServers`: project MCP servers and how to connect to them
- `claude`: Claude-specific generated settings inputs
- `codex`: Codex-specific generated settings inputs
- `cursor`: Cursor-specific generated settings inputs

Current shape:

```json
{
  "shared": {
    "setupScript": "bash scripts/codex/setup.sh",
    "devCommand": "poetry run bash",
    "devTerminalDescription": "Interactive development shell inside the Poetry environment"
  },
  "mcpServers": {
    "langfuse-docs": {
      "transport": "http",
      "url": "https://langfuse.com/api/mcp"
    }
  },
  "claude": {
    "settings": {}
  },
  "codex": {
    "environment": {
      "version": 1,
      "name": "langfuse-python"
    }
  },
  "cursor": {
    "environment": {
      "agentCanUpdateSnapshot": false
    }
  }
}
```

## How Shims Are Generated

`scripts/agents/sync-agent-shims.py` reads `.agents/config.json` and writes the
tool discovery files that those products require.

Generated local artifacts:

- `.claude/settings.json`
- `.claude/skills/*`
- `.cursor/environment.json`
- `.cursor/mcp.json`
- `.vscode/mcp.json`
- `.mcp.json`
- `.codex/config.toml`
- `.codex/environments/environment.toml`

The repo root discovery files remain committed as symlinks:

- `AGENTS.md` -> `.agents/AGENTS.md`
- `CLAUDE.md` -> `AGENTS.md`

This keeps provider discovery stable while `.agents/` remains the source of
truth.

## When To Edit `config.json`

Edit `.agents/config.json` when you need to:

- add, remove, or update a shared MCP server
- change the shared setup/bootstrap command
- change the default terminal command or terminal label used by generated shims
- adjust generated Claude, Cursor, or Codex settings that are intentionally
  modeled in the shared config

Do not edit generated shim files by hand. Edit the canonical files in
`.agents/` instead.

## Workflow

After editing `.agents/config.json` or `.agents/skills/**`:

1. Run `python3 scripts/agents/sync-agent-shims.py`
2. Run `python3 scripts/agents/sync-agent-shims.py --check`
3. Run `poetry run pytest tests/test_sync_agent_shims.py`
4. Verify you did not stage generated files under `.claude/skills/` or any of
   the generated MCP/runtime config paths
5. Update `.agents/AGENTS.md` or `CONTRIBUTING.md` if the shared workflow
   materially changed

`bash scripts/postinstall.sh` runs the sync/check flow as a convenience helper,
and `bash scripts/codex/setup.sh` uses it during agent environment bootstrap.

## Adding Shared Skills

Shared skills live under `.agents/skills/`.

Use them for durable, reusable guidance such as:

- repeated SDK maintenance workflows
- review checklists that should apply across tools
- repo-specific runbooks that should not live only in a provider-specific folder

Do not use skills for one-off notes or tool runtime configuration.

`python3 scripts/agents/sync-agent-shims.py` projects shared skills into
`.claude/skills/` so Claude can discover the same repo-owned skills.

For the skill authoring workflow, see [skills/README.md](skills/README.md).
