# Shared Skills

Shared repo skills for any coding agent working in Langfuse Python.

Use these from `AGENTS.md`. Claude Code reaches the same shared instructions via
the root `CLAUDE.md` compatibility symlink. Shared skills should stay focused on
reusable implementation guidance rather than runtime automation.

For the shared agent config and generated shim model, start with
[`../README.md`](../README.md).

Shared skills should use progressive disclosure:

- keep `SKILL.md` short
- link to focused `references/` docs instead of copying long guidance into one
  file
- add helper scripts only when they materially reduce repeated work

There are no repo-level shared skills yet. Add one when a workflow is repeated
often enough that it should be standardized across tools.

## Adding a New Shared Skill

1. Create a folder under `.agents/skills/<skill-name>/`.
2. Add a short `SKILL.md` entrypoint.
3. Add `references/` docs or helper scripts only when they are needed.
4. Keep the skill tightly scoped to one domain or workflow.
5. Link the skill from `.agents/AGENTS.md` if it is broadly relevant.
6. Run `python3 scripts/agents/sync-agent-shims.py`.
7. Run `python3 scripts/agents/sync-agent-shims.py --check`.
8. Run `poetry run pytest tests/test_sync_agent_shims.py`.
