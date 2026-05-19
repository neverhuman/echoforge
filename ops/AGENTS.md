# Ops — Agent Guidance

Ownership: governance
Test lane: `bash ops/run-lane.sh fast`

## What This Directory Contains

- `run-lane.sh` — main lane dispatcher; all CI and local validation goes through this
- `pre-push.sh` — git pre-push hook runner
- `ci/lib.sh` — shared CI helper functions (sourced by CI scripts and workflows)
- `git-hooks/pre-push` — canonical git hook (symlink or install via `just install-hooks`)

## Lane System

Each lane in `run-lane.sh` maps to a proof lane defined in `agent/proof-lanes.toml`.
To add a new lane:
1. Add a `case` entry in `ops/run-lane.sh`
2. Add a `[[lane]]` entry in `agent/proof-lanes.toml`
3. Map affected paths in `agent/test-map.json`

## CI/Local Parity

Every CI job MUST delegate to `ops/run-lane.sh` — no inline commands in workflows.
Local dev should match CI exactly: `bash scripts/ci-local.sh` runs the same lanes as CI.

## Installing Git Hooks

```bash
ln -sf ../../ops/git-hooks/pre-push .git/hooks/pre-push
```
