# EchoForge Governance

EchoForge is radar-first, strict-open, and provenance-first.

## What the scaffold covers
- Root governance and bootstrap policy in `AGENTS.md`.
- Jankurai path ownership in `agent/owner-map.json`.
- Validation suite mapping in `agent/test-map.json`.
- Generated-zone policy in `agent/generated-zones.toml`.
- A one-command local bootstrap check via `rtk just fast`.

## What this repo should keep out of Git
- Large generated artifacts.
- Solver outputs.
- Array stores and benchmark payloads.
- Build outputs and local environments.

## Safe editing boundary
- Governance workers may edit only the bootstrap and policy surfaces listed in `AGENTS.md`.
- Runtime and contract directories stay reserved until an explicit implementation task opens them.
