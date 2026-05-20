# Contracts — Agent Guidance

Ownership: schema_contract
Test lane: `bash ops/run-lane.sh contracts`

## What This Directory Contains

`schema_catalog.json` is the canonical 12-schema catalog defining all JSON schema contracts
used by EchoForge crates and the Python ai-service. Schemas are versioned and must remain
backward-compatible across minor versions.

## Rules for Agents

- Never modify `schema_catalog.json` directly — run `bash ops/run-lane.sh contracts` to regenerate
- Schema changes require updating the contract version in `Cargo.toml` workspace
- All schema fields must have descriptions; no anonymous properties
- Breaking changes (removing/renaming fields) require a major version bump
- Run `bash ops/run-lane.sh validate-schemas` after any schema change

## Adding a New Schema

1. Add the schema definition to `schemas/<name>.schema.json`
2. Register it in `schema_catalog.json` with a unique `$id`
3. Run `bash ops/run-lane.sh contracts` to validate
4. Add a Rust contract test in `tests/contracts/`

<!-- jankurai merge marker: review and merge canonical guidance for contracts/AGENTS.md -->
