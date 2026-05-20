# EchoForge Agent Instructions

## Scope
EchoForge is a strict-open, radar-first, GPU-native synthetic sensing foundry. The product promise is public-proxy object signatures, uncertainty-scored radar artifacts, and reproducible validation evidence.

## Required rules
- Use `rtk` as the prefix for shell commands in this repo.
- Do not edit `schemas/`, `crates/`, `python/ai-service/`, or `docker/` unless a later task explicitly authorizes it.
- Do not overwrite, revert, or “clean up” changes made by other workers.
- Do not claim exact measured truth, proprietary-equivalent behavior, or classified fidelity for any object, platform, or sensor.
- Keep generated data and solver outputs out of Git.
- Treat hard negatives as robustness work, not evasion optimization.

## Ownership model
- `AGENTS.md`, `agent/**`, `.github/**`, `.gitignore`, `Justfile`, and other bootstrap policy files are governance-owned.
- The source of truth for ownership and generated zones is in:
  - `agent/owner-map.json`
  - `agent/test-map.json`
  - `agent/generated-zones.toml`
- If two tasks touch overlapping files, stop and ask before editing.

## Repo layout
- `agent/`: Jankurai/bootstrap metadata and worker maps.
- `.github/`: advisory workflow scaffolding and repo policy automation.
- `docs/`: human-readable governance and scaffold notes.
  - `docs/architecture.md` — layer map and stack decisions
  - `docs/testing.md` — proof lanes, cost budgets, kill-switches
  - `docs/audit-rubric.md` — jankurai rule interpretations and boundary/zone guidance
  - `docs/validation-tiers.md` — tier definitions and escalation paths
- `agent/generated-zones.toml` — generated artifact zone declarations
- `agent/boundaries.toml` — boundary slice definitions and exception roots
- `schemas/`, `crates/`, `python/ai-service/`, `docker/`: reserved runtime and contract work.

## Validation
- Fast local validation is `rtk just fast`.
- If governance files change, also run `rtk jankurai adapters verify .`.
- If you need a repo score or doctor output, use the `score` and `doctor` recipes in `Justfile`.

## Receipts
- Every worker leaves a receipt under `.agents/receipts/<slice>/<timestamp>.md`.
- Receipts should list files changed, commands run with `rtk`, validation results, generated artifacts, and any remaining risk.

## Handoff discipline
- Be explicit about what was changed and what was intentionally left alone.
- If a dependency, license, or bootstrap assumption blocks work, stop and report it instead of widening scope.

## Coordination
- Multi-agent work coordinates via `FUCKIT.md` (local-only, gitignored).
- Protocol spec at `docs/comms-protocol.md` (committed). Read before claiming a slot.
- Atomic slot claim by editing one row in the Work Slots table. Conflicts resolve by quote-and-respond in `## Messages`.

## Compliance
- Vendor banlist: `agent/banned-terms.toml`, enforced by `tools/vendor_scrub.mjs` and the `vendor-scrub` CI job (blocking).
- License allow-list: `deny.toml`, enforced by `cargo deny check`.
- Strict-open SBOMs: roll up to `sbom/index.md` via `tools/sbom_emit.sh`.
- Receipts under `.agents/receipts/<slice>/<UTC>.md` must satisfy `tools/receipt_guard.mjs` (filename + H2 sections + `rtk` command reference).
- Pre-commit hooks via `lefthook.yml`; install with `lefthook install`.
- Dual license MIT OR Apache-2.0; see `LICENSE`, `SECURITY.md`, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`.

## Stack discipline
- Rust + Vite + TypeScript + React only outside deep science.
- Python is reserved for deep-science cross-checks (Mie series scipy oracle, validators that need scipy/numpy). All product, control-plane, and tooling code lives in Rust or TypeScript.


<!-- jankurai merge marker: review and merge canonical guidance for AGENTS.md -->
