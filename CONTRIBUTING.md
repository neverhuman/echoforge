# Contributing to EchoForge

Thank you for the interest. EchoForge is a strict-open, radar-first
synthetic-sensing foundry. Read `AGENTS.md` and `docs/comms-protocol.md`
before opening a PR.

## Ground rules

1. Use `rtk` as the prefix for shell commands in this repo (see `RTK.md`).
2. Do not edit `schemas/`, `crates/`, `python/ai-service/`, or `docker/`
   unless your task or the approved plan file authorizes it.
3. Do not overwrite, revert, or "clean up" changes made by other workers
   without coordinating in the local-only scratchpad and quoting in
   `## Messages`.
4. Do not claim exact measured truth, proprietary-equivalent behavior, or
   classified fidelity for any object, platform, or sensor.
5. Keep generated data and solver outputs out of Git (see
   `agent/generated-zones.toml` and `.gitignore`).
6. Treat hard negatives as robustness work, not evasion optimization.

## Receipts

Every slice of work leaves a receipt under
`.agents/receipts/<slice>/<UTC-timestamp>.md`. The receipt must include
the H2 sections checked by `tools/receipt_guard.mjs`:

```
# <slice> Receipt
## Files Changed
- path: short purpose
## Commands Run
- rtk <cmd>: result
## Validation Results
- <lane>: pass|fail|skip
## Notes
- risks, follow-ups, open items
```

Validate before commit:

```
rtk node tools/receipt_guard.mjs .agents/receipts/<slice>/<ts>.md
```

## Installing hooks

```
brew install lefthook   # or: npm i -D lefthook
lefthook install
```

The pre-commit hooks run rustfmt, clippy (`-D warnings`), ruff,
biome, JSON-schema validation, vendor-scrub, and receipt-presence.
The pre-push hook runs `rtk just fast`.

## Coordinating with other agents

Multi-agent work coordinates through a local-only scratchpad per
`docs/comms-protocol.md`. Claim a slot atomically by editing one
table row, document handoffs explicitly, and never delete another
agent's content.

## Running the lanes

```
rtk just fast            # workspace tests + Jankurai adapters verify
rtk just contracts       # schemas + Rust core + Python contract tests
rtk just science-smoke   # sig + radar + dataset + science-validate tests
rtk just dataset-smoke   # dataset crate
rtk just gpu-smoke       # GPU doctor placeholder (real GPU is via xbabe2 receipt)
rtk just vendor-scrub    # banlist check
rtk just receipts        # receipt-guard --all
rtk just sbom            # emit sbom/*.cdx.json + sbom/index.md
rtk just licenses        # cargo-deny + pip-licenses + npm sbom roll-up
```

## Activating currently-disabled CI suites

Currently `gpu-smoke`, `dataset-smoke`, and `science-smoke` are advisory.
Activation criteria for each is documented in `agent/test-map.json`
under the suite's `notes` field; flip `enabled` to `true` only after
the criteria are met.

## License

By contributing, you agree your contributions are dual-licensed under
MIT OR Apache-2.0 (see `LICENSE`).
