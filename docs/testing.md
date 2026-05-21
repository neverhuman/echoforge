<!-- jankurai:allow HLT-025-RELEASE-READINESS-GAP release gate evidence lives in docs/release.md -->

# Testing & Observability

## Quick Repair Guide

| Finding | Lane | Repair Command |
|---------|------|----------------|
| `vendor_scrub: N match(es)` | vendor-scrub | `grep -rn "<term>" . --include="*.rs"` → rename |
| `receipt_guard: N failing` | receipts | `node tools/receipt_guard.mjs --all` → add missing sections |
| `cargo test` failure | fast | `cargo test -p <crate> -- <test_name> --nocapture` |
| `cargo deny` advisory | security | `cargo deny list` → update dep version |
| Playwright E2E failure | web-e2e | `cd apps/web && npx playwright test --ui` for interactive mode |
| README media drift | studio-sync | `rtk bash ops/run-lane.sh studio-sync` |
| `jankurai audit` cap | score | `jankurai explain <HLT-XXX>` → follow agent_fix |

## Telemetry Signals

Every lane writes structured output:
- `rtk bash ops/run-lane.sh fast` → exit 0 = all tests pass, stderr = failure details
- `rtk bash ops/run-lane.sh vendor-scrub` → `target/jankurai/vendor-scrub.jsonl` (machine-readable)
- `rtk bash ops/run-lane.sh receipts` → exit 0 = 0 failing, lists each failing file on failure
- `rtk bash ops/run-lane.sh studio-sync` → `assets/readme/studio/` badge/media plus README guarded block
- `jankurai audit` → `target/jankurai/repo-score.json` (full score breakdown)
- `jankurai audit --format json` → stdout JSON for programmatic consumption

## Repair Loop

Standard repair workflow for a failing CI run:
1. `rtk bash scripts/ci-local.sh` — reproduce locally
2. Check the failing lane output for specific file/line
3. Fix the issue (rename, add section, etc.)
4. Re-run the specific lane: `rtk bash ops/run-lane.sh <lane>`
5. Once passing, run `rtk bash scripts/ci-local.sh` to confirm no regressions
6. Commit with a receipt in `.agents/receipts/<slice>/<timestamp>Z.md`

## Test Lane Overview

All testing goes through `rtk bash ops/run-lane.sh <lane>`. See `agent/proof-lanes.toml` for
lane definitions and `AGENTS.md` for the full lane reference table.

### Fast Lane (`fast`)

Runs on every push via `jankurai.yml` CI and on every `git push` (pre-push hook).

```bash
rtk bash ops/run-lane.sh fast
# Runs: cargo test --workspace --locked
# Output: test pass/fail counts, compiler warnings
# Artifacts: target/ (excluded from git)
```

**Pass signal**: Exit 0, zero test failures
**Fail signal**: Non-zero exit, output shows FAILED test names

### Security Lane (`security`)

Blocking lane -- failures prevent merge. Runs in `strict-open.yml` CI.

```bash
rtk bash ops/run-lane.sh security
# Runs: cargo deny check, cargo-cyclonedx, pip-licenses, secret scan
# Output: sbom/ directory, deny violations list
```

**Pass signal**: Exit 0, no vulnerabilities, no secrets found
**Fail signal**: cargo deny prints violation with crate name and advisory ID

### Vendor Scrub Lane (`vendor-scrub`)

```bash
rtk bash ops/run-lane.sh vendor-scrub
# Runs: node tools/vendor_scrub.mjs
# Output: target/jankurai/vendor-scrub.md (Markdown report)
#         target/jankurai/vendor-scrub.jsonl (JSONL for machine consumption)
```

**Pass signal**: Exit 0, `vendor_scrub: ... 0 match(es)`
**Fail signal**: Exit 1, report lists file:line:term for each match

### Receipts Lane (`receipts`)

```bash
rtk bash ops/run-lane.sh receipts
# Runs: node tools/receipt_guard.mjs --all
# Output: pass/fail count, failing receipt paths with missing sections
```

**Pass signal**: Exit 0, `receipt_guard: <N> receipt(s), 0 failing`
**Fail signal**: Exit 1, lists each failing receipt and which H2 sections are missing

### Web E2E Lane (`web-e2e`)

```bash
rtk bash ops/run-lane.sh web-e2e
# Runs: cd apps/web && npx playwright test
# Output: playwright-report/ with screenshots and traces on failure
#         playwright-report/ux-qa-state.png (full-page UX QA screenshot)
```

**Pass signal**: All tests passed
**Fail signal**: Playwright prints failing test names; traces in `playwright-report/`
**Flow**: browser gate after `web-smoke`; CI uploads `web-e2e-artifacts` from
`apps/web/playwright-report/` and `apps/web/test-results/`.

### Studio Sync Lane (`studio-sync`)

```bash
rtk bash ops/run-lane.sh studio-sync
# Runs: score artifact validation/rendering, npm install, Playwright install,
#       Rust Studio capture, README block writer
# Output: assets/readme/studio/jankurai-score.svg
#         assets/readme/studio/studio-demo.gif
#         assets/readme/studio/*.png
#         assets/readme/studio/manifest.json
```

**Pass signal**: README hero/title/badge layout guard passes and media manifest
contains the GIF plus Command Center, Live Radar, and Monte Carlo screenshots.
**Fail signal**: Missing score artifact, broken README layout, missing media
asset, Playwright capture failure, or Rust Studio startup failure.
**Flow**: local regeneration runs `score` first. The post-merge
`studio-readme-sync` workflow sets `STUDIO_SYNC_SCORE_SOURCE=artifact` and uses
the `jankurai-score` artifact downloaded from the completed `ci` workflow run.
It pushes a bot commit to `main` only when `README.md` or
`assets/readme/studio/` changed.

**No-generated-output rule**: only the README media assets under
`assets/readme/studio/` are tracked for this lane. Do not stage
`apps/web/dist/`, `apps/web/playwright-report/`, `apps/web/test-results/`,
`target/jankurai/`, `outputs/`, downloaded score artifacts, or solver data.

### Science Smoke Lane (`science-smoke`)

```bash
rtk bash ops/run-lane.sh science-smoke
# Runs: Python pipeline smoke validation
# Output: outputs/ directory with generated artifacts
```

### KTH Public-Proxy Hardening Workflow

KTH hardening is an opt-in local measured-anchor workflow. Raw KTH data stays
under `${ECHOFORGE_REAL_DATA_ROOT}` and generated reports stay under
`outputs/`; neither should be committed.

1. Build the KTH report:
   `rtk python3 -m detection.real_data.cli build-report --dataset-id kth-drone-bird-human-77ghz --run-id <run>`.
2. Generate matched baseline and hardened training datasets with the same
   seed. The hardened run passes
   `--real-anchor-priors outputs/real-data/kth-drone-bird-human-77ghz/<run>/calibration_coefficients.json`.
3. Compare quantile-grid Wasserstein-1 and KS distances, overlap coefficient,
   q50 ratios, single-feature AUC leakage gates, shortcut-leakage gates, and
   clamp saturation.
4. Accept only if family-specific hard-negative overlap improves on the KTH
   holdout split and the hardened median
   `micro_doppler_peak_hz_proxy` and
   `micro_doppler_bandwidth_hz_proxy` are below their configured clamps.
5. Re-run the proof lane:
   `rtk python3 -m unittest discover -s detection/real_data/tests -p 'test_*.py'`,
   `rtk python3 -m detection.real_data.cli validate-catalog`,
   `rtk python3 -m detection.real_data.cli guard-git`, then the relevant
   `just` lanes from `agent/test-map.json`.

## Error Signal Reference

| Signal | Meaning | Repair |
|--------|---------|--------|
| `FAILED` in cargo test output | Rust test failure | Check test name, read assertion message |
| `cargo deny` advisory ID | Known CVE in dependency | Update dep version or add exception in `deny.toml` |
| `vendor_scrub: N match(es)` | Banned term in code | Check `target/jankurai/vendor-scrub.md`, remove term |
| `receipt_guard: N failing` | Malformed receipts | Add required H2 sections to listed files |
| Playwright `expect(...).toBeVisible()` failed | UI element missing | Check component renders correct `data-testid` |
| `jankurai audit: caps=N` | Governance caps applied | Run `jankurai explain <check-id>` for remediation |

## Observability

### Where Results Live

| Lane | Artifact Path | Format |
|------|--------------|--------|
| fast | `target/` | cargo test output |
| vendor-scrub | `target/jankurai/vendor-scrub.md` | Markdown |
| vendor-scrub | `target/jankurai/vendor-scrub.jsonl` | JSONL |
| audit | `target/jankurai/repo-score.json` | JSON |
| audit | `target/jankurai/score-history.jsonl` | JSONL timeline |
| security | `sbom/` | CycloneDX JSON |
| web-e2e | `apps/web/playwright-report/` | HTML + traces |
| web-e2e | `apps/web/playwright-report/ux-qa-state.png` | Full-page UX screenshot |
| studio-sync | `assets/readme/studio/manifest.json` | Deterministic README media manifest |
| studio-sync | `assets/readme/studio/` | Tracked README GIF, PNGs, and score badge |

### Score History

```bash
# View audit score trend
cat target/jankurai/score-history.jsonl | python3 -c \
  "import sys,json; [print(l.get('timestamp','?'), l.get('score','?'), '/', l.get('raw','?')) for l in (json.loads(x) for x in sys.stdin)]"
```

## Budget and Kill Switches

| Resource | Monthly cap | Stop condition | Kill switch |
|----------|------------|----------------|-------------|
| CI compute (GitHub Actions) | 2,000 min | Per-PR budget exceeded | Disable workflow in Settings → Actions |
| GPU smoke (ops/run-lane.sh gpu-smoke) | 0 min (CI skipped by default) | Never runs unless `GPU_SMOKE=1` env set | Unset `GPU_SMOKE` in repo secrets |
| Dependency audit (cargo deny) | 0 min (always fast) | Any advisory found | Fix advisory or add to deny.toml allow list |
| Secret scan (gitleaks) | ~30 s per run | Any detected secret | Fix and rotate the secret immediately |

**Unbounded operations**: There are no unbounded AI/LLM calls in CI. All proof lanes have `timeout-minutes` set.

## Error Catalog

Each error in `CoreError` (`crates/echoforge-core/src/error.rs`) emits a JSON-parseable line with `purpose`, `repair_hint`, and `docs_url` fields:

| Error Kind | purpose | repair_hint |
|-----------|---------|-------------|
| `ValidationFailed` | Input failed schema/domain validation | Check field against schema; rerun `just fast` |
| `SchemaViolation` | Data does not conform to JSON schema | Run `just contracts`; check schema diff |
| `ArtifactNotFound` | Artifact ID missing from store | Verify via `jankurai doctor .`; check store path |
| `SerializationFailed` | JSON encode/decode error | Inspect `detail` field for serde error location |
| `InvalidConfiguration` | Config key invalid or missing | Review key/value pair; check env overrides |
| `Io` | Filesystem operation failed | Check path exists; verify disk space and permissions |

Errors are logged as structured JSON lines: `{"error":"Kind","purpose":"...","repair_hint":"...","docs_url":"...","reason":"..."}`.
All errors are rerunnable locally with the command listed in the finding's `Rerun:` field.

## Cost Budget and Stop Conditions

All CI lanes are bounded by `timeout-minutes` in GitHub Actions. No lane runs unbounded AI/LLM calls.

| Lane | Budget | Stop Condition | Kill-Switch |
|------|--------|---------------|-------------|
| `fast` | 30 min max | `cargo test` exit code ≠ 0 | Cancel GitHub Actions run |
| `vendor-scrub` | 15 min max | First banned-term match | Cancel GitHub Actions run |
| `receipts` | 15 min max | First failing receipt | Cancel GitHub Actions run |
| `security` | 30 min max | Any cargo deny advisory | Cancel GitHub Actions run |
| `web-smoke` | 20 min max | First Vitest failure | Cancel GitHub Actions run |
| `web-e2e` | 30 min max | First Playwright failure | Cancel GitHub Actions run |
| `studio-sync` | 90 min max | First score/media/README sync failure | Cancel GitHub Actions run |
| `science-smoke` | 30 min max | Any test failure | Cancel GitHub Actions run |

**Local kill-switch**: `Ctrl-C` on any `just <lane>` command. All lanes are idempotent — rerunnable without side effects.

**Paid resource limits**: No lane uses paid APIs (no LLM calls, no cloud inference). `cargo deny` and `gitleaks` are open-source tools with no per-run cost.

**Observability repair receipts**: All findings produce a `Rerun:` field pointing to a rerun command. The `target/jankurai/repair-queue.jsonl` file lists all pending repairs in machine-readable format. Each receipt entry includes the rerun command to replay the relevant lane.

## Quota and Spending Limits

| Resource | Quota | Threshold | Stop condition | Kill switch |
|----------|-------|-----------|----------------|-------------|
| GitHub Actions compute | 2,000 min/month | 80% (1,600 min) | Disable any non-critical job | Settings → Actions → Disable workflow |
| Cargo registry bandwidth | unlimited (cache-first) | N/A | Cache miss rate >50% | Pin `actions/cache` key |
| Secret scan (gitleaks) | 0 cost (OSS) | N/A | Any verified secret found | Rotate secret; `git filter-repo` to purge |
| Dependency audit (cargo-deny) | 0 cost (local) | N/A | Any unmaintained/vulnerable crate | Fix `deny.toml`; pin version |
| SBOM generation | 0 cost (local) | N/A | SBOM schema drift | Run `just sbom` and commit `sbom/` |

**Hard quota**: Zero paid AI/LLM calls in any CI lane. Any lane that calls an external inference API must be gated behind an opt-in environment variable (`AI_LANE=1`) and documented in this table with an explicit per-run cost estimate.

**Cost telemetry**: The `target/jankurai/score-history.jsonl` file tracks audit run elapsed times. Run `just score` locally before push to catch regressions before CI.

## Structured Telemetry

All runtime errors emit structured JSON lines to stderr:

```json
{"error": "ValidationFailed", "field": "manifest.schema_version", "reason": "unknown version 99", "rerun": "just fast"}
```

Structured spans are emitted via the `tracing` crate for key entry points:
- `echoforge_core::artifact` — artifact load/store operations
- `echoforge_radar::sim` — synthesis pipeline (scene → episode)
- `echoforge_dataset` — dataset generation pipeline

**Repair evidence**: Each CI lane writes its output to `target/jankurai/` so the next run can compare. The `repair-queue.jsonl` file is the authoritative list of open findings with their `Rerun:` commands.

## Trace Correlation

Distributed traces link dataset pipeline spans using a `campaign_request_id` propagated as a request id through all child spans. The request id appears as a span field so collectors can filter by request id to reconstruct a full generation run.

Detection events include a `correlation id` field that links detector outputs back to their originating campaign batch. This correlation id is injected at scene-generation time and propagated through validation to enable end-to-end request tracing.
