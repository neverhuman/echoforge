# Release Management

## Version Source

The **single source of truth** for the EchoForge version is the `[workspace.package]` section
in `Cargo.toml` at the repository root. All workspace crates inherit this version — never set
a crate-level version independently.

```toml
# Cargo.toml (root)
[workspace.package]
version = "0.X.Y"
```

When cutting a release, bump `version` here. CI propagates the value automatically.

## Versioning Scheme

EchoForge uses [Semantic Versioning](https://semver.org/) (`MAJOR.MINOR.PATCH`).

- **MAJOR**: Breaking schema changes, incompatible API removals
- **MINOR**: New capabilities, additive schema changes, new detection algorithms
- **PATCH**: Bug fixes, performance improvements, documentation updates

## Changelog

All user-visible changes must be recorded in `CHANGELOG.md` at the repository root before
a release is tagged. Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

```bash
# Verify CHANGELOG.md has an entry for the current version before tagging
grep "## \[$(cargo metadata --no-deps --format-version 1 | python3 -c \
  "import sys,json; print(json.load(sys.stdin)['packages'][0]['version'])\")]" CHANGELOG.md
```

## CI Evidence

Release gate checks run in two GitHub Actions workflow files:

| Workflow file | Purpose |
|---|---|
| `.github/workflows/ci.yml` | Fast, contracts, `web-smoke`, `web-e2e`, and score artifact publication |
| `.github/workflows/strict-open.yml` | Security, vendor-scrub, receipts |
| `.github/workflows/studio-readme-sync.yml` | Post-merge README score badge and Studio media refresh |

A release is only cut from a commit where both workflows show green on `main`.
`web-smoke` is the quick web gate; `web-e2e` is the browser gate and depends on
`web-smoke` so the cheaper build/unit pass fails first.

After a successful `ci` run on `main`, `studio-readme-sync` checks out the merged
commit, downloads the `jankurai-score` artifact from that completed run, renders
`assets/readme/studio/jankurai-score.svg`, regenerates the Playwright Studio
media in `assets/readme/studio/`, rewrites only the guarded Studio block at the
top of `README.md`, and pushes `chore: sync studio README media` back to `main`
only when the README/media diff is non-empty.

`ci` uploads `web-e2e-artifacts` from `apps/web/playwright-report/` and
`apps/web/test-results/`; the score job uploads `jankurai-score` from
`target/jankurai/**`. README/assets-only bot commits are ignored by the `ci`
path filters to avoid a sync loop.

## Release Checklist

Before cutting a release:

- [ ] `rtk just web-smoke`
- [ ] `rtk cargo test -p echoforge-studio --locked`
- [ ] `rtk just science-smoke`
- [ ] `rtk jankurai adapters verify .`
- [ ] `rtk bash ops/run-lane.sh fast` — all Rust tests green
- [ ] `rtk bash ops/run-lane.sh contracts` — schema contracts validated
- [ ] `rtk bash ops/run-lane.sh vendor-scrub` — 0 banned term matches
- [ ] `rtk bash ops/run-lane.sh receipts` — 0 failing receipts
- [ ] `rtk bash ops/run-lane.sh security` — cargo deny clean, SBOM generated
- [ ] `rtk bash ops/run-lane.sh web-e2e` — Playwright E2E passes with UX QA screenshot/artifacts
- [ ] `rtk bash ops/run-lane.sh studio-sync` — README badge/media block regenerates without solver output noise
- [ ] `CHANGELOG.md` updated with all changes since last release
- [ ] Staged jankurai hook passes on the release commit before every push
- [ ] Git tag created: `git tag -a v<VERSION> -m "Release v<VERSION>"`

## Automated Release Steps (`just`)

The `Justfile` provides composable recipes. Key release targets:

```bash
just test          # full test suite (equivalent to fast + contracts lanes)
just lint          # clippy + fmt check
just release-check # runs all blocking lanes in sequence
```

Run `just --list` to see all available recipes.

Commit workflow for this release line should stay small and vertical: once the staged jankurai hook passes on a coherent slice, commit it instead of batching unrelated work.

## Deployment Steps

### Docker Images

```bash
# Build CPU worker
docker build -f docker/cpu-worker.Dockerfile -t echoforge-cpu:v<VERSION> .

# Build GPU worker
docker build -f docker/gpu-worker.Dockerfile -t echoforge-gpu:v<VERSION> .

# Push to registry
docker push echoforge-cpu:v<VERSION>
docker push echoforge-gpu:v<VERSION>
```

### Binary Artifacts

```bash
# Build optimized binaries (--locked ensures Cargo.lock is respected)
cargo build --release --workspace --locked

# Binaries are at:
# target/release/echoforge-cli
# target/release/echoforge-validate
```

## Build Reproducibility and Provenance

EchoForge aims for reproducible builds:

- `Cargo.lock` is committed and `--locked` is required in release builds
- All CI steps pin GitHub Actions to full commit SHAs (not floating tags)
- SBOM is generated at release time via `cargo-cyclonedx` (CycloneDX JSON format)

```bash
# Generate SBOM
rtk bash ops/run-lane.sh sbom
# Outputs to sbom/ directory (CycloneDX JSON per crate + workspace rollup)
```

Provenance verification: compare the `sbom/` artifacts from the release tag against the
registry image layers using `syft` or `grype` for supply-chain audits.

## Rollback Procedure

1. Identify the last known-good release tag:
   ```bash
   git tag --sort=-creatordate | head -5
   ```
2. Check out the tag:
   ```bash
   git checkout v<LAST_GOOD>
   ```
3. Rebuild and redeploy Docker images from that tag.
4. For schema rollbacks: schema changes are required to be backward-compatible, so old
   clients continue to work against the restored schema.
5. Notify downstream consumers via the release notes in `CHANGELOG.md`.

## Security Launch Gate

A release is blocked unless all of the following produce green evidence in CI:

| Gate | Command | Evidence artifact |
|---|---|---|
| Dependency audit | `cargo deny check` | CI log — zero violations |
| Secret scan | `gitleaks detect --source . --no-git` | CI log — zero secrets found |
| SBOM generation | `bash ops/run-lane.sh security` | `sbom/*.cdx.json` committed |
| License check | `bash ops/run-lane.sh licenses` | CI log — allowlist only |
| Vendor scrub | `bash ops/run-lane.sh vendor-scrub` | CI log — 0 banned term matches |
| Governance audit | `jankurai audit` | score ≥ 85, caps = 0, findings = 0 |

No launch proceeds if any of the above fails.

## Monitoring

After deployment, the following signals are watched:

- **Studio service health**: `GET /api/health` → HTTP 200 with `{"status":"ok"}` within 2 s
- **Mesh endpoint latency**: `GET /api/mesh/list` P99 < 500 ms (logged by studio service)
- **Error rate**: non-2xx responses from `/api/mesh/*` logged to stderr (structured JSON); alert on sustained error rate > 1%
- **Binary STL size**: any response > 50 MB to `/api/mesh/:id` triggers a warning log
- **Structured error events**: all `CoreError` variants are logged with `purpose`, `repair_hint`, and `docs_url` fields for downstream alerting

## Abuse Controls

EchoForge is a strict-open simulation platform with no user accounts. Abuse surface is limited:

- **Request rate**: studio service enforces a configurable per-IP request cap (default: 60 req/min)
- **Payload size**: binary STL responses are capped at the physics budget for canonical primitives (< 100 MB)
- **Banned-term enforcement**: `agent/banned-terms.toml` prevents overclaiming language from reaching published artifacts; enforced via `ops/run-lane.sh vendor-scrub` in every CI run
- **Supply-chain**: all GitHub Actions are pinned to full commit SHAs; dependency tree is audited on every push via `cargo deny check`
- **No external data ingestion**: the studio service does not accept user-supplied geometry or scenario files

## Backup and Data Durability

EchoForge does not maintain a persistent runtime database. Durable state lives exclusively in:

- **Git history** — all schemas, configs, and contracts are in the repository
- **SBOM artifacts** — regenerated at each release; committed to `sbom/`
- **Docker images** — pushed to registry with immutable version tags

Recovery: check out any release tag and rebuild images. No database migration or data restore is required.

## Version History

See `CHANGELOG.md` for the full version history and change notes.

## Release Artifacts

Each release produces:
- `echoforge-cli` binary (Linux x86_64)
- `echoforge-validate` binary (Linux x86_64)
- `echoforge-cpu:v<VERSION>` Docker image
- `echoforge-gpu:v<VERSION>` Docker image
- `sbom/` directory with CycloneDX SBOM for all components
- `agent/repo-score.json` audit snapshot
