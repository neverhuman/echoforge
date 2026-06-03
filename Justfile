set shell := ["sh", "-eu", "-c"]

default: fast

demo:
    rtk bash ops/run-lane.sh demo

bootstrap:
    rtk jankurai version
    rtk jankurai adapters verify .

fast:
    rtk cargo nextest run --workspace --locked --jobs $(nproc 2>/dev/null || echo 4) 2>/dev/null || rtk cargo test --workspace --locked
    rtk jankurai adapters verify .
    rtk just paper

# Accelerated test run using nextest (faster parallel execution; falls back to cargo test if nextest is absent)
fast-nx:
    rtk cargo nextest run --workspace --locked --jobs $(nproc 2>/dev/null || echo 4) 2>/dev/null || rtk cargo test --workspace --locked
    rtk jankurai adapters verify .

# Build timings report for CI performance tuning
timings:
    rtk cargo build --workspace --locked --timings

# Syntax-only check — fastest feedback loop before full build
check:
    rtk cargo check --workspace --locked

# Incremental check — minimal rebuild, no linker (sccache-friendly)
check-fast:
    CARGO_INCREMENTAL=1 rtk cargo check --workspace --locked

# Nextest with sccache-compatible incremental flags
fast-cached:
    CARGO_INCREMENTAL=1 rtk cargo nextest run --workspace --locked 2>/dev/null || rtk cargo test --workspace --locked

# sccache-accelerated build: set RUSTC_WRAPPER=sccache for compiler-cache speedup
fast-sccache:
    CARGO_INCREMENTAL=0 RUSTC_WRAPPER=sccache rtk cargo nextest run --workspace --locked --jobs $(nproc 2>/dev/null || echo 4) 2>/dev/null || rtk cargo test --workspace --locked

contracts:
    rtk cargo test -p echoforge-contracts-smoke --locked

drift:
    rtk cargo test -p echoforge-core --test schema_round_trip --locked
    rtk cargo test -p echoforge-core --test schemars_drift --locked
    rtk cargo test -p echoforge-contracts-smoke --locked

web-smoke:
    rtk bash ops/run-lane.sh web-smoke

web-e2e:
    rtk bash ops/run-lane.sh web-e2e

paper:
    rtk bash ops/run-lane.sh paper

studio:
    rtk bash ops/run-lane.sh studio

studio-sync:
    rtk bash ops/run-lane.sh studio-sync

science-smoke:
    rtk python3 -m unittest discover -s detection/real_data/tests -p 'test_*.py'
    rtk cargo test -p echoforge-sig --locked
    rtk cargo test -p echoforge-radar --locked
    rtk cargo test -p echoforge-dataset --locked

ml-pipelines-smoke:
    rtk cargo test -p echoforge-dataset --locked
    rtk cargo test -p echoforge-cli --locked
    rtk cargo test -p echoforge-studio --locked
    rtk python3 -m detection.pipeline_contracts.runner run-suite --suite evidence-ladder-v1 --repo-root . --data-root outputs/training-data/best-final-scenario-v1 --out-root outputs/ml-pipelines --workers-per-pipeline 20 --max-concurrent 3 --seed 20260520390001 --validation-tier evidence_ladder_v1 --smoke

gpu-smoke:
    rtk cargo test -p echoforge-gpu-doctor --locked

dataset-smoke:
    rtk cargo test -p echoforge-dataset --locked

doctor:
    mkdir -p target/jankurai
    rtk jankurai doctor . --fail-on critical --json target/jankurai/doctor.json --md target/jankurai/doctor.md

score:
    rtk bash ops/ci/score.sh

score-fast:
    mkdir -p target/jankurai
    jankurai audit . --changed-fast --changed-from origin/main --json target/jankurai/fast-score.json --md target/jankurai/fast-score.md

release-check:
    rtk just web-smoke
    rtk cargo test -p echoforge-studio --locked
    rtk just science-smoke
    rtk jankurai adapters verify .
    rtk just fast

# Targeted single-package test (no rtk wrapper — for direct narrow-scope proof runs)
test-pkg pkg:
    cargo nextest run -p {{pkg}} --locked

# sccache-accelerated cache lane: RUSTC_WRAPPER=sccache with per-package scope
just-cache pkg:
    CARGO_INCREMENTAL=0 RUSTC_WRAPPER=sccache cargo nextest run -p {{pkg}} --locked

validate-tier:
    rtk cargo run -p echoforge-cli -- validate tests/science/fixtures/bundles/v1_pass --target-tier v1 || true

science-validate:
    rtk cargo test -p echoforge-validate --locked
    rtk cargo test -p echoforge-tests-science --locked
    rtk cargo test -p echoforge-tests-science --locked --test micro_doppler_rotating_rod --test micro_doppler_propeller

vendor-scrub:
    rtk bash ops/run-lane.sh vendor-scrub

receipts:
    rtk bash ops/run-lane.sh receipts

sbom:
    rtk bash ops/run-lane.sh sbom

licenses:
    rtk bash ops/run-lane.sh licenses

validate-schemas:
    rtk cargo test -p echoforge-contracts-smoke --locked

gpu-receipt-doctor:
    @if [ -d .agents/receipts/gpu-xbabe2 ]; then ls -1t .agents/receipts/gpu-xbabe2 | head -1 | xargs -I {} node tools/receipt_guard.mjs .agents/receipts/gpu-xbabe2/{}; else echo "no gpu receipts yet"; fi

audit:
    cargo audit
    npm audit --audit-level=high
    actionlint .github/workflows/*.yml

security-lane:
    rtk bash ops/run-lane.sh security

security:
    rtk bash ops/run-lane.sh security
