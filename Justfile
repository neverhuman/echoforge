set shell := ["sh", "-eu", "-c"]

default: fast

demo:
    rtk npm run web:build
    HOST=127.0.0.1 PORT=8080 rtk cargo run -p echoforge-studio --locked

bootstrap:
    rtk jankurai version
    rtk jankurai adapters verify .

fast:
    rtk cargo test --workspace --locked
    rtk jankurai adapters verify .

contracts:
    rtk cargo test -p echoforge-contracts-smoke --locked

drift:
    rtk cargo test -p echoforge-core --test schema_round_trip --locked
    rtk cargo test -p echoforge-core --test schemars_drift --locked
    rtk cargo test -p echoforge-contracts-smoke --locked

web-smoke:
    rtk npm run web:smoke

science-smoke:
    rtk cargo test -p echoforge-sig --locked
    rtk cargo test -p echoforge-radar --locked
    rtk cargo test -p echoforge-dataset --locked

gpu-smoke:
    rtk cargo test -p echoforge-gpu-doctor --locked

dataset-smoke:
    rtk cargo test -p echoforge-dataset --locked

doctor:
    mkdir -p target/jankurai
    rtk jankurai doctor . --fail-on critical --json target/jankurai/doctor.json --md target/jankurai/doctor.md

score:
    mkdir -p target/jankurai
    rtk jankurai audit . --mode advisory --json target/jankurai/repo-score.json --md target/jankurai/repo-score.md

validate-tier:
    rtk cargo run -p echoforge-cli -- validate tests/science/fixtures/bundles/v1_pass --target-tier v1 || true

science-validate:
    rtk cargo test -p echoforge-validate --locked
    rtk cargo test -p echoforge-tests-science --locked
    rtk cargo test -p echoforge-tests-science --locked --test micro_doppler_rotating_rod --test micro_doppler_propeller

vendor-scrub:
    node tools/vendor_scrub.mjs

receipts:
    node tools/receipt_guard.mjs --all

sbom:
    bash tools/sbom_emit.sh

licenses:
    rtk cargo deny check
    bash tools/sbom_emit.sh

validate-schemas:
    rtk cargo test -p echoforge-contracts-smoke --locked

gpu-receipt-doctor:
    @if [ -d .agents/receipts/gpu-xbabe2 ]; then ls -1t .agents/receipts/gpu-xbabe2 | head -1 | xargs -I {} node tools/receipt_guard.mjs .agents/receipts/gpu-xbabe2/{}; else echo "no gpu receipts yet"; fi
