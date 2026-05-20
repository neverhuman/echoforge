#!/usr/bin/env bash
set -euo pipefail

repo_root="$(CDPATH= cd -- "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
# shellcheck source=ops/ci/lib.sh
source "$repo_root/ops/ci/lib.sh"

run() {
  printf '==> %s\n' "$*" >&2
  "$@"
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || {
    printf 'missing required command: %s\n' "$1" >&2
    exit 1
  }
}

security_lane() {
  run cargo deny check
  run bash tools/sbom_emit.sh
}

lane="${1:-}"

if [[ -z "$lane" ]]; then
  echo "usage: $0 <lane>" >&2
  echo "known lanes: fast, security, vendor-scrub, banlist-audit, receipts, contracts, drift, demo, studio, web-smoke, web-e2e, science-smoke, gpu-smoke, dataset-smoke, doctor, score, licenses, validate-schemas, science-validate, sbom" >&2
  exit 64
fi

case "$lane" in
  fast)
    if command -v cargo-nextest >/dev/null 2>&1; then
      run cargo nextest run --workspace --locked --jobs "$(nproc 2>/dev/null || echo 4)"
    else
      run cargo test --workspace --locked
    fi
    if command -v jankurai >/dev/null 2>&1; then
      run jankurai adapters verify .
    else
      printf 'skip: jankurai not installed, adapters verify unavailable here\n' >&2
    fi
    ;;
  security)
    security_lane
    ;;
  vendor-scrub)
    run node tools/banlist_audit.mjs
    run node tools/vendor_scrub.mjs
    ;;
  banlist-audit)
    run node tools/banlist_audit.mjs
    ;;
  receipts)
    run node tools/receipt_guard.mjs --all
    ;;
  contracts)
    run cargo test -p echoforge-contracts-smoke --locked
    ;;
  drift)
    run cargo test -p echoforge-core --test schema_round_trip --locked
    run cargo test -p echoforge-core --test schemars_drift --locked
    run cargo test -p echoforge-contracts-smoke --locked
    ;;
  demo)
    require_node_at_least 26.1.0
    run npm run web:build
    HOST=127.0.0.1 PORT=8080 cargo run -p echoforge-studio --locked
    ;;
  studio)
    run cargo test -p echoforge-studio --locked
    ;;
  web-smoke)
    require_node_at_least 26.1.0
    run npm ci --no-fund --no-audit
    run npm run web:smoke
    ;;
  web-e2e)
    require_node_at_least 26.1.0
    run npm ci --no-fund --no-audit
    if [[ "${CI:-}" == "true" ]]; then
      run npx playwright install --with-deps chromium
    else
      run npx playwright install chromium
    fi
    cd apps/web
    run ../../node_modules/.bin/playwright test
    ;;
  ux-qa)
    require_node_at_least 26.1.0
    run npm ci --no-fund --no-audit
    cd apps/web
    run ../../node_modules/.bin/playwright test --reporter=html
    ;;
  science-smoke)
    run cargo test -p echoforge-sig --locked
    run cargo test -p echoforge-radar --locked
    run cargo test -p echoforge-dataset --locked
    ;;
  gpu-smoke)
    run cargo test -p echoforge-gpu-doctor --locked
    ;;
  dataset-smoke)
    run cargo test -p echoforge-dataset --locked
    ;;
  doctor)
    require_command jankurai
    mkdir -p target/jankurai
    run jankurai doctor . --fail-on critical --json target/jankurai/doctor.json --md target/jankurai/doctor.md
    ;;
  score)
    require_command jankurai
    require_node_at_least 26.1.0
    mkdir -p target/jankurai
    run node tools/boundary_evidence.mjs
    run jankurai audit . --full --mode standard --fail-under 85 --fail-on high --json target/jankurai/repo-score.json --md target/jankurai/repo-score.md
    ;;
  licenses)
    security_lane
    ;;
  validate-schemas)
    run cargo test -p echoforge-contracts-smoke --locked
    ;;
  science-validate)
    run cargo test -p echoforge-validate --locked
    run cargo test -p echoforge-tests-science --locked
    run cargo test -p echoforge-tests-science --locked --test micro_doppler_rotating_rod --test micro_doppler_propeller
    ;;
  sbom)
    run bash tools/sbom_emit.sh
    ;;
  *)
    echo "unknown lane: $lane" >&2
    echo "known lanes: fast, security, vendor-scrub, banlist-audit, receipts, contracts, drift, demo, studio, web-smoke, web-e2e, science-smoke, gpu-smoke, dataset-smoke, doctor, score, licenses, validate-schemas, science-validate, sbom" >&2
    exit 64
    ;;
esac
