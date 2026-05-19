#!/usr/bin/env bash
# Local CI runner — mirrors the pre-merge GitHub Actions lanes.
set -euo pipefail
# shellcheck source=ops/ci/lib.sh
source "$(git rev-parse --show-toplevel)/ops/ci/lib.sh"
cd "$(git rev-parse --show-toplevel)"

ci_header "fast lane"
run_lane fast

ci_header "contracts lane"
run_lane contracts

ci_header "vendor-scrub lane"
bash ops/ci/vendor-scrub.sh

ci_header "receipts lane"
bash ops/ci/receipts.sh

ci_header "security lane"
bash ops/ci/security.sh

ci_header "web-smoke lane"
npm install --no-fund --no-audit
run_lane web-smoke

ci_header "All local CI lanes passed"
