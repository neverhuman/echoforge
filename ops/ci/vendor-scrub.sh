#!/usr/bin/env bash
# Vendor-scrub lane: check for banned vendor terms.
set -euo pipefail
source "$(git rev-parse --show-toplevel)/ops/ci/lib.sh"
ci_header "Vendor scrub lane"
run_lane vendor-scrub
