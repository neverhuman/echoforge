#!/usr/bin/env bash
# Receipts lane: validate all agent receipts.
set -euo pipefail
source "$(git rev-parse --show-toplevel)/ops/ci/lib.sh"
ci_header "Receipts lane"
run_lane receipts
