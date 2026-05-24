#!/usr/bin/env bash
# Web e2e lane: Playwright end-to-end tests.
set -euo pipefail
source "$(git rev-parse --show-toplevel)/ops/ci/lib.sh"
run_lane web-e2e
