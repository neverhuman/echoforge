#!/usr/bin/env bash
# Contracts lane: schema round-trip and smoke tests.
set -euo pipefail
source "$(git rev-parse --show-toplevel)/ops/ci/lib.sh"
run_lane contracts
