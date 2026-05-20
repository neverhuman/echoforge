#!/usr/bin/env bash
# Fast lane: unit tests + adapter verification.
set -euo pipefail
source "$(git rev-parse --show-toplevel)/ops/ci/lib.sh"
run_lane fast
