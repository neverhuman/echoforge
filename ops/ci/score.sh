#!/usr/bin/env bash
# Governance score lane: jankurai audit.
set -euo pipefail
source "$(git rev-parse --show-toplevel)/ops/ci/lib.sh"
run_lane score
