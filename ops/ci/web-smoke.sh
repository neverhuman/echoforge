#!/usr/bin/env bash
# Web smoke lane: npm install + web smoke tests.
set -euo pipefail
source "$(git rev-parse --show-toplevel)/ops/ci/lib.sh"
run_lane web-smoke
