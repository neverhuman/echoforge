#!/usr/bin/env bash
# Governance score lane: deterministic boundary evidence plus blocking jankurai audit.
set -euo pipefail
source "$(git rev-parse --show-toplevel)/ops/ci/lib.sh"
require_node_at_least 26.1.0
run_lane score
