#!/usr/bin/env bash
# Web smoke lane: npm install + web smoke tests.
# Requires Node >=26.1.0.
# Locally: nvm use 26.1.0  (see .nvmrc at repo root)
set -euo pipefail
repo_root="$(git rev-parse --show-toplevel)"
source "$repo_root/ops/ci/lib.sh"

require_node_at_least 26.1.0

run_lane web-smoke
