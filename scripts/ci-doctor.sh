#!/usr/bin/env bash
# CI health check — verifies all required tools are present.
set -euo pipefail
# shellcheck source=ops/ci/lib.sh
source "$(git rev-parse --show-toplevel)/ops/ci/lib.sh"

ci_header "Running CI doctor"
require_command cargo
require_command node
require_node_at_least 26.1.0

if command -v jankurai >/dev/null 2>&1; then
  jankurai doctor .
else
  echo "warning: jankurai not installed; skipping doctor check"
fi
