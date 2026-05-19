#!/usr/bin/env bash
# Bootstrap lane: advisory scaffold check + fast lane.
set -euo pipefail
source "$(git rev-parse --show-toplevel)/ops/ci/lib.sh"

REPO_ROOT="$(git rev-parse --show-toplevel)"

ci_header "Advisory scaffold check"
for path in \
  AGENTS.md \
  .gitignore \
  Justfile \
  Cargo.toml \
  pyproject.toml \
  package.json \
  agent/owner-map.json \
  agent/test-map.json \
  agent/generated-zones.toml
do
  test -f "${REPO_ROOT}/${path}" || { echo "missing required file: ${path}" >&2; exit 1; }
done

ci_header "Fast lane"
run_lane fast
