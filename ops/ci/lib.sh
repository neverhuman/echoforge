#!/usr/bin/env bash
# Shared CI helper functions — sourced by CI scripts and GitHub Actions workflows.
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"

require_command() {
  command -v "$1" >/dev/null 2>&1 || { echo "error: required command not found: $1" >&2; exit 1; }
}

run_lane() {
  bash "${REPO_ROOT}/ops/run-lane.sh" "$@"
}

ci_header() {
  echo "==> [ci] $*"
}
