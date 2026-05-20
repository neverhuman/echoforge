#!/usr/bin/env bash
# Shared CI helper functions — sourced by CI scripts and GitHub Actions workflows.
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
# shellcheck source=ops/ci/node-toolchain.sh
source "${REPO_ROOT}/ops/ci/node-toolchain.sh"

require_command() {
  command -v "$1" >/dev/null 2>&1 || { echo "error: required command not found: $1" >&2; exit 1; }
}

require_node_at_least() {
  local required_version="${1:?usage: require_node_at_least <semver>}"
  echoforge_require_node_toolchain
  require_command node

  local current_version
  current_version="$(node --version | sed 's/^v//')"

  local lowest_version
  lowest_version="$(printf '%s\n%s\n' "$required_version" "$current_version" | sort -V | head -n1)"
  if [[ "$lowest_version" != "$required_version" ]]; then
    printf 'error: Node %s+ required (got v%s)\n' "$required_version" "$current_version" >&2
    exit 1
  fi
}

run_lane() {
  bash "${REPO_ROOT}/ops/run-lane.sh" "$@"
}

ci_header() {
  echo "==> [ci] $*"
}
