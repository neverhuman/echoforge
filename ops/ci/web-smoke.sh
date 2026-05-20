#!/usr/bin/env bash
# Web smoke lane: npm install + web smoke tests.
# Requires Node >=20 (vitest 4 / vite 8 use node:util styleText, added in Node 20).
# Locally: nvm use 20  (see .nvmrc at repo root)
set -euo pipefail
repo_root="$(git rev-parse --show-toplevel)"
source "$repo_root/ops/ci/lib.sh"

node_major="$(node --version | sed 's/v//' | cut -d. -f1)"
if [[ "$node_major" -lt 20 ]]; then
  printf 'ERROR: Node >=20 required (got %s). Run: nvm use 20\n' "$(node --version)" >&2
  exit 1
fi

run_lane web-smoke
