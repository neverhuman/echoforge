#!/usr/bin/env bash
# Install the same Jankurai binary for local and GitHub score lanes.
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
JANKURAI_REV="6f1aa45fca09ebb523f79b38ad465da28a86dfb1"
JANKURAI_VERSION="jankurai 1.5.1"
MARKER="${REPO_ROOT}/target/jankurai/jankurai-rev"

mkdir -p "${REPO_ROOT}/target/jankurai"

if command -v jankurai >/dev/null 2>&1 &&
  [[ "$(jankurai --version)" == "${JANKURAI_VERSION}" ]] &&
  [[ -f "${MARKER}" ]] &&
  [[ "$(cat "${MARKER}")" == "${JANKURAI_REV}" ]]; then
  jankurai --version
  exit 0
fi

cargo install \
  --git https://github.com/neverhuman/jankurai.git \
  --rev "${JANKURAI_REV}" \
  --locked \
  --force \
  jankurai

jankurai --version
printf '%s\n' "${JANKURAI_REV}" >"${MARKER}"
