#!/usr/bin/env bash
# Fast lane: unit tests + adapter verification.
set -euo pipefail
source "$(git rev-parse --show-toplevel)/ops/ci/lib.sh"

git_crypt_preflight() {
  if [[ "${CI:-}" != "true" ]]; then
    return 0
  fi

  require_command git-crypt
  require_command base64

  local protected_paths=(
    detection/crypt_ip_impl/advanced_main_run_detectors.py
    detection/crypt_ip_impl/ei_evolution_trace.py
  )

  git-crypt status -e "${protected_paths[@]}"

  if [[ -z "${ECHOFORGE_GIT_CRYPT_KEY_B64:-}" ]]; then
    printf 'error: missing ECHOFORGE_GIT_CRYPT_KEY_B64 secret for protected EI runtime unlock\n' >&2
    exit 1
  fi

  local key_file
  key_file="$(mktemp)"
  trap 'rm -f "$key_file"' RETURN
  printf '%s' "$ECHOFORGE_GIT_CRYPT_KEY_B64" | base64 --decode >"$key_file"
  git-crypt unlock "$key_file"
}

git_crypt_preflight
run_lane fast
