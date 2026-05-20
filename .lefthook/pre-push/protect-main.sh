#!/usr/bin/env bash
# Blocks any direct push to the main branch.
# Pre-push stdin format per ref: local_ref local_sha remote_ref remote_sha
set -euo pipefail

while IFS= read -r line; do
  remote_ref="$(printf '%s' "$line" | awk '{print $3}')"
  if [[ "$remote_ref" == "refs/heads/main" ]]; then
    printf '\n[protect-main] BLOCKED: direct push to main is not allowed.\n' >&2
    printf 'Open a pull request instead:  gh pr create --fill\n\n' >&2
    exit 1
  fi
done
