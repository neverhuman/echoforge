#!/usr/bin/env bash
# Install EchoForge git hooks.
# Run once after cloning, and again whenever `lefthook install` is called
# (lefthook install overwrites .git/hooks/pre-push with its generic wrapper).
#
# Usage: bash ops/setup-git-hooks.sh
set -euo pipefail

repo_root="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
cd "$repo_root"

printf '==> lefthook install\n'
if command -v lefthook >/dev/null 2>&1; then
  lefthook install
elif npx lefthook --version >/dev/null 2>&1; then
  npx lefthook install
else
  printf 'warning: lefthook not found; skipping lefthook install\n' >&2
fi

printf '==> installing branch-protection pre-push hook\n'
cp tools/git-hooks/pre-push .git/hooks/pre-push
chmod +x .git/hooks/pre-push
printf '    .git/hooks/pre-push installed\n'

printf '==> done\n'
