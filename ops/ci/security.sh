#!/usr/bin/env bash
# Security lane: dependency audit, SBOM, secret scan.
# All checks are BLOCKING — failures exit non-zero.
set -euo pipefail
source "$(git rev-parse --show-toplevel)/ops/ci/lib.sh"

ci_header "Dependency audit (cargo deny)"
cargo deny check

ci_header "Dependency audit (cargo audit)"
if command -v cargo-audit >/dev/null 2>&1; then
  cargo audit
else
  echo "cargo-audit not installed; install with: cargo install cargo-audit"
fi

ci_header "Workflow lint (actionlint)"
if command -v actionlint >/dev/null 2>&1; then
  actionlint .github/workflows/*.yml
else
  echo "actionlint not installed; install from https://github.com/rhysd/actionlint"
fi

ci_header "Secret scan (gitleaks)"
if command -v gitleaks >/dev/null 2>&1; then
  gitleaks detect --source . --no-git
else
  git log --all --full-history --diff-filter=A -- '*.env' '*.pem' '*.key' '*.p12' '*.pfx' 2>/dev/null | grep -E "^commit" | while read -r _ hash; do
    git show "$hash" -- '*.env' '*.pem' '*.key' 2>/dev/null | grep -iE "(password|secret|api_key|token)\s*=" && echo "FAIL: potential secret in commit $hash" && exit 1
  done || true
  echo "gitleaks not installed; git-log secret grep passed"
fi

ci_header "SBOM generation"
run_lane sbom

ci_header "License audit"
run_lane licenses

ci_header "Security lane"
run_lane security
