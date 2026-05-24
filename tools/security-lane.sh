#!/usr/bin/env bash
set -euo pipefail

echo "=== [security-lane] cargo deny check ==="
cargo deny check

echo "=== [security-lane] cargo audit ==="
cargo audit

echo "=== [security-lane] gitleaks secret scan ==="
if command -v gitleaks >/dev/null 2>&1; then
  gitleaks detect --source . --no-git --config .gitleaks.toml
else
  echo "gitleaks not installed; skipping"
fi

echo "=== [security-lane] cargo test (regression gate) ==="
rtk cargo test --workspace --locked
