#!/usr/bin/env bash
# Emit an SBOM index for EchoForge. Strict-open posture: every dependency
# graph is enumerated by its native ecosystem tool, results are normalized
# to CycloneDX JSON under `sbom/`, and a human-readable `sbom/index.md`
# summary is committed.
#
# Tools (install in CI as needed):
#   cargo install cargo-cyclonedx
#   pip install pip-licenses
#   npm  (built-in `npm sbom --sbom-format=cyclonedx` since npm 10)

set -euo pipefail

OUT="sbom"
mkdir -p "${OUT}"

echo "== Rust cargo SBOM ==" >&2
if command -v cargo-cyclonedx >/dev/null 2>&1; then
  rtk cargo cyclonedx --format json --target-cyclonedx-version 1.5 --output-path "${OUT}/cargo.cdx.json" >/dev/null
else
  echo "warn: cargo-cyclonedx not installed, skipping cargo SBOM" >&2
fi

echo "== npm SBOM ==" >&2
if command -v npm >/dev/null 2>&1; then
  npm sbom --sbom-format=cyclonedx > "${OUT}/npm.cdx.json" 2>/dev/null || {
    echo "warn: npm sbom failed, skipping" >&2
  }
fi

echo "== Python license inventory ==" >&2
if command -v pip-licenses >/dev/null 2>&1; then
  pip-licenses --format=json > "${OUT}/python.licenses.json" || true
else
  echo "warn: pip-licenses not installed, skipping python licenses" >&2
fi

echo "== Roll-up ==" >&2
{
  echo "# EchoForge SBOM Index"
  echo
  echo "Generated: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo
  echo "## Artifacts"
  for f in cargo.cdx.json npm.cdx.json python.licenses.json; do
    if [ -f "${OUT}/${f}" ]; then
      bytes=$(wc -c <"${OUT}/${f}")
      echo "- \`sbom/${f}\` (${bytes} bytes)"
    else
      echo "- \`sbom/${f}\` (missing)"
    fi
  done
} > "${OUT}/index.md"

echo "sbom_emit: wrote ${OUT}/index.md" >&2
