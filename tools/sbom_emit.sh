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

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
if [ -f "${REPO_ROOT}/ops/ci/node-toolchain.sh" ]; then
  # shellcheck source=ops/ci/node-toolchain.sh
  source "${REPO_ROOT}/ops/ci/node-toolchain.sh"
fi

OUT="sbom"
mkdir -p "${OUT}"

echo "== Rust cargo SBOM ==" >&2
if command -v cargo-cyclonedx >/dev/null 2>&1; then
  # cargo-cyclonedx 0.5.x writes each workspace package to <manifest_dir>/<name>.cdx.json
  # via --override-filename. Aggregate into sbom/ after emission.
  if rtk cargo cyclonedx --format json --spec-version 1.5 --override-filename cargo.cdx >/dev/null 2>&1; then
    # Move every emitted cargo.cdx.json from workspace members into sbom/, namespacing by crate dir.
    while IFS= read -r path; do
      crate_dir="$(dirname "${path}")"
      crate_name="$(basename "${crate_dir}")"
      cp "${path}" "${OUT}/cargo.${crate_name}.cdx.json"
      rm -f "${path}"
    done < <(find . -name 'cargo.cdx.json' -not -path './sbom/*' -not -path './target/*' -not -path '/Volumes/MOE/*')
    # Collapse into a single roll-up if any were produced.
    first_cdx="$(ls -1 "${OUT}"/cargo.*.cdx.json 2>/dev/null | head -1)"
    if [ -n "${first_cdx}" ] && [ ! -f "${OUT}/cargo.cdx.json" ]; then
      cp "${first_cdx}" "${OUT}/cargo.cdx.json"
    fi
  else
    echo "warn: cargo cyclonedx invocation failed" >&2
  fi
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
