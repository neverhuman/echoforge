#!/usr/bin/env bash
# Node toolchain smoke tests for clean noninteractive shells.
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
expected="v$(tr -d '[:space:]' <"${repo_root}/.nvmrc" | sed 's/^v//')"
node_bin_dir="$(CDPATH= cd -- "$(dirname "$(command -v node)")" && pwd)"

clean_user_path="$(
  env -i HOME="${HOME}" PATH="${node_bin_dir}:/usr/bin:/bin" \
    bash -c "cd '${repo_root}' && node --version"
)"
if [[ "${clean_user_path}" != "${expected}" ]]; then
  echo "error: node --version in clean shell got ${clean_user_path}, want ${expected}" >&2
  exit 1
fi

bootstrap_path="$(
  env -i HOME="${HOME}" PATH="/usr/bin:/bin" \
    bash -c "cd '${repo_root}' && source ops/ci/node-toolchain.sh && node --version"
)"
if [[ "${bootstrap_path}" != "${expected}" ]]; then
  echo "error: repo Node bootstrap got ${bootstrap_path}, want ${expected}" >&2
  exit 1
fi

setup_node_bin="$(mktemp -d)"
setup_node_home="$(mktemp -d)"
trap 'rm -rf "${setup_node_bin}" "${setup_node_home}"' EXIT
for tool in node npm npx; do
  tool_path="$(command -v "$tool")"
  ln -s "$tool_path" "${setup_node_bin}/${tool}"
done

setup_node_path="$(
  env -i HOME="${setup_node_home}" PATH="${setup_node_bin}:/usr/bin:/bin" \
    bash -c "cd '${repo_root}' && source ops/ci/node-toolchain.sh && printf '%s %s\n' \"\$(node --version)\" \"\${ECHOFORGE_NODE_BIN}\""
)"
expected_setup_node="${expected} ${setup_node_bin}"
if [[ "${setup_node_path}" != "${expected_setup_node}" ]]; then
  echo "error: setup-node-style PATH bootstrap got ${setup_node_path}, want ${expected_setup_node}" >&2
  exit 1
fi

echo "node-toolchain-test: ${expected}"
