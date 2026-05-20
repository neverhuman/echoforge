#!/usr/bin/env bash
# Repo-local Node bootstrap. This intentionally does not depend on shell
# startup files; CI scripts source it before invoking npm/node/npx.

echoforge_node_repo_root() {
  if [[ -n "${REPO_ROOT:-}" ]]; then
    printf '%s\n' "$REPO_ROOT"
    return 0
  fi
  git rev-parse --show-toplevel
}

echoforge_node_version() {
  local repo_root raw
  repo_root="$(echoforge_node_repo_root)"
  raw="$(tr -d '[:space:]' <"${repo_root}/.nvmrc")"
  printf '%s\n' "${raw#v}"
}

echoforge_node_bin_matches() {
  local bin_dir="${1:?usage: echoforge_node_bin_matches <bin-dir> <version>}"
  local version="${2:?usage: echoforge_node_bin_matches <bin-dir> <version>}"
  local actual
  [[ -x "${bin_dir}/node" ]] || return 1
  actual="$("${bin_dir}/node" --version 2>/dev/null | sed 's/^v//')" || return 1
  [[ "${actual}" == "${version}" ]]
}

echoforge_node_bin_dir() {
  local version="${1:?usage: echoforge_node_bin_dir <version>}"
  if [[ -n "${ECHOFORGE_NODE_BIN:-}" ]]; then
    printf '%s\n' "$ECHOFORGE_NODE_BIN"
    return 0
  fi

  local nvm_bin="${HOME}/.nvm/versions/node/v${version}/bin"
  if echoforge_node_bin_matches "$nvm_bin" "$version"; then
    printf '%s\n' "$nvm_bin"
    return 0
  fi

  local node_path path_bin
  if node_path="$(command -v node 2>/dev/null)"; then
    path_bin="$(CDPATH= cd -- "$(dirname -- "$node_path")" && pwd)"
    if echoforge_node_bin_matches "$path_bin" "$version"; then
      printf '%s\n' "$path_bin"
      return 0
    fi
  fi

  printf '%s\n' "$nvm_bin"
}

echoforge_export_node_toolchain() {
  local version bin_dir
  version="$(echoforge_node_version)"
  bin_dir="$(echoforge_node_bin_dir "$version")"
  if [[ ! -x "${bin_dir}/node" ]]; then
    return 1
  fi
  case ":${PATH}:" in
    *":${bin_dir}:"*) ;;
    *) export PATH="${bin_dir}:${PATH}" ;;
  esac
  export ECHOFORGE_NODE_BIN="$bin_dir"
  export ECHOFORGE_NODE_VERSION="$version"
}

echoforge_require_node_toolchain() {
  local version bin_dir
  version="$(echoforge_node_version)"
  bin_dir="$(echoforge_node_bin_dir "$version")"
  if echoforge_export_node_toolchain; then
    return 0
  fi
  cat >&2 <<EOF
error: EchoForge requires Node v${version} from .nvmrc, but it was not found at:
  ${bin_dir}

Fix:
  nvm install ${version}

or set ECHOFORGE_NODE_BIN to a bin directory containing node/npm/npx for v${version}.
EOF
  return 1
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  set -euo pipefail
  case "${1:-}" in
    --verify)
      echoforge_require_node_toolchain
      node --version
      npm --version
      npx --version
      ;;
    "" )
      echoforge_export_node_toolchain || echoforge_require_node_toolchain
      ;;
    * )
      echo "usage: $0 [--verify]" >&2
      exit 64
      ;;
  esac
else
  echoforge_export_node_toolchain || true
fi
