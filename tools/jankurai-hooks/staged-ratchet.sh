#!/usr/bin/env bash
# Fast staged-file ratchet for pre-commit.
set -euo pipefail

if [ "${JANKURAI_SKIP_HOOKS:-}" = "1" ]; then
  exit 0
fi

repo_root="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
git_dir="$(git rev-parse --git-dir 2>/dev/null || printf '%s/.git' "$repo_root")"
case "$git_dir" in
  /*) ;;
  *) git_dir="$repo_root/$git_dir" ;;
esac
jankurai_dir="$git_dir/jankurai"
mkdir -p "$jankurai_dir"

if [ -n "${JANKURAI_BIN:-}" ] && [ -x "$JANKURAI_BIN" ]; then
  jankurai_cmd="$JANKURAI_BIN"
else
  jankurai_cmd="${JANKURAI_FALLBACK_BIN:-jankurai}"
fi

cd "$repo_root"

if [ -z "${LEFTHOOK_BIN:-}" ] && ! command -v lefthook >/dev/null 2>&1; then
  os_arch="$(uname | tr '[:upper:]' '[:lower:]')"
  cpu_arch="$(uname -m | sed 's/aarch64/arm64/;s/x86_64/x64/')"
  for candidate in \
    "$repo_root/node_modules/lefthook-${os_arch}-${cpu_arch}/bin/lefthook" \
    "$repo_root/node_modules/@evilmartians/lefthook/bin/lefthook-${os_arch}-${cpu_arch}/lefthook" \
    "$repo_root/node_modules/@evilmartians/lefthook-installer/bin/lefthook" \
    "$HOME"/.npm/_npx/*/node_modules/lefthook-"${os_arch}-${cpu_arch}"/bin/lefthook
  do
    if [ -x "$candidate" ]; then
      export LEFTHOOK_BIN="$candidate"
      export PATH="$(dirname "$candidate"):$PATH"
      break
    fi
  done
fi

report_dir="${JANKURAI_HOOK_REPORT_DIR:-target/jankurai/hooks}"
mkdir -p "$report_dir/staged"
report_json="$report_dir/pre-commit-score.json"
report_md="$report_dir/pre-commit-score.md"
report_history_jsonl="$report_dir/pre-commit-score-history.jsonl"
report_history_csv="$report_dir/pre-commit-score-history.csv"
staged_manifest="$report_dir/staged-manifest.txt"

git diff --cached --name-status --diff-filter=ACMRD > "$staged_manifest"
if [ ! -s "$staged_manifest" ]; then
  printf '{"score":0,"raw_score":0,"minimum_score":0,"hard_findings":0,"findings":0}\n' > "$report_json"
  printf '# staged ratchet\n\nNo staged files.\n' > "$report_md"
  : > "$report_history_jsonl"
  : > "$report_history_csv"
  exit 0
fi

while IFS=$'\t' read -r status path extra; do
  [ -n "${status:-}" ] || continue
  case "$status" in
    R*|C*)
      rename_from="$path"
      path="$extra"
      op="rename"
      ;;
    A)
      rename_from=""
      op="create"
      ;;
    D)
      rename_from=""
      op="delete"
      ;;
    *)
      rename_from=""
      op="modify"
      ;;
  esac

  staged_path="$report_dir/staged/${path//\//__}"
  mkdir -p "$(dirname "$staged_path")"

  candidate="$staged_path.candidate"
  baseline="$staged_path.baseline"

  if git show ":$path" >/dev/null 2>&1; then
    git show ":$path" > "$candidate"
  else
    : > "$candidate"
  fi

  baseline_source="$path"
  if [ "$op" = "rename" ] && [ -n "${rename_from:-}" ]; then
    baseline_source="$rename_from"
  fi

  if git show "HEAD:$baseline_source" >/dev/null 2>&1; then
    git show "HEAD:$baseline_source" > "$baseline"
  else
    : > "$baseline"
  fi

  file_json="$staged_path.json"
  audit_args=(
    audit-file
    --path "$path"
    --candidate "$candidate"
    --baseline "$baseline"
    --op "$op"
    --mode save-gate
    --json-out "$file_json"
  )
  if [ -n "${rename_from:-}" ]; then
    audit_args+=(--rename-from "$rename_from")
  fi
  if ! "$jankurai_cmd" "${audit_args[@]}" >/dev/null; then
    if node - "$file_json" "$staged_manifest" <<'NODE' >/dev/null 2>&1
const fs = require("node:fs");

const [fileJson, stagedManifest] = process.argv.slice(2);
const report = JSON.parse(fs.readFileSync(fileJson, "utf8"));
const staged = new Set(
  fs
    .readFileSync(stagedManifest, "utf8")
    .trim()
    .split(/\n+/)
    .filter(Boolean)
    .map((line) => {
      const [status, path, extra] = line.split("\t");
      return status.startsWith("R") || status.startsWith("C") ? extra : path;
    })
);
const hard = report?.blocking?.new_hard_findings ?? [];
if (hard.length === 0) {
  process.exit(1);
}
const ok = hard.every((finding) => {
  if (finding.check_id !== "HLT-042-CI-LOCAL-PARITY:ci") return false;
  const match = String(finding.problem ?? "").match(/missing script `([^`]+)`/);
  return Boolean(match && staged.has(match[1]));
});
process.exit(ok ? 0 : 1);
NODE
    then
      echo "jankurai staged ratchet deferred cross-staged script check for $path" >&2
      continue
    fi
    echo "jankurai staged ratchet failed for $path" >&2
    exit 1
  fi
done < "$staged_manifest"

changed_from_args=()
if git rev-parse --verify HEAD >/dev/null 2>&1; then
  changed_from_args=(--changed-from HEAD)
fi

if ! "$jankurai_cmd" audit . \
  --changed-fast \
  "${changed_from_args[@]}" \
  --mode advisory \
  --json "$report_json" \
  --md "$report_md" \
  --score-history "$report_history_jsonl" \
  --score-history-csv "$report_history_csv"
then
  echo "jankurai staged ratchet aggregate failed" >&2
  exit 1
fi
