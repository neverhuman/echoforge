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
  --mode save-gate \
  --json "$report_json" \
  --md "$report_md" \
  --score-history "$report_history_jsonl" \
  --score-history-csv "$report_history_csv"
then
  echo "jankurai staged ratchet aggregate failed" >&2
  exit 1
fi
