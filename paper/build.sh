#!/usr/bin/env bash
set -euo pipefail

repo_root="$(CDPATH= cd -- "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
out_dir="${PAPER_OUT_DIR:-$repo_root/target/paper}"
copy_tracked=0

for arg in "$@"; do
  case "$arg" in
    --copy-tracked)
      copy_tracked=1
      ;;
    *)
      printf 'usage: %s [--copy-tracked]\n' "$0" >&2
      exit 64
      ;;
  esac
done

mkdir -p "$out_dir"
cd "$repo_root"

python3 -m detection.paper_evidence_major_upgrade_v1 \
  --training-root outputs/training-data/runit-fixed-wing-pusher-proxy-v2-main-run \
  --baseline-root outputs/detection/runit-fixed-wing-pusher-proxy-v2-main-run \
  --advanced-root outputs/detection/runit-fixed-wing-pusher-proxy-v2-main-run-advanced-evolution \
  --anchor-root outputs/real-data/kth-drone-bird-human-77ghz/kth-measured-v1 \
  --out-root outputs/paper-evidence/major-upgrade-v1 \
  --force

python3 paper/generate_figures.py \
  --training-root outputs/training-data/runit-fixed-wing-pusher-proxy-v2-main-run \
  --baseline-root outputs/detection/runit-fixed-wing-pusher-proxy-v2-main-run \
  --advanced-root outputs/detection/runit-fixed-wing-pusher-proxy-v2-main-run-advanced-evolution \
  --paper-evidence-root outputs/paper-evidence/major-upgrade-v1 \
  --anchor-root outputs/real-data/kth-drone-bird-human-77ghz/kth-measured-v1 \
  --strict

latexmk \
  -pdf \
  -bibtex \
  -interaction=nonstopmode \
  -halt-on-error \
  -file-line-error \
  -outdir="$out_dir" \
  paper/echoforge_ieee.tex

python3 paper/validate_paper.py \
  --tex paper/echoforge_ieee.tex \
  --bib paper/references.bib \
  --pdf "$out_dir/echoforge_ieee.pdf" \
  --figures-dir paper/figures \
  --paper-evidence-root outputs/paper-evidence/major-upgrade-v1

if [[ "$copy_tracked" -eq 1 ]]; then
  cp "$out_dir/echoforge_ieee.pdf" paper/echoforge_ieee.pdf
fi
