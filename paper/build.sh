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
training_root="outputs/training-data/fixed-wing-pusher-proxy-main-run"
baseline_root="outputs/detection/fixed-wing-pusher-proxy-main-run"
advanced_root="outputs/detection/fixed-wing-pusher-proxy-main-run-advanced-evolution"
paper_evidence_root="outputs/paper-evidence/current"
anchor_root="outputs/real-data/kth-drone-bird-human-77ghz/kth-measured"
real_data_root="${ECHOFORGE_REAL_DATA_ROOT:-$HOME/.cache/echoforge/real-data}"
kth_cache_root="$real_data_root/kth-drone-bird-human-77ghz"
training_records="$training_root/records.csv"
baseline_summary="$baseline_root/performance_summary.json"
advanced_lock="$advanced_root/selection_lock.json"
advanced_trace="$advanced_root/evolution_trace.csv"
anchor_report="$anchor_root/raw_shape_report.json"

if ! python3 - <<'PY'
import importlib.util
import sys

sys.exit(0 if importlib.util.find_spec("numpy") is not None else 1)
PY
then
  python3 -m pip install --user --quiet numpy
fi

if ! python3 - <<'PY'
import importlib.util
import sys

sys.exit(0 if importlib.util.find_spec("matplotlib") is not None else 1)
PY
then
  python3 -m pip install --user --quiet matplotlib
fi

if ! python3 - <<'PY'
import importlib.util
import sys

sys.exit(0 if importlib.util.find_spec("PIL") is not None else 1)
PY
then
  python3 -m pip install --user --quiet pillow
fi

# GitHub Actions starts from a clean checkout, so bootstrap the generated
# evidence roots instead of depending on local outputs.
if ! python3 - "$kth_cache_root" <<'PY'
import hashlib
import sys
from pathlib import Path

root = Path(sys.argv[1])
expected = {
    "data_SAAB_SIRS_77GHz_FMCW.npy": "01d66ba7b1ccc04477a9e69b2813f251",
    "ReadMe.txt": "efd7164e5855d2a0016c7a85d9006bc8",
}
for name, digest in expected.items():
    path = root / name
    if not path.is_file():
        raise SystemExit(1)
    if hashlib.md5(path.read_bytes()).hexdigest() != digest:
        raise SystemExit(1)
PY
then
  mkdir -p "$kth_cache_root"
  curl -L --fail --continue-at - \
    -o "$kth_cache_root/data_SAAB_SIRS_77GHz_FMCW.npy" \
    "https://zenodo.org/records/5896641/files/data_SAAB_SIRS_77GHz_FMCW.npy"
  curl -L --fail \
    -o "$kth_cache_root/ReadMe.txt" \
    "https://zenodo.org/records/5896641/files/ReadMe.txt"
fi

if [[ ! -s "$anchor_report" ]]; then
  python3 -m detection.real_data.cli build-report \
    --dataset-id kth-drone-bird-human-77ghz \
    --run-id kth-measured \
    --raw-root "$real_data_root" \
    --out-root outputs/real-data
fi

if [[ ! -s "$training_records" ]]; then
  python3 -m detection.generate_main_run \
    --profile fixed-wing-pusher-proxy \
    --out-root "$training_root" \
    --scenario-groups 10000 \
    --seed 202605210136 \
    --jamming-deception-rate 0.15 \
    --force
fi

if [[ ! -s "$baseline_summary" ]]; then
  python3 -m detection.run_main_run_detectors \
    --data-root "$training_root" \
    --out-root "$baseline_root" \
    --folds 5 \
    --seed 202605210136 \
    --force
fi

# The advanced detector lane writes the paper evidence lock and supporting
# diagnostics.
if [[ ! -s "$advanced_lock" || ! -s "$advanced_trace" || \
      detection/advanced_main_run_detectors.py -nt "$advanced_trace" || \
      detection/crypt_ip_impl/advanced_main_run_detectors.py -nt "$advanced_trace" || \
      detection/ei_evolution_trace.py -nt "$advanced_trace" || \
      detection/crypt_ip_impl/ei_evolution_trace.py -nt "$advanced_trace" || \
      detection/paper_evidence_builder.py -nt "$advanced_trace" ]]; then
  python3 -m detection.run_advanced_main_run_detectors \
    --data-root "$training_root" \
    --out-root "$advanced_root" \
    --folds 5 \
    --seed 202605210136 \
    --search-profile aggressive \
    --candidate-limit 128 \
    --evolution-rounds 5 \
    --evolution-sample-rows 4500 \
    --write-component-scores \
    --force
fi

python3 -m detection.paper_evidence \
  --training-root "$training_root" \
  --baseline-root "$baseline_root" \
  --advanced-root "$advanced_root" \
  --anchor-root "$anchor_root" \
  --out-root "$paper_evidence_root" \
  --strict-ei-trace \
  --force

python3 paper/generate_metric_macros.py \
  --paper-evidence-root "$paper_evidence_root" \
  --strict

python3 paper/generate_source_appendix.py --strict

python3 paper/generate_figures_focused.py \
  --training-root "$training_root" \
  --baseline-root "$baseline_root" \
  --advanced-root "$advanced_root" \
  --paper-evidence-root "$paper_evidence_root" \
  --strict

python3 paper/validate_visuals.py

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
  --paper-evidence-root "$paper_evidence_root"

if [[ "$copy_tracked" -eq 1 ]]; then
  cp "$out_dir/echoforge_ieee.pdf" paper/echoforge_ieee.pdf
fi
