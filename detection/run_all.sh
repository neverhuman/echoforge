#!/usr/bin/env bash
# Run the top-three runnable detection consumers in sequence.
# Usage: bash detection/run_all.sh [--smoke] [--data-root PATH] [--out-root PATH]
set -euo pipefail

DATA_ROOT="outputs/training-data/shahed136-public-proxy-ml-training-v2-standard"
SMOKE_ROOT="outputs/training-data/shahed136-public-proxy-ml-training-v2-smoke"
OUT_ROOT="outputs/detection"
SEED=136
HORIZONS="5,15,45"
EPOCHS=2
SMOKE=0
SKIP_GENERATE=0
FORCE_REGENERATE=0
MAX_RECORDS=""
RECORDS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --smoke) SMOKE=1; shift ;;
        --skip-generate) SKIP_GENERATE=1; shift ;;
        --force-regenerate) FORCE_REGENERATE=1; shift ;;
        --data-root) DATA_ROOT="$2"; shift 2 ;;
        --out-root) OUT_ROOT="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --horizons) HORIZONS="$2"; shift 2 ;;
        --epochs) EPOCHS="$2"; shift 2 ;;
        --max-records) MAX_RECORDS="$2"; shift 2 ;;
        --records) RECORDS="$2"; shift 2 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

DETECTION_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[[ $SMOKE -eq 1 && "$DATA_ROOT" == "outputs/training-data/shahed136-public-proxy-ml-training-v2-standard" ]] && DATA_ROOT="$SMOKE_ROOT"

if [[ $SKIP_GENERATE -eq 0 ]]; then
    if [[ ! -f "$DATA_ROOT/frame_features.npz" || ! -f "$DATA_ROOT/records.csv" || ! -f "$DATA_ROOT/split_manifest.csv" || $FORCE_REGENERATE -eq 1 ]]; then
        [[ -z "$RECORDS" ]] && RECORDS=$([[ $SMOKE -eq 1 ]] && echo 1000 || echo 50000)
        SCALE=$([[ $SMOKE -eq 1 ]] && echo smoke || echo standard)
        python3 "$DETECTION_DIR/generate_ml_training_v2.py" \
            --out-root "$DATA_ROOT" --records "$RECORDS" \
            --scale-name "$SCALE" --seed "$SEED" --force
    fi
fi

SCRIPTS=(
    "01_cfar_tbd_fusion.py"
    "02_lightgbm_window_gbdt.py"
    "03_catboost_ordered_boosting.py"
)

for SCRIPT in "${SCRIPTS[@]}"; do
    ARGS=(python3 "$DETECTION_DIR/$SCRIPT" --data-root "$DATA_ROOT" --out-root "$OUT_ROOT"
          --seed "$SEED" --horizons "$HORIZONS" --epochs "$EPOCHS")
    [[ -n "$MAX_RECORDS" ]] && ARGS+=(--max-records "$MAX_RECORDS")
    [[ $SMOKE -eq 1 ]] && ARGS+=(--smoke)
    echo "running ${ARGS[*]}"
    "${ARGS[@]}"
done

REPORT="$OUT_ROOT/reports/auc_table.md"
[[ -f "$REPORT" ]] && echo "wrote $REPORT"
