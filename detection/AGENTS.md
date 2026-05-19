# Detection Scripts — Agent Guidance

Ownership: runtime_core (science exception)
Test lane: `bash ops/run-lane.sh science-smoke`

## Science Exception Declaration

These Python scripts are standalone ML detection experiments under the science/ML
exception granted in AGENTS.md §3. They are NOT product code — they are research
pipelines that produce trained model artifacts consumed by the Rust runtime.

## Script Inventory

| Script | Algorithm | Output |
|--------|-----------|--------|
| `01_cfar_tbd_fusion.py` | CFAR + TBD Hough fusion | detection report JSON |
| `02_lightgbm_window_gbdt.py` | LightGBM windowed GBDT | model artifact (.lgbm) |
| `03_catboost_ordered_boosting.py` | CatBoost ordered boosting | model artifact (.cbm) |
| `04_tcn_inception_time.py` | TCN + InceptionTime | model artifact (.onnx) |
| `05_multiview_radar_transformer.py` | Multiview radar transformer | model artifact (.onnx) |
| `run_all.py` | Orchestrator | runs all above |

## Rules for Agents

- Scripts read from `outputs/` (generated data) and write model artifacts to `outputs/`
- Scripts must not write to `crates/`, `schemas/`, or `contracts/`
- Script results feed into Rust via ONNX artifacts or JSON reports — never direct coupling
- Do not add new Python dependencies without updating `pyproject.toml`
- Detection experiments must cite primary literature (IEEE/arXiv) in docstrings
