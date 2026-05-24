# Fixed-Wing Pusher-Prop Main Run

This page documents the `runit` main-run generator and detector baseline for a
strict-open fixed-wing pusher-prop public-proxy detection corpus. The corpus is
synthetic, reproducible, and audit-oriented. It does not assert measured
platform behavior, classified-fidelity behavior, or deployment performance.

## Scope

- Positive class: `fixed_wing_pusher_prop_public_proxy`.
- Negative class: `confuser_or_sensor_artifact`.
- Scenario groups: 10,000 in the full run.
- Phase records: 30,000, because each group is sliced into three locked phases.
- Positive scenario groups: 50 total.
- Positive phase records: 50 in each phase.
- Negative scenario groups: 9,750 total.
- Holdout: 1,500 scenario groups, including 8 positive groups.
- Train/CV pool: 8,500 scenario groups, including 42 positive groups.
- CV folds: five folds assigned at scenario-group level.

The phase windows are:

| phase_id | window |
| --- | --- |
| `initial_take_up` | 0-30 s |
| `climb_transition` | 30-90 s |
| `cruise_altitude` | 90-150 s |

## Commands

Smoke generation:

```bash
rtk python3 detection/generate_main_run.py --profile fixed-wing-pusher-proxy-v2 --out-root outputs/training-data/runit-fixed-wing-pusher-proxy-v2-main-run-smoke --scenario-groups 120 --positive-groups 12 --seed 202605210136 --force --smoke
```

Full generation:

```bash
rtk python3 detection/generate_main_run.py --profile fixed-wing-pusher-proxy-v2 --out-root outputs/training-data/runit-fixed-wing-pusher-proxy-v2-main-run --scenario-groups 10000 --seed 202605210136 --force
```

Detector and fusion run:

```bash
rtk python3 detection/run_main_run_detectors.py --data-root outputs/training-data/runit-fixed-wing-pusher-proxy-v2-main-run --out-root outputs/detection/runit-fixed-wing-pusher-proxy-v2-main-run --folds 5 --seed 202605210136
```

Validation:

```bash
rtk python3 -m unittest discover -s detection/real_data/tests -p 'test_*.py'
rtk just science-smoke
rtk bash ops/run-lane.sh vendor-scrub
rtk bash ops/run-lane.sh receipts
rtk just fast
```

## Artifact Layout

Generated data stays under `outputs/` and is not committed.

| path | purpose |
| --- | --- |
| `scenario_manifest.csv` | One row per scenario group with split role, fold, time lock, stratum, label, and audit metadata pointers. |
| `records.csv` | One row per scenario group and phase, carrying shared record identity and raw stream references. |
| `raw_complex_iq/` | Sharded active-radar complex IQ/pre-FFT proxy streams. |
| `acoustic_cues/` | Sharded time-locked acoustic cue streams. |
| `passive_rf_cues/` | Optional no-signal/passive-RF provenance features. |
| `raw_stream_index.csv` | Record-to-active-radar shard map. |
| `acoustic_stream_index.csv` | Record-to-acoustic shard map. |
| `passive_rf_index.csv` | Record-to-passive-RF shard map. |
| `detector_views/*.csv` | Detector-facing feature views with common record identity. |
| `restricted_truth/` | Generated audit metadata excluded from detector feature columns. |
| `dataset_manifest.json` | Run configuration, labels, phases, detector views, and leakage denylist. |
| `quality_report.json` | Count checks, split checks, phase checks, leakage checks, and raw stream summary. |

Detector outputs are written separately under `outputs/detection/...`:

| path | purpose |
| --- | --- |
| `classical_radar_processing.csv` | CA/OS-CFAR, MTD, and track-lifecycle proxy scores. |
| `ml_detector_baselines.csv` | Dependency-free tabular and sequence baseline scores. |
| `fusion_predictions.csv` | Calibrated late-fusion probabilities over all records. |
| `calibration_folds.csv` | Per-fold calibration accounting. |
| `calibration_manifest.json` | Split isolation proof for train/CV-only calibration. |
| `performance_metrics.csv` | Per-method ROC AUC, average precision, and train/CV-thresholded metrics for all records, train/CV, holdout, and holdout phases. |
| `performance_summary.json` | Holdout performance summary and threshold-selection metadata. |
| `fusion_quality_report.json` | Detector identity and holdout-isolation status. |

## Split Policy

Split assignment is deterministic from the run seed. Holdout groups are chosen
first, then positive quotas are allocated inside holdout and train/CV. For the
default full run this locks to 8 holdout positive groups and 42 train/CV
positive groups. The five CV folds are assigned by scenario group, with positive
and negative groups distributed independently so every fold receives both label
types in normal runs.

All three phase records for a scenario group share the same `scenario_group_id`,
`time_lock_id`, `split_role`, and `cv_fold`. The holdout split has an empty
`cv_fold` field and is never used for calibration.

## Detector Views

The generator exports four detector-view datasets and one fusion input view:

| view | method family |
| --- | --- |
| `high_resolution_xku_cuas.csv` | High-resolution X/Ku C-UAS radar features. |
| `tactical_s_band_aesa.csv` | Tactical S-band AESA radar features. |
| `gbad_3d4d_cueing.csv` | Medium-range 3D/4D GBAD cueing radar features. |
| `distributed_acoustic_cue.csv` | Distributed acoustic cue-network features. |
| `layered_fusion_c2.csv` | Provenance-aware fusion inputs from all source views. |

Each view includes the same `record_id`, `scenario_group_id`, `time_lock_id`,
phase timestamps, split fields, label fields, and raw stream references. Model
feature audits treat identity and split fields as metadata, not trainable
features.

## Noise Matrix

The Monte Carlo strata cover the following clutter and interference regimes:

`weibull_clutter`, `k_like_clutter`, `urban_edge`, `vegetation_motion`,
`sea_clutter`, `terrain_glint`, `rain`, `dust`, `open_sky`, `rfi_burst`,
`agc_compression`, `dropped_cpi`, `prf_ambiguity`, `doppler_folding`,
`clock_drift`, `calibration_offset`, `multipath_masking`, `dropout`,
`occlusion`, `scintillation`, `quantization`, and `mixed_scene`.

Negative groups include birds, RC fixed-wing aircraft, weather cells, ground
vehicles, wind turbines, terrain glints, multipath ghosts, RFI bursts, and
clutter-only counterfactuals. These hard negatives are for robustness and
false-alarm analysis.

## Fusion Method

`run_main_run_detectors.py` reads the same raw active-radar and acoustic shards
referenced by `records.csv`. It computes classical radar scores, acoustic
cadence scores, dependency-free ML baseline scores, and calibrated late-fusion
probabilities.

Calibration uses only the train/CV pool. Train/CV rows are scored by locked
folds, with each fold calibrated from the other folds. Holdout rows are scored
by a calibrator fit on the complete train/CV pool, and the calibration manifest
records `holdout_fit_record_count: 0`.

## Performance Outputs

`performance_metrics.csv` reports seven detector branches:

- `high_resolution_xku_cuas`,
- `tactical_s_band_aesa`,
- `gbad_3d4d_cueing`,
- `distributed_acoustic_cue`,
- `tabular_ml_baseline`,
- `sequence_ml_proxy`,
- `layered_fusion_c2`.

Each method has rows for all records, train/CV, holdout, and each holdout phase.
Thresholded metrics use a threshold selected on train/CV by maximum F1. The
paper-facing headline KPI is `LCB95 Recall@≤1%FPR`; ROC AUC and average
precision remain diagnostic ranking metrics. The values are synthetic benchmark
evidence only, not field-performance claims.

## Leakage Guard

Detector feature columns exclude scenario seeds, restricted generator metadata,
target role, target platform family, split keys, scenario group IDs, time locks,
fold IDs, and restricted truth paths. The ID columns remain present for audit and
joining, but the manifest marks them as metadata rather than model features.

The unit tests in `detection/real_data/tests/test_main_run.py` verify:

- run counts and phase windows,
- fixed-wing public-proxy positive labeling,
- scenario-group split and fold locking,
- raw stream alignment,
- detector-view record identity,
- leakage guard feature columns,
- fusion holdout isolation,
- performance metrics for all detector branches, with the low-FPR KPI as the headline guardrail.
