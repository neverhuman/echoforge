# Phase Pd/Pfa Report current

Status: current 24-group smoke report plus template fields for later regeneration.
Metric values below came from the local smoke run; do not commit generated
CSV/NPZ/JSON outputs from `outputs/`.

## Source Run

```bash
rtk python3 detection/generate_ml_training.py \
  --out-root outputs/training-data/shahed136-public-proxy-ml-training-smoke \
  --scenario-groups 24 \
  --max-time-s 150 \
  --scale-name smoke \
  --force
```

Expected generated inputs:

- `quality_report.json`
- `phase_operational_metrics.csv`
- `negative_control_audit.json`
- `generator_truth_denylist.json`
- `speed_prior_manifest.json`
- `kinematics_audit.csv`
- `calibration_anchor_manifest.json`
- `calibration_distance.csv`
- `calibration_report.json`
- `acoustic_node_detections.csv`
- `acoustic_cue_tracks.csv`
- `acoustic_phase_metrics.csv`
- `acoustic_product_schema.json`
- `acoustic_cue_quality.json`
- `feature_schema.json`
- `science_assumptions.json`

## Acceptance Summary

| Gate | Expected Status | Notes |
|---|---|---|
| Phase IDs | `initial_take_up`, `climb_transition`, `cruise_altitude` | Names must match exactly. |
| Phase windows | `pass` | `initial_take_up` is `0-30 s`; `climb_transition` is `30-90 s`; `cruise_altitude` begins at `90 s`. |
| Truth denylist | `pass` | Restricted generator truth must not appear in frame or aggregate feature columns. |
| Speed-prior kinematics | `pass` | Baseline positives stay in the 45-60 m/s pusher-prop prior; fast prop/jet variants stay stress-only. |
| Calibration anchor | `calibration_anchor_status` present | `pass` only when lawful measured-anchor distributions are supplied and distance checks pass; otherwise `reference_only` keeps smoke at unvalidated/basic. |
| Negative controls | `pass` | Seed, row-index, audit-metadata, shuffled-label, and target-masked probes must stay below the configured gate. |
| Initial low Pd | allowed | Low Pd is acceptable when line of sight is masked or the target is below the horizon. |
| Acoustic cueing sidecar | `pass` | Passive acoustic products report phase metrics, pre-radar-LOS cues, uncertainty ellipses, and broad false-cue sources without storing raw audio. |

## Phase Metrics

| Phase ID | Window | Record Count | Positive Count | Negative Count | Pd Any CFAR | Pfa Any CFAR | First-Hit Latency s | Track Initiation Latency s | Fragmentation Rate | Missed-Track Rate | False-Track Rate | Horizon-Masked Fraction | LOS Eligible Fraction | Micro-Doppler Confidence | Low-Pd Policy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `initial_take_up` | `0-30 s` | 96 | 24 | 72 | 0.291667 | 0.472222 | 19.928571 | 18.666667 | 0.750000 | 0.750000 | 0.416667 | 0.851264 | 0.148736 | 0.278307 | Low Pd allowed when LOS/horizon masked. |
| `climb_transition` | `30-90 s` | 96 | 24 | 72 | 1.000000 | 0.486111 | 17.020833 | 17.187500 | 3.125000 | 0.000000 | 0.486111 | 0.657197 | 0.342803 | 0.322893 | Track initiation and confirmation phase. |
| `cruise_altitude` | `90+ s` | 96 | 24 | 72 | 1.000000 | 0.583333 | 1.812500 | 0.543478 | 2.791667 | 0.041667 | 0.416667 | 0.596591 | 0.403409 | 0.335462 | Primary coherent Doppler and micro-Doppler interval. |

## Counterfactual Audit

| Probe | AUC | Gate | Status | Policy |
|---|---:|---:|---|---|
| seed-only | 0.500000 | `0.65` | pass | Must not predict labels from seeds. |
| row-index-only | 0.444444 | `0.65` | pass | Must not predict labels from row order. |
| audit-metadata probe | 0.439300 | `0.65` | pass | Audit-only metadata must not be treated as model-facing input. |
| shuffled-label sensor-feature | 0.323560 | `0.65` | pass | Shuffled labels should stay near chance. |
| target-masked counterfactual | 0.500000 | `0.65` | pass | Target-masked versus no-target records should not create a strong detector shortcut. |
| restricted generator-family probe | 1.000000 | n/a | audit-only | Demonstrates why class/family/scene-role metadata is denied to models. |

## Speed-Prior Audit

Fill this table from `quality_report.json` and `kinematics_audit.csv` when
regenerating at larger scale.

| Check | Value | Status | Policy |
|---|---:|---|---|
| baseline cruise speed band | `45-60 m/s` | pass | Baseline positives use the piston pusher-prop public-prior band. |
| fast prop / jet blended into positives | `0` | pass | Stress classes are not baseline positive data. |
| estimated ground speed exposed | `false` | pass | No true-speed or estimated-ground-speed feature is model-facing. |
| normal-speed low-radial examples | `2` | report | Low radial velocity can be crossing geometry, not slow target truth. |

## Calibration Distance

current smoke remains `unvalidated/basic synthetic public-proxy` evidence unless lawful
measured-anchor distributions are supplied and `calibration_anchor_status`
reports `pass`. Calibration artifacts must cite strict-open sources and target
distributions only. Do not vendor measured traces, raw captures, proprietary
sensor data, deployment metadata, or exact object signatures into the repo.

Expected calibration-anchor artifacts:

- `calibration_anchor_manifest.json`: citation IDs, license/source boundary,
  anchor eligibility, target distribution names, feature families, units, and
  claim boundary.
- `calibration_distance.csv`: per-anchor distribution-distance table used
  for the rows below.
- `calibration_report.json`: Rust-compatible calibration gate input with
  `anchors` and `observations`. Rollup status, including
  `calibration_anchor_status` and the unchanged unvalidated/basic claim boundary, is in
  `quality_report.json` and `calibration_anchor_manifest.json`.

| Anchor Set | Feature Family | Distance Metric | Value | Acceptance Band | Status |
|---|---|---|---:|---:|---|
| strict-open citation target distribution | Range/Doppler observables | TBD | TBD | TBD | synthetic-only or pass |
| strict-open citation target distribution | Micro-Doppler observables | TBD | TBD | TBD | synthetic-only or pass |
| strict-open citation target distribution | Clutter and false-alarm observables | TBD | TBD | TBD | synthetic-only or pass |

## Acoustic Cue Products

The acoustic sidecar is an acoustic-only cueing product. It does not claim
radar-fused posterior tracks; those belong to a later fusion lane.

Expected acoustic artifacts:

- `acoustic_node_detections.csv`: per-node source IDs, DOA/TDOA-style
  bearings, acoustic SNR proxy, cue latency, and engine/prop spectral features.
- `acoustic_cue_tracks.csv`: triangulated cue tracks, source attribution,
  uncertainty ellipses, cue confidence, and broad false-cue source categories.
- `acoustic_phase_metrics.csv`: `pd_any_acoustic_cue`,
  `pfa_any_acoustic_cue`, `first_cue_latency_s`,
  `track_initiation_latency_s`, `track_fragmentation_rate`,
  `false_track_rate`, and `missed_track_rate` for every current phase.
- `acoustic_cue_quality.json`: summary status, false-cue source counts,
  pre-radar-LOS cue evidence, and `raw_audio_stored: false`.

`initial_take_up` may report acoustic cues before radar line of sight. This is
expected for passive acoustic gap filling and must not be presented as
fire-control radar quality.

Current 24-group smoke acoustic metrics:

| Phase ID | Acoustic Pd | Acoustic Pfa | First-Cue Latency s | Track Init Latency s | Fragmentation Rate | False-Track Rate | Missed-Track Rate | Pre-Radar-LOS Cue Count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `initial_take_up` | 1.000000 | 0.333333 | 4.132473 | 4.132473 | 0.229167 | 0.333333 | 0.000000 | 47 |
| `climb_transition` | 1.000000 | 0.375000 | 4.843750 | 4.843750 | 0.166667 | 0.375000 | 0.000000 | 0 |
| `cruise_altitude` | 0.916667 | 0.229167 | 4.021390 | 4.021390 | 0.145833 | 0.229167 | 0.083333 | 0 |

## Claim Boundary

This report is simulator validation evidence for a strict-open synthetic
public-proxy benchmark. It must not be presented as measured radar performance,
real deployment performance, classified fidelity, or proprietary sensor
behavior. A current smoke run does not become measured-anchored unless the
calibration artifacts above are produced from lawful anchor distributions and
the rollup status passes.
