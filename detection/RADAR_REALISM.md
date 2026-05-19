# Radar Realism current

current is the physics-first radar realism gate for the synthetic public-proxy
detection benchmark. It is not measured radar validation and does not claim
classified fidelity, proprietary sensor behavior, or real-object signature
equivalence.

The current contract is implemented by `detection/generate_ml_training.py`. The
generator publishes detector-facing frame products, operational phase metrics,
counterfactual audits, and generator-truth denylist evidence while keeping
restricted metadata out of model features.

## Three-Tier Phase Model

current replaces the older launch/climb/cruise phase labels with exactly three phase
IDs. Reports, quality gates, and downstream benchmark tables must use these
names verbatim.

| Tier | Phase ID | Time Window | Public-Proxy Kinematics | Radar Meaning | Lead Metrics |
|---|---|---:|---|---|---|
| 1 | `initial_take_up` | `0-30 s` | Booster/catapult transition, low altitude, unstable attitude, speed ramp from launch to partial powered flight | Geometry and line-of-sight limited; long-range ground radar Pd can legitimately be near zero when the target is horizon-masked | First hit probability, LOS eligibility, horizon-masked fraction, false alarms |
| 2 | `climb_transition` | `30-90 s` | Piston-prop climb, increasing speed and altitude, changing aspect/RCS | Track initiation and confirmation; classification remains uncertain | Pd/Pfa by range/altitude, track initiation latency, track fragmentation |
| 3 | `cruise_altitude` | `90+ s` | Stable pusher-prop cruise, broad altitude uncertainty, baseline speed prior below | Best interval for coherent Doppler, micro-Doppler, and classification evidence | Pd/Pfa, track continuity, micro-Doppler confidence, confuser separation |

Acceptance policy:

- Do not require `initial_take_up` to meet a preferred AUC band. If line of
  sight is masked or the target is below the horizon, low Pd is physically
  allowed and should be reported, not hidden.
- Report metrics by phase. The lead metrics are per-phase Pd/Pfa, first-hit
  latency, track initiation latency, track fragmentation, false-track rate,
  missed-track rate, micro-Doppler confidence, horizon-masked fraction, and
  LOS eligibility.
- Treat AUC as a secondary leakage diagnostic. "Best model AUC" is not the current
  headline metric.

## Speed-Prior Policy

Public speed assumptions are broad priors, not measured truth.

| Class | Policy |
|---|---|
| Baseline piston pusher-prop public proxy | Use a broad `45-60 m/s` working band, with `50-55 m/s` as the main cruise estimate. This is the baseline positive class. |
| Fast prop or modified variants | Keep above-baseline speeds as separate stress classes, not blended into the baseline positive distribution. |
| Jet-powered public proxy variants | Keep as `fast_jet_owa_public_proxy` stress data. The current current generator uses a separate fast-jet stress family around `110+ m/s`. |

Radar-observable policy:

- `radial_velocity_mps` is a radar observable and may be exposed to detector
  products.
- True ground speed is restricted generator truth. A crossing target can have
  low radial velocity while moving at normal cruise speed.
- `estimated_ground_speed_mps` may only be exposed after it is derived from
  track history or multi-view geometry. Until that estimator exists, it remains
  denylisted with other truth-like fields.

current writes `speed_prior_manifest.json` and `kinematics_audit.csv` as
audit-only artifacts. The audit verifies that baseline positives stay in the
baseline pusher-prop prior, fast prop and jet variants stay in stress classes,
and examples with normal cruise speed but low radial velocity are tracked as
line-of-sight geometry cases rather than slow targets.

## Calibration-Anchor Artifacts

Calibration-anchor integration is a distribution-distance audit, not a measured
trace ingestion path. current smoke remains `unvalidated/basic synthetic public-proxy` unless
lawful measured-anchor distributions are supplied and the calibration gate
reports a passing status.

Expected current calibration artifacts are:

- `calibration_anchor_manifest.json`: strict-open citation metadata, anchor
  eligibility notes, target distribution names, feature families, units, and
  license/source boundaries. It must not contain measured traces, raw samples,
  proprietary captures, deployment metadata, or exact platform signatures.
- `calibration_distance.csv`: per-feature-family distribution-distance rows
  comparing synthetic public-proxy outputs against the declared target
  distributions. Rows should include the anchor set, feature family, metric,
  value, acceptance band, status, and citation IDs.
- `calibration_report.json`: Rust-compatible calibration gate input with
  exactly `anchors` and `observations` collections. Rollup status and claim
  boundary fields live in `quality_report.json` and
  `calibration_anchor_manifest.json`.

The expected smoke status field is `calibration_anchor_status`. In ordinary
smoke runs without lawful measured anchors, it should report a reference-only
status and must not promote the benchmark beyond `unvalidated/basic`. Only runs with
eligible strict-open measured-anchor distributions and passing
distribution-distance checks may report this field as `pass`.

## Physics-First Generation Contract

current SNR must emerge from first-order radar-equation terms, not from a direct
target knob. The documented budget is:

- transmitter power
- transmit and receive gains
- wavelength/frequency
- aspect/frequency/polarization RCS lookup
- range-to-the-fourth loss
- propagation loss
- clutter loss
- receiver noise and bandwidth
- coherent processing gain
- system loss

`target_snr_db` is not a current generation input. Detector-facing products may
include derived observables such as `snr_db`, but truth or diagnostic SNR fields
stay out of model features.

Complex IQ, pulse compression, Doppler processing, angle processing, and
micro-Doppler extraction should preserve phase until detector-facing product
boundaries. Magnitude-only products are acceptable only after the coherent
processing boundary has been crossed.

Current current smoke products still mark micro-Doppler and spectrum summaries as
proxy detector products until `complex-iq-end-to-end` and
`coherent-microdoppler` land. The smoke gate is meant to enforce taxonomy,
truth-denylist, counterfactual, and operational-metric discipline while those
physics slots remain open.

Positive examples and confusers must flow through the same scene-generation
path. Class-specific behavior belongs in target traits, priors, RCS tables, and
source models, not in separate shortcut generator paths.

## Passive Acoustic Cue Products

current emits an acoustic cueing sidecar for the `acoustic-cueing-network-products`
lane. These artifacts are passive acoustic observation summaries, not raw
audio and not radar-fused tracks. Radar/acoustic posterior fusion belongs to a
later fusion lane.

Expected acoustic artifacts:

- `acoustic_node_detections.csv`: per-node cue rows with source IDs,
  DOA/TDOA-style bearing fields, cue latency, acoustic SNR proxy, engine/prop
  spectral features, and broad false-cue source categories.
- `acoustic_cue_tracks.csv`: triangulated cue tracks with source
  attribution, uncertainty ellipses, cue confidence, cue latency, and spectral
  summaries.
- `acoustic_phase_metrics.csv`: acoustic Pd/Pfa, first-cue latency,
  track-init latency, fragmentation, false-track rate, missed-track rate, and
  pre-radar-LOS cue counts by `initial_take_up`, `climb_transition`, and
  `cruise_altitude`.
- `acoustic_product_schema.json` and `acoustic_cue_quality.json`: schema
  and quality evidence. `raw_audio_stored` must remain `false`.

The acoustic sidecar may show cues during `initial_take_up` even when radar Pd
is low because radar line-of-sight is horizon masked. That is a reporting
feature, not a contradiction. It reflects acoustic cueing as a gap-filler
signal and does not claim a fire-control-quality radar track.

## Truth-Denylist Policy

Generation must fail if restricted generator truth appears in published frame
columns or aggregate feature columns. current writes `generator_truth_denylist.json`
with the evaluated status.

Restricted feature names include:

- `scenario_seed`
- `object_seed`
- `class_id`
- `target_family`
- `scene_role`
- `phase_id`
- `altitude_m`
- `nominal_snr_db`
- `link_budget_snr_db`
- `raw_rcs_dbsm`
- `rcs_dbsm`
- `validation_tier`
- `calibration_anchor_ids`
- `source_metadata`
- `true_speed_mps`
- `ground_speed_mps`
- `estimated_ground_speed_mps`

Allowed detector-facing inputs are limited to frame tensor columns,
valid-frame masks, and aggregate sensor observables declared by
`feature_schema.json`. `records.csv`, `split_manifest.csv`, and
`restricted_truth/*.json` are audit-only.

## Counterfactual Controls

Each current counterfactual group keeps the site realization, sensor realization,
clutter regime, interference state, scan cadence, and phase model matched while
varying the scene role.

Required negative controls:

- `no_target_counterfactual`
- `target_masked_counterfactual`
- seed-only probe
- row-index-only probe
- audit-metadata probe
- shuffled-label sensor-feature probe
- restricted generator-family probe

The restricted generator-family probe is reported to demonstrate why class,
family, scene-role, seed, phase, and truth metadata are audit-only. It is not a
valid detector feature set.

## current Smoke Command

Generated outputs must remain under `outputs/` or another ignored artifact
directory.

```bash
rtk python3 detection/generate_ml_training.py \
  --out-root outputs/training-data/shahed136-public-proxy-ml-training-smoke \
  --scenario-groups 64 \
  --max-time-s 150 \
  --scale-name smoke \
  --force
```

Expected smoke evidence:

- `quality_report.json` has `benchmark_profile:
  ml-training-three-tier`.
- `phase_ids` are exactly `initial_take_up`, `climb_transition`, and
  `cruise_altitude`.
- `phase_windows_status` is `pass`.
- `truth_denylist_status` is `pass`.
- `speed_prior_kinematics_status` is `pass`.
- `calibration_anchor_status` is present. It is `pass` only when lawful
  measured-anchor distributions were supplied; otherwise it records
  `reference_only` and the benchmark remains `unvalidated/basic`.
- `acoustic_cueing_status` is `pass` and `acoustic_cue_quality.json`
  confirms no raw audio is stored.
- `negative_control_status` is `pass`.
- `initial_take_up_low_pd_allowed` is `true`.
- `phase_operational_metrics.csv` reports Pd/Pfa and lifecycle metrics by
  phase.
- `acoustic_phase_metrics.csv` reports acoustic Pd/Pfa, latency,
  fragmentation, false-track, and missed-track metrics by phase.

The companion runner reuses or generates the dataset, prints the current status, and
writes `run_summary.json` without training detector models:

```bash
rtk python3 detection/run_all.py \
  --smoke \
  --data-root outputs/training-data/shahed136-public-proxy-ml-training-smoke \
  --out-root outputs/detection-smoke \
  --skip-generate
```

## Claim Boundary

current remains `unvalidated/basic synthetic public-proxy` unless a lawful measured-anchor gate
explicitly reports passing distribution-distance checks through
`calibration_report.json`, `calibration_distance.csv`, and
`calibration_anchor_manifest.json`. It does not ship measured traces, real
deployment assumptions, engagement guidance, or exact platform behavior.
Public sources are used as broad priors, target distributions, and stress-class
boundaries, not as claims of measured target performance.
