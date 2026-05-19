# Benchmark current Regeneration Report

Data root: `outputs/training-data/shahed136-public-proxy-ml-training-smoke`

## Decision

- Decision: `no_go`
- Regeneration ready: `false`
- Reason: current smoke artifacts are useful integrity evidence, but regeneration-ready claims are blocked by false-alarm/lifecycle calibration, reference-only calibration anchors, stale dependency aliases, and incomplete detector/fusion product lanes.

## Claim Boundary

This report is a regeneration readiness audit over synthetic public-proxy smoke artifacts. It is not measured radar validation, deployment guidance, or a claim of real sensor performance.

## Resolved Integrity Gates

| Gate | Status |
|---|---|
| `quality_report_status` | `pass` |
| `phase_windows_status` | `pass` |
| `negative_control_status` | `pass` |
| `truth_denylist_status` | `pass` |
| `speed_prior_kinematics_status` | `pass` |
| `calibration_artifact_status` | `pass` |
| `acoustic_cueing_status` | `pass` |
| `raw_audio_stored` | `False` |

## Readiness Metrics

| Metric | Value |
|---|---|
| `record_count` | 288 |
| `scenario_group_count` | 24 |
| `phase_ids` | climb_transition, cruise_altitude, initial_take_up |
| `min_radar_pd_any_cfar` | 0.292 |
| `max_radar_pfa_any_cfar` | 0.583 |
| `max_radar_false_track_rate` | 0.486 |
| `max_radar_track_fragmentation_rate` | 3.125 |
| `max_radar_missed_track_rate` | 0.750 |
| `max_acoustic_pfa_any_cue` | 0.375 |
| `pre_radar_los_acoustic_cue_count` | 47 |
| `calibration_anchor_status` | reference_only |
| `calibration_validation_tier` | unvalidated/basic synthetic public-proxy benchmark |
| `calibration_distance_failures` | none |

## Radar Phase Metrics

| Phase | Pd | Pfa | First Hit s | Track Init s | Frag | False Track | Missed | Horizon Masked |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| initial_take_up | 0.292 | 0.472 | 19.929 | 18.667 | 0.750 | 0.417 | 0.750 | 0.851 |
| climb_transition | 1.000 | 0.486 | 17.021 | 17.188 | 3.125 | 0.486 | 0.000 | 0.657 |
| cruise_altitude | 1.000 | 0.583 | 1.812 | 0.543 | 2.792 | 0.417 | 0.042 | 0.597 |

## Acoustic Phase Metrics

| Phase | Pd | Pfa | First Cue s | Frag | False Track | Missed | Pre-Radar LOS Cues |
| --- | --- | --- | --- | --- | --- | --- | --- |
| initial_take_up | 1.000 | 0.333 | 4.132 | 0.229 | 0.333 | 0.000 | 47.000 |
| climb_transition | 1.000 | 0.375 | 4.844 | 0.167 | 0.375 | 0.000 | 0.000 |
| cruise_altitude | 0.917 | 0.229 | 4.021 | 0.146 | 0.229 | 0.083 | 0.000 |

## Blockers

- `radar_false_alarm_calibration` (`blocked`): max radar Pfa=0.583, max false-track rate=0.486
  Required next evidence: empirical Pfa and false-track calibration tied to lifecycle metrics
- `track_lifecycle_baseline` (`blocked`): max fragmentation=3.125, max missed-track rate=0.750
  Required next evidence: track initiation, deletion, fragmentation, missed-track, and saturation baseline
- `measured_anchor_promotion` (`blocked`): calibration_anchor_status=reference_only
  Required next evidence: lawful measured-anchor distributions and passing distance checks
- `dependency_alias_reconciliation` (`blocked`): active current rows still reference stale  aliases while landed Rust receipts use link-budget-wire-in, complex-iq-end-to-end, k-weibull-clutter-wire-in, three-tier-phase-aware-detector, and unified-synthesize-scene
  Required next evidence: row reconciliation or explicit alias map in the regeneration receipt
- `detector_stack_completion` (`blocked`): Ku/X C-UAS, medium-range 3D surveillance, and detector fusion cue product rows remain open
  Required next evidence: source-attributed detector products and fused tracks before benchmark-facing detector regeneration

## Next Safe Packets

- `track-lifecycle-baseline`
- `ku-x-band-cuas-aesa-products`
- `medium-range-3d-surveillance-products`
- `detector-fusion-cue-products`
- `lane-g-c-multidist-cfar`
