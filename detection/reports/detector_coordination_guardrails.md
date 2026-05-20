# Detector current Coordination Guardrails

Status: coordination report based on the current local `FUCKIT.md`,
`detection/RADAR_REALISM.md`, `detection/reports/phase_pd_pfa.md`, and
the latest detector/radar messages as of `20260518T201219Z`.

This report is a guardrail for Radar/Detector current coordination. It is not a new
benchmark result, measured radar validation, or claim of proprietary-equivalent
sensor behavior.

## Guardrails

### 1. current Report Phases vs Rust Detector Sub-Tiers

Do not conflate the current report phase windows with the Rust detector's internal
phase-aware sub-tiers.

| Layer | Windows | Meaning |
|---|---|---|
| current report phases | `0-30 s`, `30-90 s`, `90+ s` | Public benchmark reporting phases: `initial_take_up`, `climb_transition`, and `cruise_altitude`. These are the only phase IDs for current quality reports, phase Pd/Pfa tables, acoustic phase metrics, and benchmark-facing evidence. |
| Rust detector sub-tiers | `1-3 s`, `3-30 s`, `>30 s` | Internal detector logic for boost, climb-out, and cruise discrimination in the phase-aware Rust detector. These are implementation sub-tiers and must not replace current report phase IDs. |

The Rust sub-tier timing can explain detector behavior inside the broader current
phase rows, but downstream reports should continue to aggregate and label
results by `initial_take_up`, `climb_transition`, and `cruise_altitude`.

### 2. Smoke/Template Done vs Lifecycle-Complete Done

`phase-pd-pfa-report` is smoke/template complete. It provides a 24-group
smoke report, table schema, expected artifacts, and claim boundaries.

It is not lifecycle-complete evidence for the current benchmark. Lifecycle-complete
status still depends on the open calibration and detector work needed for
empirical false-alarm behavior, track lifecycle baselines, false-track
calibration, and larger-scale regeneration evidence.

Use "done" carefully:

- `done` for smoke/template means the report shape and smoke artifacts exist.
- `done` for lifecycle-complete should require calibrated empirical Pfa,
  false-track/missed-track evidence, track initiation/fragmentation evidence,
  and reconciled benchmark regeneration inputs.

### 3. Acoustic-Only Cues vs Later Radar/Acoustic Fusion

The current acoustic lane emits passive acoustic cue products only:

- `acoustic_node_detections.csv`
- `acoustic_cue_tracks.csv`
- `acoustic_phase_metrics.csv`
- `acoustic_product_schema.json`
- `acoustic_cue_quality.json`

These are cueing and uncertainty products, not radar-fused posterior tracks.
Acoustic cues during `initial_take_up` can legitimately precede radar line of
sight and should be reported as acoustic gap-filling evidence.

Radar/acoustic Bayesian fusion belongs to the later fusion lane. Until that lane
lands, do not describe acoustic cue tracks as fused radar tracks or as
fire-control-quality radar evidence.

### 4. AUC Is a Legacy Leakage Diagnostic

AUC remains useful for detecting shortcut leakage, especially the legacy
single-feature and shuffled-label failure modes from earlier detector work.

For current, AUC is not the headline realism gate. The headline gates are
phase-aware operational metrics: Pd/Pfa by phase, first-hit latency,
track-initiation latency, fragmentation, false-track rate, missed-track rate,
micro-Doppler confidence, horizon-masked fraction, LOS eligibility, truth
denylist status, counterfactual controls, speed-prior audit, calibration status,
and acoustic cue quality.

Do not promote "best model AUC" as the current acceptance result.

### 5. High Radar CFAR Pfa Blocks Regeneration Claims

The 24-group current smoke report shows radar CFAR false-alarm probabilities around
`0.47-0.58` across phases. Those smoke values are intentionally visible and are
too high for benchmark-regeneration claims.

Do not claim regenerated benchmark readiness from current smoke evidence until
empirical-Pfa calibration and false-track calibration are reconciled. The latest
radar messages show an empirical Pfa calibrator has landed and exposed
calibration gaps; that is evidence of honest measurement, not production-grade
closure.

Benchmark regeneration should wait for:

- empirical Pfa calibration across the relevant clutter and CFAR variants;
- false-track and missed-track calibration tied to track lifecycle metrics;
- updated phase Pd/Pfa evidence after those calibrations land;
- a receipt that names the committed packet aliases used for the regeneration.

### 6. Reconcile Stale current Dependency Rows

Some current dependency rows still reference packet names that predate the baseline
commit and the newer committed Rust packet aliases. After Claude's baseline
commit `9586ad2` (`Radar-expert credibility sweep: Wave A baseline + Waves
1-4`), future coordination should reconcile stale current dependency names to the
committed aliases and receipts.

Examples to reconcile include old current dependency names that point at open
`*` rows when the underlying Rust work has landed under non-`` packet
aliases such as:

- `link-budget-wire-in`
- `complex-iq-end-to-end`
- `k-weibull-clutter-wire-in`
- `propeller-generator-wire-in`
- `mti-mtd-filter-bank`
- `empirical-pfa-calibrator`
- `three-tier-phase-aware-detector`
- `unified-synthesize-scene`

Until the rows are reconciled, dependency status should be read from committed
packet aliases plus their receipts, not from stale row names alone.

## Coordination Rule

When a later packet produces detector-facing results, its report should state
which layer it is using:

- current report phase rows: `0-30 s`, `30-90 s`, `90+ s`.
- Rust detector sub-tier internals: `1-3 s`, `3-30 s`, `>30 s`.
- Acoustic-only cue products: passive cue evidence, not fused tracks.
- Fusion products: only after the radar/acoustic fusion lane lands.
- AUC: leakage diagnostic only.
- Regeneration claims: blocked until empirical Pfa and false-track calibration
  are closed and cited.
