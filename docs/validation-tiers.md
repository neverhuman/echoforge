# EchoForge Validation Tiers

EchoForge artifacts carry a validation tier. The system accepts two tier
vocabularies side by side — Codex's original 5-string enum and the V0–V5
ladder — so that promotion gates and existing fixtures continue to work
together without a breaking change.

## Tier ladder (V0–V5) and string-alias mapping

| Tier | String alias       | Meaning                                                   |
|------|--------------------|-----------------------------------------------------------|
| V0   | `unvalidated`      | Declared only. No automated check has passed.             |
| V1   | `basic`            | Matches an analytic canonical scatterer within tolerance. |
| V2   | `cross_checked`    | V1 + agrees with an independent source within tolerance.  |
| V3   | `benchmarked`      | V2 + passes a benchmark suite with stable metrics.        |
| V4   | `measured_anchored`| V3 + per-feature distribution distance against public measured anchors within tolerance. |
| V5   | `measured`         | Compared against lawful measured data with provenance.    |

Both the string-alias form and the `V0..V5` form are accepted everywhere
the field appears (Rust `ValidationInfo.tier`, schema `validation.tier`,
CLI `--target-tier`). Producers may emit either; consumers normalize via
the alias table above.

## Promotion gates (per `ef validate`)

### V0 — declared
- Always passes. Emits a `validation_report.json` with `tier = V0`,
  `overall_status = pass`, and a single placeholder check.

### V1 — analytic
All of:
- Matching canonical scatterer (sphere, flat plate, dihedral, trihedral,
  cylinder, or cone) passes within the documented tolerance band.
- Polarization completeness check passes for the polarization basis
  declared in the manifest.
- Determinism replay passes (CPU bit-identical; GPU within FP tolerance
  policy in `crates/echoforge-validate/src/determinism.rs`).
- Unit and coordinate-frame check passes against the manifest declaration.
- `qa/canonical_validation.json` exists in the bundle.

### V2 — cross-solver
V1 prerequisites plus:
- `qa/cross_solver_delta.json` exists with at least one analytic-vs-solver
  pair within the documented tolerance band.
- `qa/convergence_report.json` exists with `p_observed ≥ 0.5 * p_expected`
  and monotonically decreasing successive deltas.

### V3 — benchmarked
V2 prerequisites plus:
- A `benchmark_report.json` with a `metrics` block containing `pd`,
  `pfa`, `range_error_m`, `doppler_error_mps`, `ospa_distance`,
  `sample_count`.
- Every observed metric beats its `V3MetricThresholds` envelope
  (defaults: `pd >= 0.85`, `pfa <= 0.01`, `range_error <= 5 m`,
  `doppler_error <= 1 m/s`, `ospa <= 10`).

Implementation: `crates/echoforge-validate/src/tier_v3.rs::evaluate_v3_gate`
and `evaluate_v3_from_benchmark_json`.

### V4 — measured-anchored
V3 prerequisites plus:
- A `calibration_report.json` with at least `min_anchors` (default 3)
  `V4DistributionAnchor` entries, each citing a public measured
  reference (e.g. Nature 2026 multi-sensor drone dataset, Rahman-
  Robertson K/W-band drone+bird, Karlsson 77 GHz FMCW Zenodo, VTT
  15/25 GHz fixed-wing UAV RCS).
- A `V4DistributionObservation` paired with each anchor, containing the
  simulator's samples for the same feature.
- Both distributions per anchor have at least
  `min_samples_per_distribution` (default 32) samples.
- For each anchor, the chosen 1-D distance metric (Wasserstein-1 or
  Kolmogorov-Smirnov) between observed and anchor samples is within
  the anchor-specific tolerance band.

Strict-open posture: V4 stores **distribution targets and citations
only**. Measured traces are not vendored into the repo. The actual
local-only measured data lives outside the repo per the operator's
`local_external_data_config.example.json` pattern; only the resulting
distribution distances and citations enter the calibration report.

V4 does NOT claim measured-truth equivalence; that is V5, reserved.

KTH drone/bird/human 77 GHz FMCW evidence can support V4
measured-anchor candidate claims for class-conditional distribution distance,
micro-Doppler overlap, hard-negative realism, and scan-gap/edge robustness.
For Shahed, Geran, or other Iranian-platform public proxies, KTH cannot
support V5 measured validation because it is not measured data for those
platforms and does not establish exact platform signatures, operational
detection range, proprietary sensor behavior, or classified fidelity.

Implementation: `crates/echoforge-validate/src/tier_v4.rs::evaluate_v4_gate`
and `evaluate_v4_from_calibration_report`.

### V5 — measured
Requires lawful measured data and explicit provenance; the strict-open
core does not ship V5-claimed artifacts by default. Promotion gate
ships in a follow-up packet alongside an explicit lawful-measured data
governance review.

## CLI surface

```
ef validate <BUNDLE_DIR>
    [--primitive <auto|sphere|flat_plate|dihedral|trihedral|cylinder|cone>]
    [--target-tier <v0|v1|v2>]
    [--write-report <PATH>]
    [--strict]
```

Exit codes: `0` = tier achieved, `1` = required check failed, `2` =
schema/IO error, `3` = invalid arguments. `--strict` upgrades warn → 1.

## Uncertainty score derivation

`ValidationInfo.uncertainty_score` is a scalar `[0, 1]` derived from the
aggregated error budget:

```
uncertainty_score = exp(-total_db / 6.0)
```

where `total_db = sqrt(analytic_db² + numeric_db² + method_db²)` per
`crates/echoforge-validate/src/tolerance.rs`. Rule of thumb: 6 dB of
combined error → score ≈ 0.37. The per-cell σ_db grid lives in
`tensors/uncertainty_sigma_db.zarr`; the scalar score is the manifest
surface for consumers that do not load the full uncertainty tensor.

## Fidelity class (F0–F5) — the method-ceiling axis

`ValidationInfo.fidelity_class` is an **optional** companion field to
`tier`. The two fields track different things and must not be conflated
(per the local coordination archive resolution that explicitly separates
method-ceiling from evidence-tier):

| Axis            | Field               | Vocabulary | Answers                                          |
|-----------------|---------------------|------------|--------------------------------------------------|
| Evidence        | `tier`              | V0–V5      | *How much evidence has this artifact accumulated?* |
| Method ceiling  | `fidelity_class`    | F0–F5      | *How good can this method ever be?*                |

### Fidelity ladder

| Class | Name                          | Meaning                                                                                   |
|-------|-------------------------------|-------------------------------------------------------------------------------------------|
| F0    | analytic-prior                | Closed-form / canonical scatterer prior (sphere Mie, plate PO, dihedral, etc.).            |
| F1    | primitive-decomposed          | Object decomposed into canonical primitives; superposition / facet sum.                   |
| F2    | production-SBR                | Shooting-and-bouncing-rays solver (production-quality), no cross-solver corroboration.    |
| F3    | cross-solver-bridged          | SBR + an independent solver agree within a documented delta band.                          |
| F4    | measured-anchored             | Method calibrated / anchored against lawful measured data, but not fully validated.        |
| F5    | lawful-measured-validated     | Method calibrated AND validated against lawful measured data with documented provenance.   |

### Why the two axes are independent

A producer can legitimately ship any combination of `tier` and
`fidelity_class`. Examples:

- `tier: V1-basic` + `fidelity_class: F2`: the artifact comes from a
  production-SBR method (F2), and the producer has accumulated only the
  V1-basic level of evidence (analytic spot-check passed; no
  cross-solver delta yet). Plenty of headroom on the evidence axis;
  method ceiling already chosen.
- `tier: V0-unvalidated` + `fidelity_class: F0`: the artifact is a
  declared-only analytic prior. No evidence has been collected; the
  method itself can never exceed the closed-form approximation.
- `tier: V2-cross_checked` + `fidelity_class: F4`: the method is
  measured-anchored (F4), and the producer has run the V2 cross-solver
  delta check. Note that V2 evidence does not magically promote the
  method's F-class — F-class only changes when the method itself
  changes.
- `tier: V5-measured` + `fidelity_class: F5`: fully lawful-measured and
  validated end-to-end. The strict-open core does not ship V5/F5
  artifacts.

### Promotion-gate behaviour (unchanged)

`fidelity_class` does **not** enter the `ef validate` promotion gate.
The V0/V1/V2 gate in `crates/echoforge-validate/src/report.rs`
continues to compute purely from the `ValidateChecks` block (analytic,
polarization, determinism, units/frame, cross-solver, convergence).
The fidelity field is a producer-asserted label that downstream
consumers can read alongside `tier` to reason about which artifacts to
trust for which downstream use.

### Field shape and back-compat

- JSON Schema: `schemas/common.schema.json#/$defs/validation.properties.fidelity_class`
  is an optional `string` enum constrained to `["F0", "F1", "F2", "F3",
  "F4", "F5"]`. Existing payloads that omit the field continue to
  validate unchanged.
- Rust: `crates/echoforge-core::ValidationInfo.fidelity_class:
  Option<String>` with `#[serde(default, skip_serializing_if =
  "Option::is_none")]`. Existing serialized payloads stay byte-stable
  when the field is absent.
- Producers: emit `fidelity_class` once you can assert the
  method-ceiling category; otherwise leave it absent.
