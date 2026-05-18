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
| V4   | (reserved)         | Reserved for extended benchmark / out-of-distribution.    |
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

### V3, V4, V5 — deferred
Promotion gates for V3+ ship in a follow-up packet. V5 requires lawful
measured data and explicit provenance; the strict-open core does not ship
V5-claimed artifacts.

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
