# Track Lifecycle Baseline current

Data root: `outputs/training-data/shahed136-public-proxy-ml-training-smoke`

## Status

- Artifact status: `pass`
- Detector family ID: `track-lifecycle-baseline`
- Lifecycle rows: `288`
- Model-facing default: `false`

## Claim Boundary

Synthetic track-lifecycle smoke baseline over current detector-facing frame products. Labels are used only for aggregate audit metrics; lifecycle rows are not measured tracking validation.

## Phase Metrics

| Phase | Pd | Pfa | Confirm | False Track | Missed | Init s | Frag | Delete | Stable |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| initial_take_up | 0.292 | 0.472 | 0.250 | 0.417 | 0.750 | 18.667 | 0.750 | 0.833 | 0.125 |
| climb_transition | 1.000 | 0.486 | 1.000 | 0.486 | 0.000 | 17.188 | 3.125 | 0.167 | 0.333 |
| cruise_altitude | 1.000 | 0.583 | 0.958 | 0.417 | 0.042 | 0.543 | 2.792 | 0.043 | 0.667 |

## Audit Checks

| Check | Value |
|---|---|
| Denied lifecycle columns | none |
| Missing lifecycle columns | none |
| Missing phase metric columns | none |
| Missing source IDs | 0 |
| Detector IDs OK | True |

## Interpretation

- This baseline gives later detector products a reproducible lifecycle reference for initiation, deletion, fragmentation, false tracks, and misses.
- Current false-track, missed-track, and fragmentation values remain blocker evidence for regeneration-ready claims.
- `initial_take_up` may legitimately remain weak for radar because line of sight is often horizon-masked; the row is reported rather than hidden.
