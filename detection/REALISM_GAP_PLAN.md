# Detection Realism Gap Plan

EchoForge detection metrics must be treated as simulator validation evidence, not measured radar performance. The reviewer feedback in `tips/not_real/*.txt` changes the bar from "make detection harder" to "make the benchmark scientifically falsifiable."

## Corrections In V2

- Use 50 scenario strata with grouped holdout coverage across band, range, grazing angle, clutter, aspect, motion, interference, visibility bucket, and confuser family.
- Treat SNR diagnostics as derived link-budget outputs, not label-only model features.
- Keep generator metadata out of model inputs. Detection scripts consume frame products and numeric sensor observables only; truth-like fields such as unsimulated altitude/elevation are audit-only until an angle-processing chain exists.
- Add causal horizon guards: feature builders can only consume frames with `time_s <= horizon_s`.
- Add leakage diagnostics: single-feature AUC, label/metadata audit, allowed-metadata-only probe, seed/hash-only probe, row-index-only probe, and shuffled-label probe.
- Add per-stratum, per-confuser, operating-point, calibration, track-latency, missed-track, fragmentation, and false-track reports.
- Store public calibration sources as citations and distribution targets only. No measured traces are vendored.

## Scientific Assumptions

- Current validation tier is `V0/V1 synthetic public-proxy`; it is not V5 measured-data validation.
- The link budget is a transparent first-order monostatic radar-equation diagnostic with public-proxy assumptions for band center, power, gain, bandwidth, noise figure, processing gain, and unmodeled losses.
- Clutter, RFI, multipath, impairment, and micro-Doppler products are synthetic observables designed for leakage and robustness testing, not calibrated sensor truth.
- Hard negatives are robustness work. They are not evasion optimization and are not operationally tuned to a classified or proprietary sensor.

## Deferred Correctness Work

- Complex IQ radar cube: waveform, propagation delay, phase, range processing, coherent Doppler, angle processing, and receiver impairments.
- Aspect/frequency/polarization RCS tables with uncertainty, canonical scatterer checks, and cross-solver comparisons.
- Terrain, land-cover, vegetation, infrastructure, low-grazing multipath, shadowing, and clutter maps tied to measured or public-domain anchors.
- Physical rotor/propeller, bird wingbeat, turbine, and vehicle micro-Doppler from coherent slow-time data.
- Multi-scan tracker with association, initiation, deletion, missed detections, track swaps, and false-track lifecycle.
- Measured-data validation using lawful public or locally authorized radar datasets, reported as sim-to-real distribution error before classifier AUC.

## Non-Goals

- Do not tune knobs simply to hit a preferred AUC band.
- Do not add arbitrary noise as a substitute for physical assumptions.
- Do not report exact measured-truth, proprietary-equivalent, or classified-fidelity claims.
