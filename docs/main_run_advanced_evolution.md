# Main Run Advanced Evolution Lane

This page documents the separate advanced detector evolution lane for the
`runit-fixed-wing-pusher-proxy-v2-main-run` synthetic public-proxy corpus. It
does not replace or overwrite the canonical main-run detector outputs under
`outputs/detection/runit-fixed-wing-pusher-proxy-v2-main-run/`.

The lane is strict-open synthetic evidence only. It does not claim measured
truth, proprietary-equivalent behavior, classified fidelity, deployment
performance, or guaranteed real-world transfer for any platform or sensor.

## Command

```bash
rtk python3 detection/run_advanced_main_run_detectors.py \
  --data-root outputs/training-data/runit-fixed-wing-pusher-proxy-v2-main-run \
  --out-root outputs/detection/runit-fixed-wing-pusher-proxy-v2-main-run-advanced-evolution \
  --folds 5 \
  --seed 202605210136 \
  --search-profile v2_aggressive \
  --candidate-limit 512 \
  --evolution-rounds 10 \
  --evolution-sample-rows 12000 \
  --feature-cache outputs/detection/runit-fixed-wing-pusher-proxy-v2-main-run-advanced-evolution-feature-cache/v2_features.npz \
  --force
```

Smoke validation uses the same runner with a generated smoke corpus and
`--search-profile smoke --candidate-limit 96`.

## Method

The selected advanced family is
`spectral_transport_hypergraph_fusion`, with the locked winner
`meta_fusion.top8.v2_aggressive.geodesic_odds`. It loads `records.csv`, the raw
active radar IQ shards, acoustic shards, passive RF shards, and detector-view
CSVs. It then builds clean-room NumPy feature lifts:

- radar spectral, covariance, phase-coherence, wavelet-packet, DMD, and
  topology-proxy summaries,
- acoustic wavelet, cadence, entropy, cross-node coherence, and DMD summaries,
- passive RF no-signal/RFI/clock/provenance quality interactions,
- three-phase sequence deltas and hypergraph-style cross-modal edge features.

The `v2_aggressive` profile adds the clean-room transport and geometry layer:

- spectral conditioning, MP-style shrinkage mass, and multi-lag decorrelation,
- Hankel/SSA rank summaries, Morlet scattering summaries, and diffusion-Laplacian
  eigengap proxies,
- train/CV-only quantile transport, copula-rank geometry, Wasserstein-style
  prototype distances, and information-geometric alpha means.

Candidate evolution compares at least 96 candidate/calibrator combinations.
Internal selection optimizes average precision first, ROC AUC second, with
penalties for high false-positive rate and weak `initial_take_up` performance.
The second stage then fuses the top train/CV candidates with nonnegative
weights from a clean-room SHADE plus multi-resolution DE search before the
winner is locked.

## Isolation

Candidate generation, candidate selection, calibration, and threshold selection
use train/CV rows only. The holdout split is scored once for the single selected
winner after the train/CV winner is locked.

`selection_lock.json` is written before the holdout score pass starts.

The runner writes:

| path | purpose |
| --- | --- |
| `candidate_leaderboard.csv` | Train/CV-only candidate ranking and calibration metadata. |
| `advanced_feature_manifest.json` | Feature families, denylist, and split-isolation policy. |
| `evolution_trace.jsonl` | Candidate evolution trace, one train/CV candidate row per line. |
| `advanced_predictions.csv` | Selected winner predictions for train/CV and holdout. |
| `performance_metrics.csv` | Surface metrics plus train/CV candidate rows and selected holdout rows. |
| `performance_summary.json` | Summary, selected candidate, thresholds, and holdout policy. |
| `fusion_quality_report.json` | Output, isolation, and promotion-gate status. |

## Lock Discipline

The current canonical target to beat is `layered_fusion_c2`:

| split | AP | ROC AUC | F1 |
| --- | ---: | ---: | ---: |
| train/CV | 0.314444 | 0.924003 | 0.402888 |
| holdout | 0.128188 | 0.937723 | 0.240964 |

The selected winner is locked after train/CV-only search and holdout scoring is
used for the final evidence pass only. AP and ROC AUC remain secondary
diagnostics; a rerun-selected candidate is a reproducibility finding, not an
automatic replacement for the locked output.

Full-stream result:

| selected candidate | train/CV rank | train/CV AP | train/CV ROC AUC | holdout ROC AUC | holdout AP | accuracy | precision | recall | FPR | F1 | gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `meta_fusion.top8.v2_aggressive.geodesic_odds` | 1 | 0.841459 | 0.948087 | 0.958147 | 0.893817 | 0.998444 | 0.947368 | 0.750000 | 0.000223 | 0.837209 | pass |

The gate passed on the full 30,000-record stream, so the result is a locked
advanced experiment rather than a negative-only run.
