# Real-Data Measured-Anchor Lane

EchoForge uses public real radar datasets as calibration and benchmarking
anchors, not as committed training data and not as measured truth for any
specific public-proxy object. The first implementation target is V4-ready
infrastructure: registry metadata, local-only adapters, reference-only reports
when data is absent, and distribution-distance reports when a lawful local
anchor is supplied.

## Storage Policy

- Raw external data root:
  `${ECHOFORGE_REAL_DATA_ROOT:-$HOME/.cache/echoforge/real-data}`.
- Derived local reports:
  `outputs/real-data/<dataset_id>/<run_id>/`.
- Git-tracked content:
  `detection/real_data/catalog.json`, adapter code, docs, and tiny synthetic
  test fixtures only.
- Git-excluded content:
  raw measured archives, raw ADC/IQ, `.mat`, `.bag`, `.ulg`, `.pcap`, large
  binary tensors, normalized measured samples, and all generated real-data
  reports.

Adapters are manual and opt-in. CI dry-runs registry and policy checks but does
not download datasets. Missing local datasets produce `reference_only` anchor
reports instead of failures.

## Primary V1 Anchors

| Dataset | Role | Access and License Posture | First Adapter |
|---|---|---|---|
| KTH drone/bird/human 77GHz FMCW | Drone-vs-bird/human micro-Doppler and hard-negative calibration | Zenodo, CC-BY-4.0 | `kth_microdoppler_v1` |
| Open Radar Initiative outdoor moving objects | UAV/person/bicycle/vehicle surveillance products with range, azimuth, radial velocity, SNR, and Doppler spectra | Dataset CC-BY-NC-4.0; repo GPL-3.0 | `ori_outdoor_track_products_v1` |
| Han/Jung 2026 time-synchronized drone radar/RF | Direct drone radar/RF candidate | Article CC-BY-NC-ND-4.0; data access and license must be verified before local use | `han_jung_timesync_drone_v1` |
| RAD-DAR / RDRD | Compact drone/car/person Doppler benchmark | Kaggle account required; CC-BY-4.0 shown on Kaggle | `raddar_rdrd_v1` |
| IDF-DS fixed-wing telemetry | Takeoff, climb, cruise kinematic priors | Radar-free telemetry; verify Zenodo data license before use | `idf_ds_kinematics_v1` |

Secondary anchors such as RaDICaL, UW Raw ADC, RADIal, RADDet, and CARRADA are
registered for raw-chain or tensor-chain validation. K-Radar and RADIATE are
registered as later weather, adverse-condition, and large-scene clutter anchors.

## Adapter Contract

Each adapter writes only local derived reports:

- `anchor_manifest.json`
- `feature_distributions.json`
- `calibration_distance.csv`

With no local observations, `calibration_anchor_status` is `reference_only`.
With a local `observations.csv` under the dataset cache directory, the generic
adapter summarizes feature distributions and marks the result as
`measured_anchor_candidate`. That status is not a V5 measured-trace validation
claim and does not allow raw or normalized measured data into Git.

Minimal local observations format:

```csv
feature,value
snr_db,9.4
snr_db,10.1
doppler_bandwidth_hz,42.0
```

## Commands

```bash
rtk python3 -m detection.real_data.cli validate-catalog
rtk python3 -m detection.real_data.cli dry-run --dataset-id kth-drone-bird-human-77ghz
rtk python3 -m detection.real_data.cli build-report --dataset-id kth-drone-bird-human-77ghz --run-id local-check
rtk python3 -m detection.real_data.cli guard-git
```

## Metrics Policy

Detector-facing model features must be observable radar products. Generator
truth, split metadata, public-proxy family labels, synthetic shortcut features,
`normalized_snr`, and direct `*_proxy` micro-Doppler fields are audit-only.

AUC is retained as a leakage diagnostic. The realism gate reports `no_go` when
calibration, Pfa, lifecycle, or domain-holdout gates fail, even if AUC is high.
Headline metrics for V4 readiness are phase Pd/Pfa, first-hit latency, track
initiation, fragmentation, false-track and missed-track rates, calibration
distance, and leave-domain-out robustness.

## Claim Boundary

Real-data anchors are public measured references for distribution comparisons.
They do not create exact measured truth for any Iranian platform, proprietary
sensor equivalence, classified fidelity, or deployment-performance claims.
