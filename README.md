<img src="./assets/ecoforge.png" alt="EchoForge hero" width="100%" />

# EchoForge

![Jankurai score](./assets/readme/studio/jankurai-score.svg)

EchoForge is a strict-open, radar-first, GPU-native synthetic sensing foundry. It publishes public-proxy object signatures, uncertainty-scored radar artifacts, and reproducible validation evidence so downstream work can be inspected, rerun, and compared without claiming measured truth or proprietary-equivalent sensor behavior.

The claim boundary stays narrow: public-source priors only, explicit uncertainty, hard negatives treated as robustness work, and no classified, vendor-private, or exact field-performance claims.

## Studio Preview

![EchoForge Studio demo](./assets/readme/studio/studio-demo.gif)

EchoForge Studio is a same-origin Rust + Vite interface for live radar playback, Monte Carlo campaign setup, run archive/restore/duplicate workflows, artifact review, and headless API usage. The UI keeps source cards, validation tier, uncertainty language, seed, and scenario hash visible before export.

| Command Center | Live Radar | Monte Carlo Builder |
| --- | --- | --- |
| ![Command Center](./assets/readme/studio/command-center.png) | ![Live Radar](./assets/readme/studio/live-radar.png) | ![Monte Carlo Builder](./assets/readme/studio/monte-carlo-builder.png) |

README media is tracked under [assets/readme/studio](./assets/readme/studio/) with a deterministic capture manifest at [assets/readme/studio/manifest.json](./assets/readme/studio/manifest.json). Regenerate after a Studio UI change with the same lane used by the post-merge sync:

```bash
rtk bash ops/run-lane.sh studio-sync
```

## Value

- Public-proxy object and scenario packs that are readable as source artifacts, not hidden datasets.
- Detector realism surfaces that separate detection, tracking, classification, and fusion.
- Hard-negative and clutter fixtures that exercise robustness instead of optimizing evasion.
- Validation reports and smoke gates that keep generated outputs reproducible and reviewable.

## What's Included

- Source-pack fixtures: public-proxy dossiers, object cards, physics dossiers, and pack manifests.
- Detector surfaces: detector family matrices, detector cards, detector graphs, and sensor cards.
- Scenario fixtures: public-proxy scenes, airspace-object libraries, and weather profiles.
- Validation surfaces: benchmark reports, phase metrics, and smoke outputs that stay out of Git.

## Sources

- Public-proxy dossier: [object-packs/public-proxy/source_dossier.yaml](./object-packs/public-proxy/source_dossier.yaml), [object-packs/public-proxy/object_card.yaml](./object-packs/public-proxy/object_card.yaml), [object-packs/public-proxy/physics_dossier.md](./object-packs/public-proxy/physics_dossier.md)
- Public-proxy pack manifest: [object-packs/public-proxy/pack.manifest.json](./object-packs/public-proxy/pack.manifest.json)
- Radar platform pack: [object-packs/radar-platforms-v1/pack.manifest.json](./object-packs/radar-platforms-v1/pack.manifest.json), [object-packs/radar-platforms-v1/saab_giraffe_1x.yaml](./object-packs/radar-platforms-v1/saab_giraffe_1x.yaml)
- Hard-negative pack: [object-packs/hard-negatives/pack.manifest.json](./object-packs/hard-negatives/pack.manifest.json), [object-packs/hard-negatives/birds.yaml](./object-packs/hard-negatives/birds.yaml), [object-packs/hard-negatives/rain_cell.yaml](./object-packs/hard-negatives/rain_cell.yaml), [object-packs/hard-negatives/wind_turbine_large.yaml](./object-packs/hard-negatives/wind_turbine_large.yaml)
- Airspace-object library: [configs/monte-carlo/airspace-objects.json](./configs/monte-carlo/airspace-objects.json)
- Scenario fixtures: [scenarios/public-proxy-clutter/scenario.yaml](./scenarios/public-proxy-clutter/scenario.yaml), [configs/scenarios/uae-coastal-surveillance.json](./configs/scenarios/uae-coastal-surveillance.json), [tests/schemas/scenario.sample.json](./tests/schemas/scenario.sample.json), [tests/schemas/weather_profile.sample.json](./tests/schemas/weather_profile.sample.json)

## Detectors

- Detector family matrix: [detection/detector_osint_family_matrix.py](./detection/detector_osint_family_matrix.py), [detection/reports/detector_osint_family_matrix.md](./detection/reports/detector_osint_family_matrix.md)
- Detector card: [schemas/detector_card.schema.json](./schemas/detector_card.schema.json), [tests/schemas/detector_card.sample.json](./tests/schemas/detector_card.sample.json)
- Detector graph: [schemas/detector_graph.schema.json](./schemas/detector_graph.schema.json), [tests/schemas/detector_graph.sample.json](./tests/schemas/detector_graph.sample.json)
- Radar platform card: [schemas/radar_platform_card.schema.json](./schemas/radar_platform_card.schema.json), [tests/schemas/radar_platform_card.sample.json](./tests/schemas/radar_platform_card.sample.json)
- EO/IR card: [schemas/eo_ir_sensor_card.schema.json](./schemas/eo_ir_sensor_card.schema.json), [tests/schemas/eo_ir_sensor_card.sample.json](./tests/schemas/eo_ir_sensor_card.sample.json)
- Acoustic card: [schemas/acoustic_sensor_card.schema.json](./schemas/acoustic_sensor_card.schema.json), [tests/schemas/acoustic_sensor_card.sample.json](./tests/schemas/acoustic_sensor_card.sample.json)
- Passive RF card: [schemas/passive_rf_sensor_card.schema.json](./schemas/passive_rf_sensor_card.schema.json), [tests/schemas/passive_rf_sensor_card.sample.json](./tests/schemas/passive_rf_sensor_card.sample.json)
- Supporting reports: [detection/reports/phase_pd_pfa.md](./detection/reports/phase_pd_pfa.md), [detection/reports/auc_table.md](./detection/reports/auc_table.md), [detection/reports/track_lifecycle_baseline.md](./detection/reports/track_lifecycle_baseline.md), [detection/reports/benchmark_regeneration_report.md](./detection/reports/benchmark_regeneration_report.md), [detection/reports/passive_rf_eoir_passive_radar_backlog.md](./detection/reports/passive_rf_eoir_passive_radar_backlog.md)

## Modeling Results

The `runit` main run is the benchmark of record here: a 10,000-scenario-group,
30,000-phase-record Shahed-136/Geran-2 public-proxy example with all detector
branches plus the layered fusion branch. It writes generated streams and
reports under `outputs/`; see [docs/main_run.md](./docs/main_run.md) for the
full split, leakage, and claim-boundary protocol.

```bash
rtk python3 detection/generate_main_run.py --out-root outputs/training-data/runit-shahed136-main-run-v1 --scenario-groups 10000 --positive-groups 250 --seed 202605210136 --force
rtk python3 detection/run_main_run_detectors.py --data-root outputs/training-data/runit-shahed136-main-run-v1 --out-root outputs/detection/runit-shahed136-main-run-v1 --folds 5 --seed 202605210136 --force
```

Full-run holdout performance, all phases combined. Thresholds are selected on
train/CV only, then applied to the blind holdout.

| method | holdout ROC AUC | holdout AP | accuracy | precision | recall | FPR | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| [`layered_fusion_c2`](./detection/main_run_detectors.py) | 0.914031 | 0.254743 | 0.966222 | 0.344262 | 0.368421 | 0.018240 | 0.355932 |
| [`tabular_ml_baseline`](./detection/main_run_detectors.py) | 0.908963 | 0.227212 | 0.955333 | 0.251429 | 0.385965 | 0.029868 | 0.304498 |
| [`sequence_ml_proxy`](./detection/main_run_detectors.py) | 0.898339 | 0.201708 | 0.932667 | 0.188119 | 0.500000 | 0.056088 | 0.273381 |
| [`tactical_s_band_aesa`](./detection/main_run_detectors.py) | 0.886095 | 0.148732 | 0.923333 | 0.174648 | 0.543860 | 0.066803 | 0.264392 |
| [`gbad_3d4d_cueing`](./detection/main_run_detectors.py) | 0.880495 | 0.129558 | 0.937556 | 0.180077 | 0.412281 | 0.048792 | 0.250667 |
| [`high_resolution_xku_cuas`](./detection/main_run_detectors.py) | 0.880185 | 0.221105 | 0.966000 | 0.327434 | 0.324561 | 0.017328 | 0.325991 |
| [`distributed_acoustic_cue`](./detection/main_run_detectors.py) | 0.820985 | 0.148669 | 0.951778 | 0.176101 | 0.245614 | 0.029868 | 0.205128 |

The complete generated table is `outputs/detection/runit-shahed136-main-run-v1/performance_metrics.csv`;
the JSON summary is `outputs/detection/runit-shahed136-main-run-v1/performance_summary.json`.

Advanced full-stream evolution ran against the same `outputs/training-data/runit-shahed136-main-run-v1`
corpus and selected a locked meta-fusion winner after train/CV-only search.

| selected candidate | train/CV rank | train/CV AP | train/CV ROC AUC | holdout ROC AUC | holdout AP | accuracy | precision | recall | FPR | F1 | gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `meta_fusion.top8.v2_aggressive.geodesic_odds` | 1 | 0.957622026 | 0.997433776 | 0.996928025 | 0.942983898 | 0.993556 | 0.947368 | 0.789474 | 0.001140 | 0.861244 | pass |

The advanced-evolution outputs live under `outputs/detection/runit-shahed136-main-run-v1-advanced-evolution/`.

Synthetic public-proxy benchmark evidence only; these numbers are not measured
truth or field-performance claims.

## Scenarios and Weather

- Public-proxy clutter baseline: [scenarios/public-proxy-clutter/scenario.yaml](./scenarios/public-proxy-clutter/scenario.yaml)
- Scenario fixture: [tests/schemas/scenario.sample.json](./tests/schemas/scenario.sample.json)
- Weather fixture: [tests/schemas/weather_profile.sample.json](./tests/schemas/weather_profile.sample.json)
- Scenario config: [configs/scenarios/uae-coastal-surveillance.json](./configs/scenarios/uae-coastal-surveillance.json)

## How to Run

```bash
rtk just fast
rtk just demo
rtk env HOST=127.0.0.1 PORT=8080 cargo run -p echoforge-studio --locked
rtk npm run web:build
rtk npm run web:smoke
rtk python3 detection/generate_ml_training.py --out-root outputs/training-data/shahed136-public-proxy-ml-training-smoke --scenario-groups 24 --seed 136 --scale-name smoke --max-time-s 150 --force
rtk bash detection/run_all.sh --smoke
```

Generated datasets, solver outputs, and benchmark artifacts stay under `outputs/` and out of Git.

### Studio Web And Headless

```bash
rtk npm install
rtk npm run web:build
rtk env HOST=127.0.0.1 PORT=8080 cargo run -p echoforge-studio --locked
rtk curl http://127.0.0.1:8080/api/runs
```

Current same-origin Studio endpoints:

- `GET /api/runs`, `GET /api/runs/{id}`, `GET /api/runs/{id}/artifacts`
- `POST /api/runs`, `POST /api/runs/{id}/replay`, `POST /api/runs/{id}/archive`, `POST /api/runs/{id}/restore`, `POST /api/runs/{id}/duplicate`
- `GET /api/jobs`, `POST /api/jobs`, `POST /api/jobs/{id}/cancel`
- `GET /api/contracts`, `GET /api/catalog`, `GET /api/validation/latest`
- `WS /ws/radar`: JSON lifecycle/validation/artifact/backpressure/session frames plus quantized binary scan frames

Example Monte Carlo request:

```bash
rtk curl -X POST http://127.0.0.1:8080/api/runs \
  -H 'content-type: application/json' \
  -d '{"scenario_id":"coastal-clutter","source_pack":"public-proxy-v1","object_pack":"airspace-objects-v1","hard_negatives":["bird_flock_dense","rain_cell"],"weather_profile":"uae_coastal_summer","detector_pipeline":"physics_cfar_track_fusion_v1","seed":2026052101,"run_count":12,"workers":4,"max_concurrent":2,"smoke":true,"validation_target":"V1 public-proxy"}'
```

## Where to Start

- Repo policy and agent instructions: [AGENTS.md](./AGENTS.md)
- Architecture and stack decisions: [docs/architecture.md](./docs/architecture.md)
- Proof lanes, cost budgets, kill-switches: [docs/testing.md](./docs/testing.md)
- Audit rubric and boundary guidance: [docs/audit-rubric.md](./docs/audit-rubric.md)
- Validation tier definitions: [docs/validation-tiers.md](./docs/validation-tiers.md)
- Bootstrap and smoke recipes: [Justfile](./Justfile)
- Ownership and generated-zone maps: [agent/owner-map.json](./agent/owner-map.json), [agent/test-map.json](./agent/test-map.json), [agent/generated-zones.toml](./agent/generated-zones.toml)
- Boundary slice definitions: [agent/boundaries.toml](./agent/boundaries.toml)
- Schema fixture guide: [tests/schemas/README.md](./tests/schemas/README.md)
- Validation surface: [detection/reports/detector_osint_family_matrix.md](./detection/reports/detector_osint_family_matrix.md), [detection/reports/track_lifecycle_baseline.md](./detection/reports/track_lifecycle_baseline.md)
- Public-proxy pack entrypoint: [object-packs/public-proxy/pack.manifest.json](./object-packs/public-proxy/pack.manifest.json)
