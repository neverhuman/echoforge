<img src="./assets/ecoforge.png" alt="EchoForge hero" width="100%" />

# EchoForge

EchoForge is a strict-open, radar-first, GPU-native synthetic sensing foundry. It publishes public-proxy object signatures, uncertainty-scored radar artifacts, and reproducible validation evidence so downstream work can be inspected, rerun, and compared without claiming measured truth or proprietary-equivalent sensor behavior.

The claim boundary stays narrow: public-source priors only, explicit uncertainty, hard negatives treated as robustness work, and no classified, vendor-private, or exact field-performance claims.

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

## Scenarios and Weather

- Public-proxy clutter baseline: [scenarios/public-proxy-clutter/scenario.yaml](./scenarios/public-proxy-clutter/scenario.yaml)
- Scenario fixture: [tests/schemas/scenario.sample.json](./tests/schemas/scenario.sample.json)
- Weather fixture: [tests/schemas/weather_profile.sample.json](./tests/schemas/weather_profile.sample.json)
- Scenario config: [configs/scenarios/uae-coastal-surveillance.json](./configs/scenarios/uae-coastal-surveillance.json)

## How to Run

```bash
rtk just fast
rtk just demo
HOST=127.0.0.1 PORT=8080 rtk cargo run -p echoforge-studio --locked
rtk npm run web:build
rtk npm run web:smoke
rtk python3 detection/generate_ml_training.py --out-root outputs/training-data/shahed136-public-proxy-ml-training-smoke --scenario-groups 24 --seed 136 --scale-name smoke --max-time-s 150 --force
rtk python3 detection/run_all.py --smoke
```

Generated datasets, solver outputs, and benchmark artifacts stay under `outputs/` and out of Git.

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
