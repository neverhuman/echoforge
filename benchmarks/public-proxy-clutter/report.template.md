# Public Proxy Clutter Benchmark Report Template

schema_ref: schemas/benchmark_report.schema.json
dataset_card_schema_ref: schemas/dataset_card.schema.json

## Required Sections
- summary
- dataset_scope
- split_policy
- leakage_checks
- benchmark_metrics
- known_limitations
- artifact_manifest

## Required Artifacts
- dataset_card.md
- leakage_report.json
- split_manifest.json

## Known Limitations
- Proxy data is not measured truth.
- Geometry, material, and source assumptions must remain explicit.
- Hard-negative coverage is incomplete until the pack grows.

## Suggested Closing Line
This benchmark uses public-proxy artifacts with explicit uncertainty and leakage checks.

