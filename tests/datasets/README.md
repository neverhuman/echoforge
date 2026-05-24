# Dataset Test Fixtures

These fixtures are intentionally small and neutral.

- `split_policy.json` exercises the deterministic split policy.
- `leakage_input.json` contains a deliberate cross-split duplicate to verify leakage detection.
- `benchmark_report.json` mirrors the benchmark placeholder shape.

The crates under `crates/echoforge-world` and `crates/echoforge-dataset` read these fixtures in unit tests.

