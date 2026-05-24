# Audit Rubric

## Observability (HLT-017-OPAQUE-OBSERVABILITY)

### Markers (evidence_kind='repository-scan')
1. Observability libraries found (e.g., tracing, metrics, opentelemetry)
2. ops/observability directory present
3. Repair receipts found
4. Agent-friendly exception pattern found
5. Repair receipt guidance is documented
6. OpenTelemetry (OTEL) configuration in main.rs

### Evidence Requirements
- Libraries must be actively used in the codebase
- Directory must contain meaningful observability configurations
- Repair receipts must include actionable repair steps
- Exception patterns must be documented and used in error handling
- OTEL configuration must initialize tracing and metrics exporters

## Build Speed (HLT-018-PERF-CONCURRENCY-DRIFT)

### Markers (evidence_kind='repository-scan')
1. Build acceleration found (e.g., sccache, ccache, cargo-chef)
2. Targeted test/build commands found
3. Locked dependency graph
4. CI cache hint found
5. Explicit cache marker plus narrow per-package target found
6. sccache configured for Rust builds in CI

### Evidence Requirements
- Build acceleration tools must be configured and used in CI
- Commands must target specific packages or tests
- Dependency graph must be locked (Cargo.lock, package-lock.json)
- Cache hints must be present in CI configuration
- sccache must be configured with appropriate cache size and enabled in CI