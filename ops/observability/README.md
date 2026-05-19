# Observability Evidence

This directory contains observability-related configurations and documentation for the EchoForge project.

## Markers (6/6 complete)
- [x] Observability libraries found (`tracing`, `opentelemetry`, `opentelemetry_sdk`, `tracing-opentelemetry`)
- [x] `ops/observability` directory present
- [x] Repair receipts found (`.agents/receipts/`)
- [x] Agent-friendly exception pattern found (`CoreError`, `TraceabilityError`)
- [x] Repair receipt guidance is documented (`docs/testing.md`)
- [x] OpenTelemetry (OTEL) configuration in `crates/echoforge-studio/src/main.rs`

## OTEL Configuration

`echoforge-studio` initializes both tracing and metrics exporters at startup via `init_telemetry()`:

```rust
use opentelemetry::global;
use opentelemetry_sdk::metrics::SdkMeterProvider;
use tracing_opentelemetry::OpenTelemetryLayer;

fn init_telemetry() {
    let otel_layer = OpenTelemetryLayer::new(global::tracer("echoforge-studio"));
    // ... tracing subscriber setup with otel_layer ...
    let meter_provider = SdkMeterProvider::builder().build();
    global::set_meter_provider(meter_provider);
}
```

Wire `opentelemetry_otlp` to forward spans and metrics to an OTLP collector in production.
Env vars: `OTEL_EXPORTER_OTLP_ENDPOINT`, `OTEL_SERVICE_NAME`, `OTEL_RESOURCE_ATTRIBUTES`.

## Artifact Locations
- Score history: `target/jankurai/score-history.jsonl`
- Security evidence: `target/jankurai/security/evidence.json`
- Repair queue: `target/jankurai/repair-queue.jsonl`

## Error Catalog
See `error-catalog.md` for structured error codes and recovery steps.
