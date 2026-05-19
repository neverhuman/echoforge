# Observability

EchoForge emits structured traces and metrics via OpenTelemetry (OTEL). This document covers logs, metrics, traces, audit events, SLOs, and evidence retention.

## Telemetry Initialization

`echoforge-studio` calls `init_otel()` at startup (`crates/echoforge-studio/src/main.rs`). When `OTEL_EXPORTER_OTLP_ENDPOINT` is set, spans and metrics are forwarded to an OTLP collector via gRPC (`opentelemetry-otlp` with tonic transport). Otherwise an in-process no-op provider is used.

```rust
use opentelemetry::global;
use opentelemetry_otlp::{SpanExporter, MetricExporter, WithExportConfig};
use opentelemetry_sdk::{metrics::SdkMeterProvider, trace::SdkTracerProvider, Resource};
use tracing_opentelemetry::OpenTelemetryLayer;

fn init_otel() {
    if let Ok(endpoint) = std::env::var("OTEL_EXPORTER_OTLP_ENDPOINT") {
        let resource = Resource::builder()
            .with_service_name("echoforge-studio")
            .build();
        let span_exporter = SpanExporter::builder()
            .with_tonic().with_endpoint(&endpoint).build().unwrap();
        global::set_tracer_provider(
            SdkTracerProvider::builder()
                .with_resource(resource.clone())
                .with_batch_exporter(span_exporter)
                .build()
        );
        let metric_exporter = MetricExporter::builder()
            .with_tonic().with_endpoint(&endpoint).build().unwrap();
        global::set_meter_provider(
            SdkMeterProvider::builder()
                .with_resource(resource)
                .with_periodic_exporter(metric_exporter)
                .build()
        );
    }
    tracing_subscriber::registry()
        .with(OpenTelemetryLayer::new(global::tracer("echoforge-studio")))
        .init();
}
```

## Environment Variables

| Variable | Purpose |
|---|---|
| `OTEL_EXPORTER_OTLP_ENDPOINT` | gRPC endpoint for OTLP collector (e.g. `http://localhost:4317`) |
| `OTEL_SERVICE_NAME` | Service name in trace metadata (default: `echoforge-studio`) |
| `OTEL_RESOURCE_ATTRIBUTES` | Additional resource attributes (key=value,key=value) |
| `RUST_LOG` | Tracing filter (e.g. `info`, `echoforge=debug`) |

## Logs

All crates use `tracing::{info!, warn!, error!, debug!}` for structured log emission. The `fmt` layer in `init_otel()` renders logs to stdout with target metadata. Raw `println!`/`eprintln!` is reserved for CLI user-facing output in `echoforge-cli` only.

## Metrics

The `SdkMeterProvider` is initialized globally and available via `opentelemetry::global::meter()`. Counter and histogram instruments are registered per subsystem and forwarded to the OTLP collector when configured.

## Traces

Distributed traces are captured via `tracing-opentelemetry`, which bridges `tracing` spans into OTEL spans and forwards them to the configured `SdkTracerProvider`. Service spans carry `service.name` and `service.version` resource attributes.

## Audit Events

Structured audit events (authentication, authorization decisions, pipeline completions) are emitted as `tracing::info!` spans with `audit=true` field. These are captured by the OTEL layer and forwarded alongside application traces.

## SLOs

| SLO | Target | Evidence |
|---|---|---|
| Request latency p99 | < 200ms | OTLP histogram `request_duration_ms` |
| Error rate | < 1% | OTLP counter `request_errors_total` |
| Pipeline completion | > 99% | OTLP counter `pipeline_completions_total` |

## Evidence Retention

| Artifact | Location | Retention |
|---|---|---|
| Score history | `target/jankurai/score-history.jsonl` | Git-tracked |
| Security evidence | `target/jankurai/security/evidence.json` | Git-tracked |
| Repair queue | `target/jankurai/repair-queue.jsonl` | Git-tracked |
| Repair receipts | `.agents/receipts/` | Git-tracked |
| OTLP traces/metrics | Collector (production) | 30 days |

## Error Catalog

Structured error types with recovery steps are documented in `crates/echoforge-radar/src/error.rs` and `crates/echoforge-validate/src/lib.rs`. Each error variant implements `std::error::Error` and carries a machine-readable code.

See also: `ops/observability/README.md` for observability marker evidence and OTLP wiring details.
