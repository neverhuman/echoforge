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

`echoforge-studio` initializes both tracing and metrics exporters at startup via `init_otel()` in `crates/echoforge-studio/src/main.rs`. When `OTEL_EXPORTER_OTLP_ENDPOINT` is set, spans and metrics are forwarded to an OTLP collector via gRPC using `opentelemetry-otlp` with tonic transport:

```rust
use opentelemetry::global;
use opentelemetry_otlp::{SpanExporter, MetricExporter, WithExportConfig};
use opentelemetry_sdk::{metrics::SdkMeterProvider, trace::SdkTracerProvider, Resource};
use tracing_opentelemetry::OpenTelemetryLayer;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

fn init_otel() {
    // When OTEL_EXPORTER_OTLP_ENDPOINT is set, configure OTLP gRPC exporters
    if let Ok(endpoint) = std::env::var("OTEL_EXPORTER_OTLP_ENDPOINT") {
        let resource = Resource::builder()
            .with_service_name(
                std::env::var("OTEL_SERVICE_NAME")
                    .unwrap_or_else(|_| "echoforge-studio".to_string()),
            )
            .build();
        let span_exporter = SpanExporter::builder()
            .with_tonic()
            .with_endpoint(&endpoint)
            .build()
            .expect("failed to build OTLP span exporter");
        let tracer_provider = SdkTracerProvider::builder()
            .with_resource(resource.clone())
            .with_batch_exporter(span_exporter)
            .build();
        global::set_tracer_provider(tracer_provider);
        let metric_exporter = MetricExporter::builder()
            .with_tonic()
            .with_endpoint(&endpoint)
            .build()
            .expect("failed to build OTLP metric exporter");
        let meter_provider = SdkMeterProvider::builder()
            .with_resource(resource)
            .with_periodic_exporter(metric_exporter)
            .build();
        global::set_meter_provider(meter_provider);
    } else {
        let meter_provider = SdkMeterProvider::builder().build();
        global::set_meter_provider(meter_provider);
    }
    let otel_layer = OpenTelemetryLayer::new(global::tracer("echoforge-studio"));
    tracing_subscriber::registry()
        .with(EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")))
        .with(tracing_subscriber::fmt::layer().with_target(true))
        .with(otel_layer)
        .init();
}
```

Env vars: `OTEL_EXPORTER_OTLP_ENDPOINT`, `OTEL_SERVICE_NAME`, `OTEL_RESOURCE_ATTRIBUTES`.

## Diagnostic Shaping

EchoForge uses structured tracing spans with explicit field shaping to ensure diagnostic signals are machine-readable and agent-navigable.

Key `#[tracing::instrument]` usage:
- `echoforge_studio::serve_from_env` — root entry point span, no args (avoids config secret leakage)
- `echoforge_studio::serve` — captures `host` and `port` as span fields for request tracing

Structured field patterns in spans use `%` (Display) and `?` (Debug) format specifiers:
```rust
tracing::info!(otel_endpoint = %endpoint, "OTLP tracing and metrics exporters initialized");
```

**Request ID propagation**: Each campaign run is assigned a `campaign_request_id` (the request id flows through all dataset pipeline spans). Consumers can filter traces by request id to reconstruct the full scene-generation → validation chain.

**Correlation ID**: A `correlation id` is embedded in detector-event records and links detection events back to the originating campaign batch. The correlation id appears in `detector_events.json` as `campaign_request_id` and in distributed spans as the `request_id` span field.

## Artifact Locations
- Score history: `target/jankurai/score-history.jsonl`
- Security evidence: `target/jankurai/security/evidence.json`
- Repair queue: `target/jankurai/repair-queue.jsonl`

## Error Catalog
See `error-catalog.md` for structured error codes and recovery steps.
