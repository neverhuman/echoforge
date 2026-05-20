use opentelemetry::global;
use opentelemetry_otlp::{MetricExporter, SpanExporter, WithExportConfig};
use opentelemetry_sdk::{metrics::SdkMeterProvider, trace::SdkTracerProvider, Resource};
use tracing_opentelemetry::OpenTelemetryLayer;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

// Initializes OpenTelemetry (OTEL) tracing and metrics exporters.
// When OTEL_EXPORTER_OTLP_ENDPOINT is set, spans and metrics are forwarded
// to an OTLP collector via gRPC; otherwise a no-op in-process provider is used.
// Env vars: OTEL_EXPORTER_OTLP_ENDPOINT, OTEL_SERVICE_NAME, OTEL_RESOURCE_ATTRIBUTES.
//
// Span context: each request id and correlation id from the dataset campaign pipeline
// propagates through child spans so collectors can filter by request id to reconstruct
// a full scene-generation → validation chain.
fn init_otel() {
    let filter = match EnvFilter::try_from_default_env() {
        Ok(f) => f,
        Err(_) => EnvFilter::new("info"),
    };

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

        tracing::info!(otel_endpoint = %endpoint, "OTLP tracing and metrics exporters initialized");
    } else {
        let meter_provider = SdkMeterProvider::builder().build();
        global::set_meter_provider(meter_provider);
    }

    let otel_layer = OpenTelemetryLayer::new(global::tracer("echoforge-studio"));
    tracing_subscriber::registry()
        .with(filter)
        .with(tracing_subscriber::fmt::layer().with_target(true))
        .with(otel_layer)
        .init();
}

#[tokio::main(flavor = "multi_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    init_otel();
    echoforge_studio::serve_from_env().await?;
    Ok(())
}
