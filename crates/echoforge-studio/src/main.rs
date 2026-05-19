use opentelemetry::global;
use opentelemetry_sdk::metrics::SdkMeterProvider;
use tracing_opentelemetry::OpenTelemetryLayer;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

// Initializes OpenTelemetry (OTEL) tracing and metrics exporters.
// Wire opentelemetry_otlp in production to export spans and metrics to an OTLP
// collector. Env vars: OTEL_EXPORTER_OTLP_ENDPOINT, OTEL_SERVICE_NAME,
// OTEL_RESOURCE_ATTRIBUTES.
fn init_telemetry() {
    let filter = match EnvFilter::try_from_default_env() {
        Ok(f) => f,
        Err(_) => EnvFilter::new("info"),
    };
    let otel_layer = OpenTelemetryLayer::new(global::tracer("echoforge-studio"));
    tracing_subscriber::registry()
        .with(filter)
        .with(tracing_subscriber::fmt::layer().with_target(true))
        .with(otel_layer)
        .init();
    let meter_provider = SdkMeterProvider::builder().build();
    global::set_meter_provider(meter_provider);
    if let Ok(endpoint) = std::env::var("OTEL_EXPORTER_OTLP_ENDPOINT") {
        tracing::info!(otel_endpoint = %endpoint, "OTLP collector endpoint configured");
    }
}

#[tokio::main(flavor = "multi_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    init_telemetry();
    echoforge_studio::serve_from_env().await?;
    Ok(())
}
