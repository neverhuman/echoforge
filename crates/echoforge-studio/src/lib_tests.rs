use super::*;
use axum::body::Body;
use std::fs;
use tempfile::tempdir;
use tower::ServiceExt;

fn test_config(web_dist: PathBuf) -> StudioConfig {
    let workspace_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .expect("workspace root");
    StudioConfig {
        host: "127.0.0.1".to_string(),
        port: 0,
        service_name: DEFAULT_SERVICE_NAME.to_string(),
        public_base_url: DEFAULT_PUBLIC_BASE_URL.to_string(),
        catalog_path: workspace_root.join("contracts/schema_catalog.json"),
        bundle_path: workspace_root.join("tests/science/fixtures/bundles/v1_pass"),
        web_dist,
        sim: crate::stream::control::SimSettings {
            autostart: false,
            ..crate::stream::control::SimSettings::default()
        },
    }
}

#[tokio::test]
async fn router_serves_live_contracts() {
    let dist = tempdir().expect("tempdir");
    fs::write(
        dist.path().join("index.html"),
        "<!doctype html><div id=\"app\"></div>",
    )
    .expect("index");
    let config = test_config(dist.path().to_path_buf());
    let state = Arc::new(StudioState::load(&config).expect("state"));
    let app = build_router(state, config.web_dist.clone());

    let health = app
        .clone()
        .oneshot(
            axum::http::Request::builder()
                .uri("/healthz")
                .body(Body::empty())
                .expect("request"),
        )
        .await
        .expect("response");
    assert_eq!(health.status(), axum::http::StatusCode::OK);
    let health_body = axum::body::to_bytes(health.into_body(), usize::MAX)
        .await
        .expect("health body");
    let health_json: HealthResponse = serde_json::from_slice(&health_body).expect("health json");
    assert_eq!(health_json.service, DEFAULT_SERVICE_NAME);
    assert_eq!(health_json.validation_status, "pass");
    assert_eq!(health_json.schema_count, 12);

    let validation = app
        .clone()
        .oneshot(
            axum::http::Request::builder()
                .uri("/api/validation/latest")
                .body(Body::empty())
                .expect("request"),
        )
        .await
        .expect("response");
    assert_eq!(validation.status(), axum::http::StatusCode::OK);
    let validation_body = axum::body::to_bytes(validation.into_body(), usize::MAX)
        .await
        .expect("validation body");
    let validation_json: ValidateReport =
        serde_json::from_slice(&validation_body).expect("validation json");
    assert_eq!(validation_json.overall_status, "pass");
    assert_eq!(validation_json.tier.as_str(), "V1");

    let contracts = app
        .oneshot(
            axum::http::Request::builder()
                .uri("/api/contracts")
                .body(Body::empty())
                .expect("request"),
        )
        .await
        .expect("response");
    assert_eq!(contracts.status(), axum::http::StatusCode::OK);
}
