use axum::body::Body;
use axum::http::Request;
use echoforge_studio::stream::control::SimSettings;
use echoforge_studio::{
    build_router, StudioConfig, StudioState, DEFAULT_BUNDLE_PATH, DEFAULT_CATALOG_PATH,
    DEFAULT_PUBLIC_BASE_URL, DEFAULT_SERVICE_NAME,
};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;
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
        catalog_path: workspace_root.join(DEFAULT_CATALOG_PATH),
        bundle_path: workspace_root.join(DEFAULT_BUNDLE_PATH),
        web_dist,
        sim: SimSettings {
            autostart: false,
            ..SimSettings::default()
        },
    }
}

#[tokio::test]
async fn jobs_endpoints_create_and_return_results() {
    let dist = tempfile::tempdir().expect("tempdir");
    std::fs::write(
        dist.path().join("index.html"),
        "<!doctype html><div id=\"app\"></div>",
    )
    .expect("index");
    let config = test_config(dist.path().to_path_buf());
    let state = Arc::new(StudioState::load(&config).expect("state"));
    let app = build_router(state, config.web_dist.clone());

    let create = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/jobs")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"job_type":"ml_processing","selection":"pipeline","pipeline_id":"physics_cfar_track_fusion_v1","data_root":"outputs/training-data/best-final-scenario-v1","out_root":"outputs/ml-pipelines","workers_per_pipeline":20,"max_concurrent":3,"seed":20260520390001,"smoke":true,"validation_tier":"evidence_ladder_v1"}"#,
                ))
                .expect("request"),
        )
        .await
        .expect("response");
    assert_eq!(create.status(), axum::http::StatusCode::CREATED);
    let body = axum::body::to_bytes(create.into_body(), usize::MAX)
        .await
        .expect("job body");
    let job: serde_json::Value = serde_json::from_slice(&body).expect("job json");
    let job_id = job["job_id"].as_str().expect("job id");
    assert_eq!(job["status"], "queued");

    for _ in 0..10 {
        let job_response = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri(format!("/api/jobs/{job_id}"))
                    .body(Body::empty())
                    .expect("request"),
            )
            .await
            .expect("response");
        let job_body = axum::body::to_bytes(job_response.into_body(), usize::MAX)
            .await
            .expect("job body");
        let job_json: serde_json::Value = serde_json::from_slice(&job_body).expect("job json");
        if job_json["status"] == "completed" {
            break;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }

    let results = app
        .clone()
        .oneshot(
            Request::builder()
                .uri(format!("/api/jobs/{job_id}/results"))
                .body(Body::empty())
                .expect("request"),
        )
        .await
        .expect("response");
    assert_eq!(results.status(), axum::http::StatusCode::OK);
    let body = axum::body::to_bytes(results.into_body(), usize::MAX)
        .await
        .expect("results body");
    let results_json: serde_json::Value = serde_json::from_slice(&body).expect("results json");
    assert_eq!(
        results_json["primary_pipeline_id"],
        "physics_cfar_track_fusion_v1"
    );

    let artifacts = app
        .oneshot(
            Request::builder()
                .uri(format!("/api/jobs/{job_id}/artifacts"))
                .body(Body::empty())
                .expect("request"),
        )
        .await
        .expect("response");
    assert_eq!(artifacts.status(), axum::http::StatusCode::OK);
}

#[tokio::test]
async fn jobs_endpoints_cancel_running_job() {
    let dist = tempfile::tempdir().expect("tempdir");
    std::fs::write(
        dist.path().join("index.html"),
        "<!doctype html><div id=\"app\"></div>",
    )
    .expect("index");
    let config = test_config(dist.path().to_path_buf());
    let state = Arc::new(StudioState::load(&config).expect("state"));
    let app = build_router(state, config.web_dist.clone());

    let create = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/jobs")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"job_type":"ml_processing","selection":"suite","suite_id":"evidence-ladder-v1","data_root":"outputs/training-data/best-final-scenario-v1","out_root":"outputs/ml-pipelines","workers_per_pipeline":20,"max_concurrent":3,"seed":20260520390001,"smoke":true,"validation_tier":"evidence_ladder_v1"}"#,
                ))
                .expect("request"),
        )
        .await
        .expect("response");
    assert_eq!(create.status(), axum::http::StatusCode::CREATED);
    let body = axum::body::to_bytes(create.into_body(), usize::MAX)
        .await
        .expect("job body");
    let job: serde_json::Value = serde_json::from_slice(&body).expect("job json");
    let job_id = job["job_id"].as_str().expect("job id");

    let cancel = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri(format!("/api/jobs/{job_id}/cancel"))
                .body(Body::empty())
                .expect("request"),
        )
        .await
        .expect("response");
    assert_eq!(cancel.status(), axum::http::StatusCode::OK);
}

#[tokio::test]
async fn runs_endpoints_list_and_download() {
    let dist = tempfile::tempdir().expect("tempdir");
    std::fs::write(
        dist.path().join("index.html"),
        "<!doctype html><div id=\"app\"></div>",
    )
    .expect("index");
    let config = test_config(dist.path().to_path_buf());
    let state = Arc::new(StudioState::load(&config).expect("state"));
    let app = build_router(state, config.web_dist.clone());

    let runs = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/api/runs")
                .body(Body::empty())
                .expect("request"),
        )
        .await
        .expect("response");
    assert_eq!(runs.status(), axum::http::StatusCode::OK);

    let run = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/api/runs/run-shahed-ingress-00")
                .body(Body::empty())
                .expect("request"),
        )
        .await
        .expect("response");
    assert_eq!(run.status(), axum::http::StatusCode::OK);

    let download = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/api/runs/run-shahed-ingress-00/download?kind=bundle")
                .body(Body::empty())
                .expect("request"),
        )
        .await
        .expect("response");
    assert_eq!(download.status(), axum::http::StatusCode::OK);
    let disposition = download
        .headers()
        .get(axum::http::header::CONTENT_DISPOSITION)
        .and_then(|value| value.to_str().ok())
        .unwrap_or("");
    assert!(disposition.contains("run-shahed-ingress-00-bundle.json"));
}

#[tokio::test]
async fn runs_endpoints_archive_restore_duplicate_and_queue_monte_carlo() {
    let dist = tempfile::tempdir().expect("tempdir");
    std::fs::write(
        dist.path().join("index.html"),
        "<!doctype html><div id=\"app\"></div>",
    )
    .expect("index");
    let config = test_config(dist.path().to_path_buf());
    let state = Arc::new(StudioState::load(&config).expect("state"));
    let app = build_router(state, config.web_dist.clone());

    let archived = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/runs/run-shahed-ingress-00/archive")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"reason":"operator review complete"}"#))
                .expect("request"),
        )
        .await
        .expect("response");
    assert_eq!(archived.status(), axum::http::StatusCode::OK);
    let body = axum::body::to_bytes(archived.into_body(), usize::MAX)
        .await
        .expect("archive body");
    let archived_json: serde_json::Value = serde_json::from_slice(&body).expect("archive json");
    assert_eq!(archived_json["status"], "archived");

    let restored = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/runs/run-shahed-ingress-00/restore")
                .body(Body::empty())
                .expect("request"),
        )
        .await
        .expect("response");
    assert_eq!(restored.status(), axum::http::StatusCode::OK);
    let body = axum::body::to_bytes(restored.into_body(), usize::MAX)
        .await
        .expect("restore body");
    let restored_json: serde_json::Value = serde_json::from_slice(&body).expect("restore json");
    assert_eq!(restored_json["status"], "validated");

    let duplicate = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/runs/run-shahed-ingress-00/duplicate")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"seed":424242,"mode":"monte_carlo"}"#))
                .expect("request"),
        )
        .await
        .expect("response");
    assert_eq!(duplicate.status(), axum::http::StatusCode::CREATED);
    let body = axum::body::to_bytes(duplicate.into_body(), usize::MAX)
        .await
        .expect("duplicate body");
    let duplicate_json: serde_json::Value = serde_json::from_slice(&body).expect("duplicate json");
    assert_eq!(duplicate_json["status"], "queued");
    assert_eq!(duplicate_json["config"]["seed"], 424242);
    assert_eq!(duplicate_json["config"]["mode"], "monte_carlo");

    let monte_carlo = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/runs")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"scenario_id":"coastal-clutter","source_pack":"public-proxy-v1","object_pack":"airspace-objects-v1","hard_negatives":["birds","rain_cell"],"weather_profile":"uae_coastal_summer","detector_pipeline":"physics_cfar_track_fusion_v1","seed":2026052101,"run_count":12,"workers":4,"max_concurrent":2,"smoke":true,"validation_target":"V1 public-proxy"}"#,
                ))
                .expect("request"),
        )
        .await
        .expect("response");
    assert_eq!(monte_carlo.status(), axum::http::StatusCode::CREATED);
    let body = axum::body::to_bytes(monte_carlo.into_body(), usize::MAX)
        .await
        .expect("monte carlo body");
    let monte_carlo_json: serde_json::Value =
        serde_json::from_slice(&body).expect("monte carlo json");
    assert_eq!(monte_carlo_json["status"], "queued");
    assert_eq!(monte_carlo_json["config"]["mode"], "monte_carlo");
    assert_eq!(monte_carlo_json["validation"]["tier"], "V1 public-proxy");

    let summary = app
        .oneshot(
            Request::builder()
                .uri("/api/runs/queue/summary")
                .body(Body::empty())
                .expect("request"),
        )
        .await
        .expect("response");
    assert_eq!(summary.status(), axum::http::StatusCode::OK);
    let body = axum::body::to_bytes(summary.into_body(), usize::MAX)
        .await
        .expect("summary body");
    let summary_json: serde_json::Value = serde_json::from_slice(&body).expect("summary json");
    assert!(summary_json["queued"].as_u64().unwrap_or(0) >= 2);
}
