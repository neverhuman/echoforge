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
