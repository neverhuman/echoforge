//! Studio job lifecycle for ML processing requests.

use std::time::Duration;

use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};

use crate::StudioState;

mod defaults;
mod results;
mod store;
pub mod types;

pub use store::JobStore;
pub use types::{JobComposeRequest, JobSummary};

use results::{empty_results, execute_job};

pub fn jobs_router() -> Router<std::sync::Arc<StudioState>> {
    Router::new()
        .route("/api/jobs", get(list_jobs).post(create_job))
        .route("/api/jobs/defaults", get(get_defaults))
        .route("/api/jobs/{id}", get(get_job))
        .route("/api/jobs/{id}/cancel", post(cancel_job))
        .route("/api/jobs/{id}/artifacts", get(list_artifacts))
        .route("/api/jobs/{id}/results", get(get_results))
}

async fn get_defaults() -> impl IntoResponse {
    Json(defaults::job_defaults_response())
}

async fn list_jobs(State(state): State<std::sync::Arc<StudioState>>) -> impl IntoResponse {
    Json(state.job_store.list())
}

async fn create_job(
    State(state): State<std::sync::Arc<StudioState>>,
    Json(request): Json<JobComposeRequest>,
) -> impl IntoResponse {
    let job = state.job_store.create(request.clone());
    let store = state.job_store.clone();
    let repo_root = state.repo_root.clone();
    let job_id = job.job_id.clone();
    tokio::spawn(async move {
        launch_job(store, job_id, request, repo_root).await;
    });
    (StatusCode::CREATED, Json(job))
}

async fn get_job(
    State(state): State<std::sync::Arc<StudioState>>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    match state.job_store.get(&id) {
        Some(job) => (StatusCode::OK, Json(job)).into_response(),
        None => not_found("job_not_found", &id),
    }
}

async fn cancel_job(
    State(state): State<std::sync::Arc<StudioState>>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    match state.job_store.cancel(&id) {
        Some(job) => (StatusCode::OK, Json(job)).into_response(),
        None => not_found("job_not_found", &id),
    }
}

async fn list_artifacts(
    State(state): State<std::sync::Arc<StudioState>>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    match state.job_store.get(&id) {
        Some(job) => (StatusCode::OK, Json(job.artifacts)).into_response(),
        None => not_found("job_not_found", &id),
    }
}

async fn get_results(
    State(state): State<std::sync::Arc<StudioState>>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    match state.job_store.get(&id) {
        Some(job) => (StatusCode::OK, Json(job.results)).into_response(),
        None => not_found("job_not_found", &id),
    }
}

async fn launch_job(
    store: JobStore,
    job_id: String,
    request: JobComposeRequest,
    repo_root: std::path::PathBuf,
) {
    store.update(&job_id, |job| {
        job.status = "running".to_string();
        job.progress_percent = 10;
        job.message = "validating ML pipeline request".to_string();
    });

    tokio::time::sleep(Duration::from_millis(40)).await;
    if is_cancelled(&store, &job_id) {
        return;
    }

    let request_for_exec = request.clone();
    let execution =
        tokio::task::spawn_blocking(move || execute_job(&request_for_exec, &repo_root)).await;
    let outcome = match execution {
        Ok(result) => result,
        Err(err) => Err(format!("job task join error: {err}")),
    };

    if is_cancelled(&store, &job_id) {
        return;
    }

    match outcome {
        Ok(payload) => {
            store.update(&job_id, |job| {
                job.status = "completed".to_string();
                job.progress_percent = 100;
                job.message = payload.message.clone();
                job.record_count = payload.record_count;
                job.output_dir = payload.output_dir.clone();
                job.artifacts = payload.artifacts.clone();
                job.results = payload.results.clone();
            });
        }
        Err(message) => {
            store.update(&job_id, |job| {
                job.status = "failed".to_string();
                job.progress_percent = 100;
                job.message = message.clone();
                job.artifacts.clear();
                job.results = empty_results(&request, "failed", &message);
            });
        }
    }
}

fn is_cancelled(store: &JobStore, job_id: &str) -> bool {
    store
        .get(job_id)
        .map(|job| job.status == "cancelled")
        .unwrap_or(true)
}

fn not_found(code: &str, id: &str) -> Response {
    (
        StatusCode::NOT_FOUND,
        Json(serde_json::json!({
            "error": code,
            "id": id,
        })),
    )
        .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_request_uses_evidence_ladder_validation_tier() {
        let request = JobComposeRequest::default();
        assert_eq!(request.job_type, "ml_processing");
        assert_eq!(request.selection, "pipeline");
        assert_eq!(request.validation_tier, "evidence_ladder_v1");
        assert_eq!(request.suite_id, defaults::DEFAULT_SUITE_ID);
    }
}
