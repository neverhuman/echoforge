//! Studio job lifecycle for ML processing requests.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};

use echoforge_dataset::{
    run_pipeline, run_suite, PipelineRunRequest, PipelineRunResult, PipelineSuiteResult,
};

use crate::StudioState;

const DEFAULT_PIPELINE_ID: &str = "physics_cfar_track_fusion_v1";
const DEFAULT_SUITE_ID: &str = "evidence-ladder-v1";
const DEFAULT_DATA_ROOT: &str = "outputs/training-data/best-final-scenario-v1";
const DEFAULT_OUT_ROOT: &str = "outputs/ml-pipelines";

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct JobComposeRequest {
    #[serde(default = "default_job_type")]
    pub job_type: String,
    #[serde(default = "default_selection")]
    pub selection: String,
    #[serde(default = "default_pipeline_id")]
    pub pipeline_id: String,
    #[serde(default = "default_suite_id")]
    pub suite_id: String,
    #[serde(default = "default_data_root")]
    pub data_root: String,
    #[serde(default = "default_out_root")]
    pub out_root: String,
    #[serde(default = "default_workers_per_pipeline")]
    pub workers_per_pipeline: usize,
    #[serde(default = "default_max_concurrent")]
    pub max_concurrent: usize,
    #[serde(default = "default_seed")]
    pub seed: u64,
    #[serde(default)]
    pub smoke: bool,
    #[serde(default = "default_validation_tier")]
    pub validation_tier: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct JobArtifact {
    pub id: String,
    pub kind: String,
    pub path: String,
    pub ready: bool,
    #[serde(default)]
    pub incomplete: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MetricSlice {
    pub label: String,
    pub auc: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CalibrationBin {
    pub bin: u32,
    pub lower: f64,
    pub upper: f64,
    pub count: f64,
    pub mean_score: f64,
    pub positive_rate: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ValidationGate {
    pub gate: String,
    pub status: String,
    pub detail: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PipelineSummary {
    pub pipeline_id: String,
    pub status: String,
    pub output_dir: String,
    pub roc_auc: f64,
    pub pr_auc: f64,
    pub missing_input_kind: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct JobResults {
    pub primary_pipeline_id: String,
    pub suite_id: Option<String>,
    pub auc_table_path: String,
    pub roc_points: Vec<[f64; 2]>,
    pub pr_auc: f64,
    pub phase_auc: Vec<MetricSlice>,
    pub sensor_holdout_auc: Vec<MetricSlice>,
    pub class_holdout_auc: Vec<MetricSlice>,
    pub hard_negative_breakdown: Vec<MetricSlice>,
    pub leakage_gates: Vec<ValidationGate>,
    pub calibration_bins: Vec<CalibrationBin>,
    pub export_readiness: String,
    pub pipeline_summaries: Vec<PipelineSummary>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct JobSummary {
    pub job_id: String,
    pub created_utc: String,
    pub status: String,
    pub progress_percent: u8,
    pub message: String,
    pub request: JobComposeRequest,
    pub record_count: usize,
    pub output_dir: String,
    pub artifacts: Vec<JobArtifact>,
    pub results: JobResults,
}

#[derive(Debug, Clone)]
pub struct JobStore {
    jobs: Arc<Mutex<Vec<JobSummary>>>,
}

impl JobStore {
    pub fn seeded(repo_root: &std::path::Path) -> Self {
        let request = JobComposeRequest::default();
        let job = seeded_job(repo_root, request);
        Self {
            jobs: Arc::new(Mutex::new(vec![job])),
        }
    }

    pub fn create(&self, request: JobComposeRequest) -> JobSummary {
        let job = pending_job(request);
        self.jobs.lock().expect("job store").push(job.clone());
        job
    }

    pub fn list(&self) -> Vec<JobSummary> {
        self.jobs.lock().expect("job store").clone()
    }

    pub fn get(&self, job_id: &str) -> Option<JobSummary> {
        self.jobs
            .lock()
            .expect("job store")
            .iter()
            .find(|job| job.job_id == job_id)
            .cloned()
    }

    pub fn cancel(&self, job_id: &str) -> Option<JobSummary> {
        self.update(job_id, |job| {
            if job.status != "completed" {
                job.status = "cancelled".to_string();
                job.message = "job cancelled before ML processing completed".to_string();
                job.progress_percent = job.progress_percent.min(95);
            }
        })
    }

    pub fn update<F>(&self, job_id: &str, mut update: F) -> Option<JobSummary>
    where
        F: FnMut(&mut JobSummary),
    {
        let mut jobs = self.jobs.lock().expect("job store");
        let job = jobs.iter_mut().find(|job| job.job_id == job_id)?;
        update(job);
        Some(job.clone())
    }
}

impl Default for JobComposeRequest {
    fn default() -> Self {
        Self {
            job_type: default_job_type(),
            selection: default_selection(),
            pipeline_id: default_pipeline_id(),
            suite_id: default_suite_id(),
            data_root: default_data_root(),
            out_root: default_out_root(),
            workers_per_pipeline: default_workers_per_pipeline(),
            max_concurrent: default_max_concurrent(),
            seed: default_seed(),
            smoke: true,
            validation_tier: default_validation_tier(),
        }
    }
}

pub fn jobs_router() -> Router<std::sync::Arc<StudioState>> {
    Router::new()
        .route("/api/jobs", get(list_jobs).post(create_job))
        .route("/api/jobs/{id}", get(get_job))
        .route("/api/jobs/{id}/cancel", post(cancel_job))
        .route("/api/jobs/{id}/artifacts", get(list_artifacts))
        .route("/api/jobs/{id}/results", get(get_results))
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
    let execution = tokio::task::spawn_blocking(move || execute_job(&request_for_exec, &repo_root))
        .await;
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

#[derive(Debug, Clone)]
struct JobExecutionPayload {
    message: String,
    record_count: usize,
    output_dir: String,
    artifacts: Vec<JobArtifact>,
    results: JobResults,
}

fn execute_job(
    request: &JobComposeRequest,
    repo_root: &std::path::Path,
) -> Result<JobExecutionPayload, String> {
    if request.selection == "suite" {
        let result = run_suite(
            &request.suite_id,
            Some(repo_root.to_path_buf()),
            std::path::PathBuf::from(&request.data_root),
            std::path::PathBuf::from(&request.out_root),
            request.workers_per_pipeline,
            request.max_concurrent,
            request.seed,
            request.smoke,
            request.validation_tier.clone(),
        )
        .map_err(|err| err.to_string())?;
        Ok(payload_from_suite(request, result))
    } else {
        let result = run_pipeline(PipelineRunRequest {
            pipeline_id: request.pipeline_id.clone(),
            data_root: std::path::PathBuf::from(&request.data_root),
            out_root: std::path::PathBuf::from(&request.out_root),
            workers: request.workers_per_pipeline,
            seed: request.seed,
            smoke: request.smoke,
            validation_tier: request.validation_tier.clone(),
            repo_root: Some(repo_root.to_path_buf()),
        })
        .map_err(|err| err.to_string())?;
        Ok(payload_from_pipeline(request, result))
    }
}

fn payload_from_pipeline(
    request: &JobComposeRequest,
    result: PipelineRunResult,
) -> JobExecutionPayload {
    let primary_summary = PipelineSummary {
        pipeline_id: result.pipeline_id.clone(),
        status: result.status.clone(),
        output_dir: result.output_dir.clone(),
        roc_auc: metric_value(&result.metrics, &["overall", "roc_auc"]).unwrap_or(0.5),
        pr_auc: metric_value(&result.metrics, &["overall", "pr_auc"]).unwrap_or(result.pr_auc()),
        missing_input_kind: result.missing_input_kind.clone(),
    };
    let output_dir = result.output_dir.clone();
    let result_status = result.status.clone();
    let record_count = metric_value(&result.metrics, &["overall", "count"])
        .unwrap_or(0.0)
        .round() as usize;
    let phase_auc = result.metric_rows("phase");
    let sensor_holdout_auc = result.metric_rows("sensor_holdout");
    let class_holdout_auc = result.metric_rows("class_holdout");
    let hard_negative_breakdown = result.metric_rows("hard_negative_breakdown");
    let calibration_bins = result.calibration_bins();
    let leakage_gates: Vec<_> = result
        .gates
        .iter()
        .cloned()
        .map(|gate| ValidationGate {
            gate: gate.gate,
            status: gate.status,
            detail: gate.detail,
        })
        .collect();
    let results = JobResults {
        primary_pipeline_id: result.pipeline_id.clone(),
        suite_id: None,
        auc_table_path: format!("{output_dir}/metrics.json"),
        roc_points: result.roc_points_from_metrics(),
        pr_auc: primary_summary.pr_auc,
        phase_auc,
        sensor_holdout_auc,
        class_holdout_auc,
        hard_negative_breakdown,
        leakage_gates,
        calibration_bins,
        export_readiness: if result_status == "completed" {
            "ready".to_string()
        } else {
            result_status
        },
        pipeline_summaries: vec![primary_summary],
    };
    let artifacts = result
        .artifacts
        .into_iter()
        .map(|artifact| JobArtifact {
            id: artifact.id,
            kind: artifact.kind,
            path: artifact.path,
            ready: artifact.ready,
            incomplete: artifact.incomplete,
        })
        .collect();
    JobExecutionPayload {
        message: format!(
            "{} job completed via pipeline {}",
            request.selection, request.pipeline_id
        ),
        record_count,
        output_dir,
        artifacts,
        results,
    }
}

fn payload_from_suite(
    request: &JobComposeRequest,
    result: PipelineSuiteResult,
) -> JobExecutionPayload {
    let suite_id = result.suite.clone();
    let primary = result.results.first().cloned().unwrap_or_else(|| {
        PipelineRunResult {
            pipeline_id: request.suite_id.clone(),
            run_id: String::new(),
            status: "failed".to_string(),
            message: "suite produced no results".to_string(),
            output_dir: request.out_root.clone(),
            artifacts: Vec::new(),
            metrics: serde_json::json!({}),
            gates: Vec::new(),
            notes: Vec::new(),
            missing_input_kind: None,
            worker_count: request.workers_per_pipeline,
            error_code: None,
            details: None,
        }
    });
    let primary_status = primary.status.clone();
    let pipeline_summaries: Vec<_> = result
        .results
        .iter()
        .map(|item| PipelineSummary {
            pipeline_id: item.pipeline_id.clone(),
            status: item.status.clone(),
            output_dir: item.output_dir.clone(),
            roc_auc: metric_value(&item.metrics, &["overall", "roc_auc"]).unwrap_or(0.5),
            pr_auc: metric_value(&item.metrics, &["overall", "pr_auc"]).unwrap_or(0.5),
            missing_input_kind: item.missing_input_kind.clone(),
        })
        .collect();
    let artifacts = result
        .results
        .iter()
        .flat_map(|item| {
            item.artifacts.iter().cloned().map(|artifact| JobArtifact {
                id: artifact.id,
                kind: artifact.kind,
                path: artifact.path,
                ready: artifact.ready,
                incomplete: artifact.incomplete,
            })
        })
        .collect();
    let output_dir = primary.output_dir.clone();
    let record_count = pipeline_summaries.len();
    let leakage_gates: Vec<_> = primary
        .gates
        .iter()
        .cloned()
        .map(|gate| ValidationGate {
            gate: gate.gate,
            status: gate.status,
            detail: gate.detail,
        })
        .collect();
    let results = JobResults {
        primary_pipeline_id: primary.pipeline_id.clone(),
        suite_id: Some(suite_id),
        auc_table_path: format!("{output_dir}/metrics.json"),
        roc_points: primary.roc_points_from_metrics(),
        pr_auc: metric_value(&primary.metrics, &["overall", "pr_auc"]).unwrap_or(0.5),
        phase_auc: primary.metric_rows("phase"),
        sensor_holdout_auc: primary.metric_rows("sensor_holdout"),
        class_holdout_auc: primary.metric_rows("class_holdout"),
        hard_negative_breakdown: primary.metric_rows("hard_negative_breakdown"),
        leakage_gates,
        calibration_bins: primary.calibration_bins(),
        export_readiness: if primary_status == "completed" {
            "ready".to_string()
        } else {
            primary_status
        },
        pipeline_summaries,
    };
    JobExecutionPayload {
        message: format!("suite {} completed", request.suite_id),
        record_count,
        output_dir,
        artifacts,
        results,
    }
}

fn empty_results(request: &JobComposeRequest, status: &str, message: &str) -> JobResults {
    JobResults {
        primary_pipeline_id: request.pipeline_id.clone(),
        suite_id: if request.selection == "suite" {
            Some(request.suite_id.clone())
        } else {
            None
        },
        auc_table_path: String::new(),
        roc_points: Vec::new(),
        pr_auc: 0.0,
        phase_auc: Vec::new(),
        sensor_holdout_auc: Vec::new(),
        class_holdout_auc: Vec::new(),
        hard_negative_breakdown: Vec::new(),
        leakage_gates: vec![ValidationGate {
            gate: status.to_string(),
            status: "failed".to_string(),
            detail: message.to_string(),
        }],
        calibration_bins: Vec::new(),
        export_readiness: status.to_string(),
        pipeline_summaries: Vec::new(),
    }
}

fn pending_job(request: JobComposeRequest) -> JobSummary {
    let job_hash = job_hash(&request);
    let job_id = format!("job-ml-{}-{job_hash}", request.selection);
    JobSummary {
        job_id,
        created_utc: now_utc_compact(),
        status: "queued".to_string(),
        progress_percent: 0,
        message: "ML processing job queued".to_string(),
        request: request.clone(),
        record_count: 0,
        output_dir: request.out_root.clone(),
        artifacts: Vec::new(),
        results: empty_results(&request, "queued", "ML processing job queued"),
    }
}

fn job_hash(request: &JobComposeRequest) -> String {
    let mut hasher = DefaultHasher::new();
    request.job_type.hash(&mut hasher);
    request.selection.hash(&mut hasher);
    request.pipeline_id.hash(&mut hasher);
    request.suite_id.hash(&mut hasher);
    request.seed.hash(&mut hasher);
    request.smoke.hash(&mut hasher);
    format!("{:08x}", hasher.finish() as u32)
}

fn metric_value(metrics: &serde_json::Value, path: &[&str]) -> Option<f64> {
    let mut current = metrics;
    for key in path {
        current = current.get(*key)?;
    }
    current.as_f64()
}

trait PipelineResultExt {
    fn pr_auc(&self) -> f64;
    fn roc_points_from_metrics(&self) -> Vec<[f64; 2]>;
    fn metric_rows(&self, key: &str) -> Vec<MetricSlice>;
    fn calibration_bins(&self) -> Vec<CalibrationBin>;
}

impl PipelineResultExt for PipelineRunResult {
    fn pr_auc(&self) -> f64 {
        metric_value(&self.metrics, &["overall", "pr_auc"]).unwrap_or(0.0)
    }

    fn roc_points_from_metrics(&self) -> Vec<[f64; 2]> {
        self.metrics
            .get("overall")
            .and_then(|overall| overall.get("roc_auc"))
            .map(|value| vec![[0.0, 0.0], [0.5, value.as_f64().unwrap_or(0.5)], [1.0, 1.0]])
            .unwrap_or_default()
    }

    fn metric_rows(&self, key: &str) -> Vec<MetricSlice> {
        self.metrics
            .get(key)
            .and_then(|value| value.as_array())
            .map(|rows| {
                rows.iter()
                    .map(|row| MetricSlice {
                        label: row
                            .get("phase")
                            .or_else(|| row.get("sensor_id"))
                            .or_else(|| row.get("target_family"))
                            .or_else(|| row.get("hard_negative_family"))
                            .or_else(|| row.get("label"))
                            .and_then(|value| value.as_str())
                            .unwrap_or("unknown")
                            .to_string(),
                        auc: row
                            .get("roc_auc")
                            .or_else(|| row.get("auc"))
                            .and_then(|value| value.as_f64())
                            .unwrap_or(0.5),
                    })
                    .collect()
            })
            .unwrap_or_default()
    }

    fn calibration_bins(&self) -> Vec<CalibrationBin> {
        self.metrics
            .get("calibration_report")
            .and_then(|value| value.get("bins"))
            .and_then(|value| value.as_array())
            .map(|bins| {
                bins.iter()
                    .enumerate()
                    .map(|(index, bin)| CalibrationBin {
                        bin: index as u32,
                        lower: bin.get("lower").and_then(|value| value.as_f64()).unwrap_or(0.0),
                        upper: bin.get("upper").and_then(|value| value.as_f64()).unwrap_or(0.0),
                        count: bin.get("count").and_then(|value| value.as_f64()).unwrap_or(0.0),
                        mean_score: bin
                            .get("mean_score")
                            .and_then(|value| value.as_f64())
                            .unwrap_or(0.0),
                        positive_rate: bin
                            .get("positive_rate")
                            .and_then(|value| value.as_f64())
                            .unwrap_or(0.0),
                    })
                    .collect()
            })
            .unwrap_or_default()
    }
}

fn seeded_job(repo_root: &std::path::Path, request: JobComposeRequest) -> JobSummary {
    let output_dir = repo_root.join(request.out_root.clone()).display().to_string();
    let results = empty_results(&request, "completed", "seeded ML job metadata ready");
    JobSummary {
        job_id: format!("job-ml-{}-seeded", request.selection),
        created_utc: now_utc_compact(),
        status: "completed".to_string(),
        progress_percent: 100,
        message: "seeded ML job metadata ready".to_string(),
        request,
        record_count: 0,
        output_dir,
        artifacts: Vec::new(),
        results,
    }
}

fn now_utc_compact() -> String {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    format!("{secs}Z")
}

fn default_job_type() -> String {
    "ml_processing".to_string()
}

fn default_selection() -> String {
    "pipeline".to_string()
}

fn default_pipeline_id() -> String {
    DEFAULT_PIPELINE_ID.to_string()
}

fn default_suite_id() -> String {
    DEFAULT_SUITE_ID.to_string()
}

fn default_data_root() -> String {
    DEFAULT_DATA_ROOT.to_string()
}

fn default_out_root() -> String {
    DEFAULT_OUT_ROOT.to_string()
}

fn default_workers_per_pipeline() -> usize {
    20
}

fn default_max_concurrent() -> usize {
    3
}

fn default_seed() -> u64 {
    20_260_520_390_001
}

fn default_validation_tier() -> String {
    "evidence_ladder_v1".to_string()
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
        assert_eq!(request.suite_id, DEFAULT_SUITE_ID);
    }
}
