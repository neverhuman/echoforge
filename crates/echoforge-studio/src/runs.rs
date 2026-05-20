//! Run-session metadata and download gates for the Studio control plane.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::body::Body;
use axum::extract::{Path, Query, State};
use axum::http::{header, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};

use crate::stream::control::{builtin_scenarios, ControlCommand, SimMode};
use crate::StudioState;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum RunMode {
    Live,
    Replay,
    MonteCarlo,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RunConfig {
    pub scenario_id: String,
    pub scenario_label: String,
    pub mode: RunMode,
    pub seed: u64,
    pub scenario_hash: String,
    pub object_source_card: String,
    pub material_assumption_card: String,
    pub solver_chain_version: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RunArtifact {
    pub id: String,
    pub kind: String,
    pub label: String,
    pub ready: bool,
    pub download_path: String,
    pub requires_validation: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RunValidationSummary {
    pub tier: String,
    pub grade: String,
    pub export_gate_passed: bool,
    pub source_confidence: String,
    pub uncertainty_statement: String,
    pub known_limitations: Vec<String>,
    pub leakage_guard_status: String,
    pub reproducibility_metadata: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RunSummary {
    pub run_id: String,
    pub created_utc: String,
    pub status: String,
    pub config: RunConfig,
    pub validation: RunValidationSummary,
    pub artifacts: Vec<RunArtifact>,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum ReplayMode {
    ExactSeed,
    ModifiedParameters,
    MonteCarloExpansion,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct RunReplayRequest {
    #[serde(default = "default_replay_mode")]
    pub mode: ReplayMode,
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default)]
    pub monte_carlo_count: Option<u32>,
}

fn default_replay_mode() -> ReplayMode {
    ReplayMode::ExactSeed
}

#[derive(Debug, Deserialize)]
pub struct DownloadQuery {
    #[serde(default = "default_download_kind")]
    pub kind: String,
}

fn default_download_kind() -> String {
    "bundle".to_string()
}

#[derive(Debug)]
pub struct RunStore {
    runs: Mutex<Vec<RunSummary>>,
}

impl RunStore {
    pub fn seeded() -> Self {
        let created_utc = now_utc_compact();
        let runs = builtin_scenarios()
            .into_iter()
            .enumerate()
            .map(|(idx, scenario)| {
                let run_id = format!("run-{}-{idx:02}", scenario.id);
                let seed = scenario.base_seed;
                let scenario_hash = scenario_hash(&scenario.id, seed);
                RunSummary {
                    run_id: run_id.clone(),
                    created_utc: created_utc.clone(),
                    status: "validated".to_string(),
                    config: RunConfig {
                        scenario_id: scenario.id.clone(),
                        scenario_label: scenario.label.clone(),
                        mode: RunMode::Live,
                        seed,
                        scenario_hash,
                        object_source_card: object_card_for(&scenario.id),
                        material_assumption_card: "public-proxy-material-assumptions-v0.2.0"
                            .to_string(),
                        solver_chain_version: "echoforge-studio-0.2.0".to_string(),
                    },
                    validation: public_proxy_validation(),
                    artifacts: artifacts_for(&run_id),
                }
            })
            .collect();
        Self {
            runs: Mutex::new(runs),
        }
    }

    pub fn list(&self) -> Vec<RunSummary> {
        self.runs.lock().expect("run store").clone()
    }

    pub fn get(&self, run_id: &str) -> Option<RunSummary> {
        self.runs
            .lock()
            .expect("run store")
            .iter()
            .find(|r| r.run_id == run_id)
            .cloned()
    }
}

pub fn runs_router() -> Router<std::sync::Arc<StudioState>> {
    Router::new()
        .route("/api/runs", get(list_runs))
        .route("/api/runs/{id}", get(get_run))
        .route("/api/runs/{id}/replay", post(replay_run))
        .route("/api/runs/{id}/artifacts", get(list_artifacts))
        .route("/api/runs/{id}/download", get(download_run))
}

async fn list_runs(State(state): State<std::sync::Arc<StudioState>>) -> impl IntoResponse {
    Json(state.run_store.list())
}

async fn get_run(
    State(state): State<std::sync::Arc<StudioState>>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    match state.run_store.get(&id) {
        Some(run) => (StatusCode::OK, Json(run)).into_response(),
        None => not_found("run_not_found", &id),
    }
}

async fn list_artifacts(
    State(state): State<std::sync::Arc<StudioState>>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    match state.run_store.get(&id) {
        Some(run) => (StatusCode::OK, Json(run.artifacts)).into_response(),
        None => not_found("run_not_found", &id),
    }
}

async fn replay_run(
    State(state): State<std::sync::Arc<StudioState>>,
    Path(id): Path<String>,
    Json(req): Json<RunReplayRequest>,
) -> impl IntoResponse {
    let Some(run) = state.run_store.get(&id) else {
        return not_found("run_not_found", &id);
    };
    let scenario_id = run.config.scenario_id.clone();
    let status = state
        .sim_engine
        .apply(ControlCommand::Start {
            scenario_id,
            mode: SimMode::Live,
        })
        .await;
    let body = serde_json::json!({
        "run_id": run.run_id,
        "replay_mode": req.mode,
        "seed": req.seed.unwrap_or(run.config.seed),
        "status": status,
    });
    (StatusCode::OK, Json(body)).into_response()
}

async fn download_run(
    State(state): State<std::sync::Arc<StudioState>>,
    Path(id): Path<String>,
    Query(query): Query<DownloadQuery>,
) -> impl IntoResponse {
    let Some(run) = state.run_store.get(&id) else {
        return not_found("run_not_found", &id);
    };
    if !run.validation.export_gate_passed {
        return (
            StatusCode::CONFLICT,
            Json(serde_json::json!({
                "error": "export_gate_failed",
                "message": "dataset download requires validation, provenance, leakage guard, and reproducibility metadata",
            })),
        )
            .into_response();
    }

    let allowed = ["bundle", "dataset", "validation"];
    if !allowed.contains(&query.kind.as_str()) {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": "unsupported_download_kind",
                "allowed": allowed,
            })),
        )
            .into_response();
    }

    let payload = serde_json::json!({
        "layout": "echoforge-run-bundle-v0.2.0",
        "kind": query.kind,
        "run": run,
        "download_note": "metadata-only fixture bundle; generated solver arrays stay out of Git",
    });
    let bytes = serde_json::to_vec_pretty(&payload).expect("download json");
    let filename = format!("{}-{}.json", id, query.kind);
    let mut response = Response::new(Body::from(bytes));
    *response.status_mut() = StatusCode::OK;
    response.headers_mut().insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("application/json"),
    );
    response.headers_mut().insert(
        header::CONTENT_DISPOSITION,
        HeaderValue::from_str(&format!("attachment; filename=\"{filename}\""))
            .expect("content disposition"),
    );
    response.into_response()
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

fn scenario_hash(id: &str, seed: u64) -> String {
    let mut hasher = DefaultHasher::new();
    id.hash(&mut hasher);
    seed.hash(&mut hasher);
    format!("{:016x}", hasher.finish())
}

fn object_card_for(scenario_id: &str) -> String {
    match scenario_id {
        "multi-confuser" => "public-proxy-airspace-hard-negatives-v0.2.0",
        "coastal-clutter" => "public-proxy-coastal-uav-ingress-v0.2.0",
        _ => "public-proxy-delta-uav-piston-v0.2.0",
    }
    .to_string()
}

fn public_proxy_validation() -> RunValidationSummary {
    RunValidationSummary {
        tier: "V1 public-proxy".to_string(),
        grade: "export-ready-with-limitations".to_string(),
        export_gate_passed: true,
        source_confidence: "public source-card assumptions only".to_string(),
        uncertainty_statement:
            "Uncertainty labels describe synthetic solver and scenario variation; they are not measured-truth signatures."
                .to_string(),
        known_limitations: vec![
            "No proprietary-equivalent platform behavior is claimed.".to_string(),
            "Material cards use conservative open assumptions.".to_string(),
            "Validation tier gates dataset export, not real-world identification.".to_string(),
        ],
        leakage_guard_status: "pass: split manifest and provenance metadata present".to_string(),
        reproducibility_metadata: vec![
            "run_id".to_string(),
            "seed".to_string(),
            "scenario_hash".to_string(),
            "solver_chain_version".to_string(),
        ],
    }
}

fn artifacts_for(run_id: &str) -> Vec<RunArtifact> {
    [
        ("bundle", "Run bundle"),
        ("dataset", "Dataset card and split manifest"),
        ("validation", "Validation dossier"),
    ]
    .into_iter()
    .map(|(kind, label)| RunArtifact {
        id: format!("{run_id}-{kind}"),
        kind: kind.to_string(),
        label: label.to_string(),
        ready: true,
        download_path: format!("/api/runs/{run_id}/download?kind={kind}"),
        requires_validation: true,
    })
    .collect()
}

fn now_utc_compact() -> String {
    let seconds = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    format!("unix-{seconds}")
}
