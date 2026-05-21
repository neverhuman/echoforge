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

use crate::stream::control::{builtin_scenarios, ControlCommand, SimMode};
use crate::StudioState;

mod types;

pub use types::{
    DownloadQuery, MonteCarloRunRequest, RunArchiveRequest, RunArtifact, RunConfig,
    RunDuplicateRequest, RunMode, RunQueueSummary, RunReplayRequest, RunSummary,
    RunValidationSummary,
};

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

    pub fn queue_summary(&self) -> RunQueueSummary {
        let runs = self.runs.lock().expect("run store");
        let mut validation_tiers = runs
            .iter()
            .map(|run| run.validation.tier.clone())
            .collect::<Vec<_>>();
        validation_tiers.sort();
        validation_tiers.dedup();
        RunQueueSummary {
            total: runs.len(),
            active: runs
                .iter()
                .filter(|run| run.status != "archived" && run.status != "completed")
                .count(),
            archived: runs.iter().filter(|run| run.status == "archived").count(),
            export_ready: runs
                .iter()
                .filter(|run| run.validation.export_gate_passed && run.status != "archived")
                .count(),
            queued: runs.iter().filter(|run| run.status == "queued").count(),
            newest_created_utc: runs.iter().map(|run| run.created_utc.clone()).max(),
            validation_tiers,
        }
    }

    pub fn get(&self, run_id: &str) -> Option<RunSummary> {
        self.runs
            .lock()
            .expect("run store")
            .iter()
            .find(|r| r.run_id == run_id)
            .cloned()
    }

    pub fn archive(&self, run_id: &str, _request: RunArchiveRequest) -> Option<RunSummary> {
        let mut runs = self.runs.lock().expect("run store");
        let run = runs.iter_mut().find(|r| r.run_id == run_id)?;
        run.status = "archived".to_string();
        Some(run.clone())
    }

    pub fn restore(&self, run_id: &str) -> Option<RunSummary> {
        let mut runs = self.runs.lock().expect("run store");
        let run = runs.iter_mut().find(|r| r.run_id == run_id)?;
        if run.status == "archived" {
            run.status = "validated".to_string();
        }
        Some(run.clone())
    }

    pub fn duplicate(&self, run_id: &str, request: RunDuplicateRequest) -> Option<RunSummary> {
        let mut runs = self.runs.lock().expect("run store");
        let base = runs.iter().find(|r| r.run_id == run_id)?.clone();
        let seed = request.seed.unwrap_or(base.config.seed.saturating_add(1));
        let copy_index = runs
            .iter()
            .filter(|run| run.config.scenario_id == base.config.scenario_id)
            .count();
        let new_run_id = format!("run-{}-copy-{copy_index:02}", base.config.scenario_id);
        let mut duplicated = base;
        duplicated.run_id = unique_run_id(&runs, &new_run_id);
        duplicated.created_utc = now_utc_compact();
        duplicated.status = "queued".to_string();
        duplicated.config.mode = request.mode.unwrap_or(RunMode::Replay);
        duplicated.config.seed = seed;
        duplicated.config.scenario_hash = scenario_hash(&duplicated.config.scenario_id, seed);
        duplicated.artifacts = artifacts_for(&duplicated.run_id);
        runs.push(duplicated.clone());
        Some(duplicated)
    }

    pub fn create_monte_carlo(&self, request: MonteCarloRunRequest) -> RunSummary {
        let mut runs = self.runs.lock().expect("run store");
        let scenario = builtin_scenarios()
            .into_iter()
            .find(|scenario| scenario.id == request.scenario_id);
        let scenario_label = scenario
            .as_ref()
            .map(|scenario| scenario.label.clone())
            .unwrap_or_else(|| request.scenario_id.clone());
        let copy_index = runs
            .iter()
            .filter(|run| run.config.scenario_id == request.scenario_id)
            .count();
        let run_id = unique_run_id(
            &runs,
            &format!("run-{}-mc-{copy_index:02}", request.scenario_id),
        );
        let mut validation = public_proxy_validation();
        validation.tier = request.validation_target.clone();
        validation.known_limitations.push(format!(
            "Monte Carlo request uses {} synthetic draws with {} workers; review exported artifacts before reuse.",
            request.run_count, request.workers
        ));
        validation.reproducibility_metadata.extend([
            "run_count".to_string(),
            "workers".to_string(),
            "detector_pipeline".to_string(),
            "weather_profile".to_string(),
        ]);
        let run = RunSummary {
            run_id: run_id.clone(),
            created_utc: now_utc_compact(),
            status: "queued".to_string(),
            config: RunConfig {
                scenario_id: request.scenario_id.clone(),
                scenario_label,
                mode: RunMode::MonteCarlo,
                seed: request.seed,
                scenario_hash: scenario_hash(&request.scenario_id, request.seed),
                object_source_card: format!("{} / {}", request.source_pack, request.object_pack),
                material_assumption_card: format!(
                    "{} with {} hard-negative packs",
                    request.weather_profile,
                    request.hard_negatives.len()
                ),
                solver_chain_version: format!(
                    "echoforge-studio-0.2.0+{}+smoke-{}",
                    request.detector_pipeline, request.smoke
                ),
            },
            validation,
            artifacts: artifacts_for(&run_id),
        };
        runs.push(run.clone());
        run
    }
}

pub fn runs_router() -> Router<std::sync::Arc<StudioState>> {
    Router::new()
        .route("/api/runs", get(list_runs).post(create_monte_carlo_run))
        .route("/api/runs/queue/summary", get(queue_summary))
        .route("/api/runs/{id}", get(get_run))
        .route("/api/runs/{id}/replay", post(replay_run))
        .route("/api/runs/{id}/archive", post(archive_run))
        .route("/api/runs/{id}/restore", post(restore_run))
        .route("/api/runs/{id}/duplicate", post(duplicate_run))
        .route("/api/runs/{id}/artifacts", get(list_artifacts))
        .route("/api/runs/{id}/download", get(download_run))
}

async fn list_runs(State(state): State<std::sync::Arc<StudioState>>) -> impl IntoResponse {
    Json(state.run_store.list())
}

async fn queue_summary(State(state): State<std::sync::Arc<StudioState>>) -> impl IntoResponse {
    Json(state.run_store.queue_summary())
}

async fn create_monte_carlo_run(
    State(state): State<std::sync::Arc<StudioState>>,
    Json(req): Json<MonteCarloRunRequest>,
) -> impl IntoResponse {
    let run = state.run_store.create_monte_carlo(req);
    (StatusCode::CREATED, Json(run)).into_response()
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

async fn archive_run(
    State(state): State<std::sync::Arc<StudioState>>,
    Path(id): Path<String>,
    Json(req): Json<RunArchiveRequest>,
) -> impl IntoResponse {
    match state.run_store.archive(&id, req) {
        Some(run) => (StatusCode::OK, Json(run)).into_response(),
        None => not_found("run_not_found", &id),
    }
}

async fn restore_run(
    State(state): State<std::sync::Arc<StudioState>>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    match state.run_store.restore(&id) {
        Some(run) => (StatusCode::OK, Json(run)).into_response(),
        None => not_found("run_not_found", &id),
    }
}

async fn duplicate_run(
    State(state): State<std::sync::Arc<StudioState>>,
    Path(id): Path<String>,
    Json(req): Json<RunDuplicateRequest>,
) -> impl IntoResponse {
    match state.run_store.duplicate(&id, req) {
        Some(run) => (StatusCode::CREATED, Json(run)).into_response(),
        None => not_found("run_not_found", &id),
    }
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

fn unique_run_id(runs: &[RunSummary], prefix: &str) -> String {
    if !runs.iter().any(|run| run.run_id == prefix) {
        return prefix.to_string();
    }
    let mut suffix = 1usize;
    loop {
        let candidate = format!("{prefix}-{suffix}");
        if !runs.iter().any(|run| run.run_id == candidate) {
            return candidate;
        }
        suffix += 1;
    }
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
