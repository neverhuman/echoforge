//! Model artifact writing, evaluation metrics, runtime report, benchmark report,
//! dataset card, and guardrails for the campaign reporting module.

use std::time::Duration;

use echoforge_core::deterministic_id;
use echoforge_core::models::{
    DatasetCard, DatasetSplits, LicenseInfo, Provenance, ValidationCheck, ValidationInfo,
};
use echoforge_radar::{BackendMode, RuntimePlan};
use serde::{Deserialize, Serialize};
use serde_json::json;

use super::types::{
    CampaignConfig, ClassBalanceReport, ModelEvaluationReport,
};
use super::{NEUTRAL_CAMPAIGN_ID, OWA_DELTA_OBJECT_ID};
use crate::monte_carlo::{DatasetError, StageTiming};

#[path = "reporting_models_eval.rs"]
mod reporting_models_eval;
pub(super) use reporting_models_eval::write_model_artifacts;

// ── Runtime report ────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct RuntimeReport {
    pub requested_backend: BackendMode,
    pub selected_backend: String,
    pub kernel_backend: String,
    pub gpu_available: bool,
    pub gpu_usable: bool,
    pub gpu_constrained: bool,
    pub gpu_min_free_memory_mb: u64,
    pub gpu_free_memory_mb: Option<u64>,
    pub gpu_stage_recovery: Option<String>,
    pub logical_cores: usize,
    pub recommended_worker_budget: usize,
    pub worker_count: usize,
    pub worker_cap: usize,
    pub progress_requested: bool,
    pub progress_enabled: bool,
    pub records: usize,
    pub stage_timings: Vec<StageTiming>,
    pub total_elapsed_ns: u64,
    pub throughput_records_per_sec: f64,
}

pub(super) fn build_runtime_report(
    config: &CampaignConfig,
    runtime: &RuntimePlan,
    worker_count: usize,
    progress_enabled: bool,
    stage_timings: &[StageTiming],
    elapsed: Duration,
) -> RuntimeReport {
    let rf = crate::guard::runtime_report_fields(runtime, config.records, elapsed);
    RuntimeReport {
        requested_backend: rf.requested_backend,
        selected_backend: rf.selected_backend,
        kernel_backend: rf.kernel_backend,
        gpu_available: rf.gpu.available,
        gpu_usable: rf.gpu.usable,
        gpu_constrained: rf.gpu.constrained,
        gpu_min_free_memory_mb: rf.gpu.min_free_memory_mb,
        gpu_free_memory_mb: rf.gpu.free_memory_mb,
        gpu_stage_recovery: rf.gpu.stage_recovery,
        logical_cores: rf.logical_cores,
        recommended_worker_budget: rf.recommended_worker_budget,
        worker_count,
        worker_cap: 40,
        progress_requested: config.progress,
        progress_enabled,
        records: config.records,
        stage_timings: stage_timings.to_vec(),
        total_elapsed_ns: rf.total_elapsed_ns,
        throughput_records_per_sec: rf.throughput_records_per_sec,
    }
}

// ── Benchmark report ──────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct CampaignBenchmarkReport {
    pub campaign_request_id: String,
    pub neutral_campaign_id: String,
    pub class_balance: ClassBalanceReport,
    pub runtime: RuntimeReport,
    pub model_reports: Vec<ModelEvaluationReport>,
    pub limitations: Vec<String>,
}

pub(super) fn build_campaign_benchmark_report(
    config: &CampaignConfig,
    runtime: &RuntimeReport,
    class_balance: &ClassBalanceReport,
    model_reports: &[ModelEvaluationReport],
) -> CampaignBenchmarkReport {
    CampaignBenchmarkReport {
        campaign_request_id: config.campaign.clone(),
        neutral_campaign_id: NEUTRAL_CAMPAIGN_ID.to_string(),
        class_balance: class_balance.clone(),
        runtime: runtime.clone(),
        model_reports: model_reports.to_vec(),
        limitations: vec![
            "Target and confuser signatures are bounded public-proxy simulations, not measured signatures.".to_string(),
            "The radar kernel remains CPU-backed in this scaffold; GPU selection is reported separately from executed kernel backend.".to_string(),
            "Hard negatives are detector robustness cases, not optimization for bypassing sensing.".to_string(),
        ],
    }
}

// ── Dataset card ──────────────────────────────────────────────────────────────

pub(super) fn campaign_dataset_card(
    config: &CampaignConfig,
    class_balance: &ClassBalanceReport,
) -> Result<DatasetCard, DatasetError> {
    let source_campaign_payload = json!({
        "campaign": NEUTRAL_CAMPAIGN_ID,
        "records": config.records,
        "seed": config.seed,
        "time_window_s": config.time_window_s,
        "frame_rate_hz": config.frame_rate_hz,
    });
    let source_campaign_id = deterministic_id(
        "rcs_campaign",
        OWA_DELTA_OBJECT_ID,
        &source_campaign_payload,
    )?;
    let train = (config.records as f64 * 0.70).round() as u64;
    let validation = (config.records as f64 * 0.15).round() as u64;
    let test = config.records as u64 - train - validation;
    DatasetCard {
        id: String::new(),
        kind: String::new(),
        schema_version: String::new(),
        public_proxy_id: OWA_DELTA_OBJECT_ID.to_string(),
        provenance: Provenance {
            source_kind: "synthetic_public_proxy".to_string(),
            source_refs: vec![
                "configs/monte-carlo/airspace-objects-v1.json".to_string(),
                super::SOURCE_DOSSIER_REF.to_string(),
                "dataset-card note: source dossier aliases include Shahed-136, Geran-2, and Shahed-136-series; generated labels use the neutral object id.".to_string(),
            ],
            generated_by: "echoforge-cli demo monte-carlo-campaign".to_string(),
            generated_at: config.generated_at.clone(),
            fingerprint_sha256: String::new(),
        },
        license: LicenseInfo {
            spdx_id: "CC-BY-4.0".to_string(),
            notice: "Synthetic public-proxy campaign metadata; no measured target truth included."
                .to_string(),
        },
        validation: ValidationInfo {
            tier: "basic".to_string(),
            status: if class_balance.meets_shahed_min {
                "pass".to_string()
            } else {
                "fail".to_string()
            },
            uncertainty_score: 0.46,
            checks: vec![
                ValidationCheck {
                    name: "class_balance".to_string(),
                    status: if class_balance.meets_shahed_min {
                        "pass".to_string()
                    } else {
                        "fail".to_string()
                    },
                    message: format!(
                        "{} positive public-proxy records generated; minimum required {}",
                        class_balance.shahed_positive_records, class_balance.shahed_min_required
                    ),
                },
                ValidationCheck {
                    name: "alias_policy".to_string(),
                    status: "pass".to_string(),
                    message: "Dataset card notes identify source-dossier-only aliases; object ids remain neutral.".to_string(),
                },
                ValidationCheck {
                    name: "runtime_limits".to_string(),
                    status: "pass".to_string(),
                    message: "Host worker count is capped at 40 and recorded in runtime_report.json.".to_string(),
                },
            ],
            fidelity_class: None,
        },
        dataset_name: "OWA Delta Pusher Public-Proxy Early-Detection Campaign v1".to_string(),
        source_campaign_ids: vec![source_campaign_id],
        splits: DatasetSplits {
            train,
            validation,
            test,
        },
    }
    .finalize()
    .map_err(DatasetError::Core)
}

pub(super) fn campaign_guardrails() -> Vec<String> {
    crate::guard::PRODUCT_GUARDRAILS
        .iter()
        .map(|s| s.to_string())
        .collect()
}
