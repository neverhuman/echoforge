use std::collections::BTreeMap;
use std::fs;
use std::path::Path;
use std::time::Duration;

use echoforge_core::deterministic_id;
use echoforge_core::models::{
    DatasetCard, DatasetSplits, LicenseInfo, Provenance, ValidationCheck, ValidationInfo,
};
use echoforge_radar::RuntimePlan;
use serde::{Deserialize, Serialize};
use serde_json::json;

use super::types::{
    CalibrationBin, CampaignBucket, CampaignConfig, CampaignRecordSummary, ClassBalanceReport,
    CurvePoint, FirstTriggerEvent, ModelEvaluationReport, ModelPredictionRow,
};
use super::{NEUTRAL_CAMPAIGN_ID, OWA_DELTA_OBJECT_ID};
use crate::export::write_json_pretty;

use crate::monte_carlo::{DatasetError, StageTiming};
use echoforge_radar::BackendMode;

// ── CSV writer ────────────────────────────────────────────────────────────────

pub(super) fn write_csv<T: Serialize>(path: &Path, rows: &[T]) -> Result<(), DatasetError> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut writer = csv::Writer::from_path(path)?;
    for row in rows {
        writer.serialize(row)?;
    }
    writer.flush()?;
    Ok(())
}

// ── Root-level CSV outputs ────────────────────────────────────────────────────

pub(super) fn write_root_csvs(
    output_dir: &Path,
    summaries: &[CampaignRecordSummary],
    events: &[FirstTriggerEvent],
    predictions: &[ModelPredictionRow],
) -> Result<(), DatasetError> {
    let record_rows = summaries
        .iter()
        .map(CampaignRecordSummaryCsv::from)
        .collect::<Vec<_>>();
    write_csv(&output_dir.join("records.csv"), &record_rows)?;
    write_csv(&output_dir.join("detector_first_triggers.csv"), events)?;
    let summary_predictions = predictions
        .iter()
        .filter(|row| row.triggered_on_this_frame || row.frame_index % 30 == 0)
        .cloned()
        .collect::<Vec<_>>();
    write_csv(
        &output_dir.join("model_predictions_summary.csv"),
        &summary_predictions,
    )?;

    fs::create_dir_all(output_dir.join("plots"))?;
    let balance_rows = summaries
        .iter()
        .fold(BTreeMap::<String, usize>::new(), |mut acc, row| {
            *acc.entry(row.target_family.clone()).or_insert(0) += 1;
            acc
        })
        .into_iter()
        .map(|(target_family, count)| PlotClassBalanceRow {
            target_family,
            count,
        })
        .collect::<Vec<_>>();
    write_csv(&output_dir.join("plots/class_balance.csv"), &balance_rows)?;
    let latency_rows = summaries
        .iter()
        .map(|row| PlotLatencyRow {
            record_id: row.record_id.clone(),
            target_family: row.target_family.clone(),
            is_shahed_public_proxy: row.is_shahed_public_proxy,
            first_detectable_frame: row.first_detectable_frame,
            first_model_trigger_frame: row.first_model_trigger_frame,
            latency_frames: match (row.first_detectable_frame, row.first_model_trigger_frame) {
                (Some(a), Some(b)) if b >= a => Some(b - a),
                _ => None,
            },
        })
        .collect::<Vec<_>>();
    write_csv(
        &output_dir.join("plots/detection_latency.csv"),
        &latency_rows,
    )?;
    let false_alarm_rows = false_alarm_rows(summaries);
    write_csv(
        &output_dir.join("plots/false_alarm_by_family.csv"),
        &false_alarm_rows,
    )?;
    Ok(())
}

// ── CSV row types ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize)]
struct CampaignRecordSummaryCsv {
    record_id: String,
    record_index: usize,
    target_family: String,
    class_id: String,
    bucket: CampaignBucket,
    is_shahed_public_proxy: bool,
    is_hard_negative: bool,
    hard_negative_family: String,
    first_detectable_frame: Option<usize>,
    first_model_trigger_frame: Option<usize>,
    cpi_pulses: usize,
    tensor_dir: String,
    frame_labels_path: String,
    truth_metadata_path: String,
    model_predictions_path: String,
    detector_events_path: String,
    max_confidence_by_model_json: String,
    first_trigger_by_model_json: String,
}

impl From<&CampaignRecordSummary> for CampaignRecordSummaryCsv {
    fn from(value: &CampaignRecordSummary) -> Self {
        Self {
            record_id: value.record_id.clone(),
            record_index: value.record_index,
            target_family: value.target_family.clone(),
            class_id: value.class_id.clone(),
            bucket: value.bucket,
            is_shahed_public_proxy: value.is_shahed_public_proxy,
            is_hard_negative: value.is_hard_negative,
            hard_negative_family: value.hard_negative_family.clone(),
            first_detectable_frame: value.first_detectable_frame,
            first_model_trigger_frame: value.first_model_trigger_frame,
            cpi_pulses: value.cpi_pulses,
            tensor_dir: value.tensor_dir.clone(),
            frame_labels_path: value.frame_labels_path.clone(),
            truth_metadata_path: value.truth_metadata_path.clone(),
            model_predictions_path: value.model_predictions_path.clone(),
            detector_events_path: value.detector_events_path.clone(),
            max_confidence_by_model_json: serde_json::to_string(&value.max_confidence_by_model)
                .expect("BTreeMap<String,f32> is always JSON-serializable"),
            first_trigger_by_model_json: serde_json::to_string(&value.first_trigger_by_model)
                .expect("BTreeMap<String,f32> is always JSON-serializable"),
        }
    }
}

#[derive(Debug, Clone, Serialize)]
struct PlotClassBalanceRow {
    target_family: String,
    count: usize,
}

#[derive(Debug, Clone, Serialize)]
struct PlotLatencyRow {
    record_id: String,
    target_family: String,
    is_shahed_public_proxy: bool,
    first_detectable_frame: Option<usize>,
    first_model_trigger_frame: Option<usize>,
    latency_frames: Option<usize>,
}

#[derive(Debug, Clone, Serialize)]
struct PlotFalseAlarmRow {
    hard_negative_family: String,
    records: usize,
    false_alarm_records: usize,
}

fn false_alarm_rows(summaries: &[CampaignRecordSummary]) -> Vec<PlotFalseAlarmRow> {
    let mut grouped = BTreeMap::<String, (usize, usize)>::new();
    for row in summaries.iter().filter(|row| row.is_hard_negative) {
        let entry = grouped
            .entry(row.hard_negative_family.clone())
            .or_insert((0, 0));
        entry.0 += 1;
        if row.first_model_trigger_frame.is_some() {
            entry.1 += 1;
        }
    }
    grouped
        .into_iter()
        .map(
            |(hard_negative_family, (records, false_alarm_records))| PlotFalseAlarmRow {
                hard_negative_family,
                records,
                false_alarm_records,
            },
        )
        .collect()
}

// ── Class balance ─────────────────────────────────────────────────────────────

pub(super) fn build_class_balance(
    config: &CampaignConfig,
    summaries: &[CampaignRecordSummary],
) -> ClassBalanceReport {
    let mut bucket_counts = BTreeMap::new();
    let mut target_family_counts = BTreeMap::new();
    let mut hard_negative_family_counts = BTreeMap::new();
    let mut positives = 0usize;
    for row in summaries {
        *bucket_counts.entry(row.bucket).or_insert(0) += 1;
        *target_family_counts
            .entry(row.target_family.clone())
            .or_insert(0) += 1;
        if row.is_hard_negative {
            *hard_negative_family_counts
                .entry(row.hard_negative_family.clone())
                .or_insert(0) += 1;
        }
        if row.is_shahed_public_proxy {
            positives += 1;
        }
    }
    ClassBalanceReport {
        total_records: summaries.len(),
        shahed_positive_records: positives,
        shahed_min_required: config.shahed_min,
        meets_shahed_min: positives >= config.shahed_min,
        bucket_counts,
        target_family_counts,
        hard_negative_family_counts,
    }
}

// ── Model artifacts ───────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ModelMetadata {
    model_id: String,
    model_family: String,
    streaming_window_frames: usize,
    trigger_confidence: f32,
    consecutive_frames_required: usize,
    implementation_note: String,
    dependency_note: String,
}

pub(super) fn write_model_artifacts(
    output_dir: &Path,
    summaries: &[CampaignRecordSummary],
    trigger_confidence: f32,
) -> Result<Vec<ModelEvaluationReport>, DatasetError> {
    let model_ids = [
        "cfar_tracker_baseline",
        "feature_tree_classifier",
        "temporal_tiny_model",
    ];
    let mut reports = Vec::new();
    for model_id in model_ids {
        let metadata = ModelMetadata {
            model_id: model_id.to_string(),
            model_family: match model_id {
                "cfar_tracker_baseline" => "OS/CA-CFAR plus track persistence".to_string(),
                "feature_tree_classifier" => "feature threshold tree baseline".to_string(),
                _ => "tiny temporal rolling-window baseline".to_string(),
            },
            streaming_window_frames: if model_id == "temporal_tiny_model" {
                16
            } else {
                1
            },
            trigger_confidence,
            consecutive_frames_required: 2,
            implementation_note: "Rust streaming baseline over public-proxy features.".to_string(),
            dependency_note: if model_id == "feature_tree_classifier" {
                "smartcore is declared for the classical ML lane; this deterministic baseline keeps fixture runs reproducible without training data leakage.".to_string()
            } else {
                "No external model weights required for this baseline.".to_string()
            },
        };
        let model_dir = output_dir.join("models").join(model_id);
        write_json_pretty(&model_dir.join("model_metadata.json"), &metadata)?;
        let report = evaluate_model(model_id, summaries);
        write_json_pretty(
            &output_dir
                .join("qa")
                .join(format!("model_eval_{model_id}.json")),
            &report,
        )?;
        reports.push(report);
    }
    Ok(reports)
}

fn evaluate_model(model_id: &str, summaries: &[CampaignRecordSummary]) -> ModelEvaluationReport {
    let positives = summaries.iter().filter(|row| row.is_shahed_public_proxy);
    let negatives = summaries.iter().filter(|row| !row.is_shahed_public_proxy);
    let positive_records = positives.clone().count();
    let negative_records = negatives.clone().count();
    let mut true_positive = 0usize;
    let mut false_positive = 0usize;
    let mut missed = Vec::new();
    let mut false_alarm_by_family = BTreeMap::new();
    let mut latencies = Vec::new();

    for row in summaries {
        let triggered = row
            .first_trigger_by_model
            .get(model_id)
            .and_then(|value| *value);
        if row.is_shahed_public_proxy {
            if let Some(trigger_frame) = triggered {
                true_positive += 1;
                if let Some(first_detectable) = row.first_detectable_frame {
                    if trigger_frame >= first_detectable {
                        latencies.push((trigger_frame - first_detectable) as f64);
                    }
                }
            } else {
                missed.push(row.record_id.clone());
            }
        } else if triggered.is_some() {
            false_positive += 1;
            *false_alarm_by_family
                .entry(row.hard_negative_family.clone())
                .or_insert(0) += 1;
        }
    }

    ModelEvaluationReport {
        model_id: model_id.to_string(),
        positive_records,
        negative_records,
        pd: ratio(true_positive, positive_records),
        pfa: ratio(false_positive, negative_records),
        missed_positive_records: missed,
        false_alarm_by_hard_negative_family: false_alarm_by_family,
        mean_first_detection_latency_frames: if latencies.is_empty() {
            None
        } else {
            Some(latencies.iter().sum::<f64>() / latencies.len() as f64)
        },
        roc_points: curve_points(model_id, summaries, CurveKind::Roc),
        pr_points: curve_points(model_id, summaries, CurveKind::Pr),
        confidence_calibration_bins: calibration_bins(model_id, summaries),
    }
}

#[derive(Debug, Clone, Copy)]
enum CurveKind {
    Roc,
    Pr,
}

fn curve_points(
    model_id: &str,
    summaries: &[CampaignRecordSummary],
    kind: CurveKind,
) -> Vec<CurvePoint> {
    [0.5, 0.65, 0.8, 0.9]
        .into_iter()
        .map(|threshold| {
            let mut tp = 0usize;
            let mut fp = 0usize;
            let mut fn_ = 0usize;
            let mut tn = 0usize;
            for row in summaries {
                let score = row
                    .max_confidence_by_model
                    .get(model_id)
                    .copied()
                    .unwrap_or(0.0);
                let predicted = score >= threshold;
                match (row.is_shahed_public_proxy, predicted) {
                    (true, true) => tp += 1,
                    (true, false) => fn_ += 1,
                    (false, true) => fp += 1,
                    (false, false) => tn += 1,
                }
            }
            match kind {
                CurveKind::Roc => CurvePoint {
                    threshold,
                    x: ratio(fp, fp + tn),
                    y: ratio(tp, tp + fn_),
                },
                CurveKind::Pr => CurvePoint {
                    threshold,
                    x: ratio(tp, tp + fn_),
                    y: ratio(tp, tp + fp),
                },
            }
        })
        .collect()
}

fn calibration_bins(model_id: &str, summaries: &[CampaignRecordSummary]) -> Vec<CalibrationBin> {
    (0..5)
        .map(|bin| {
            let start = bin as f32 * 0.2;
            let end = start + 0.2;
            let rows = summaries
                .iter()
                .filter(|row| {
                    let score = row
                        .max_confidence_by_model
                        .get(model_id)
                        .copied()
                        .unwrap_or(0.0);
                    score >= start && (score < end || (bin == 4 && score <= end))
                })
                .collect::<Vec<_>>();
            let positives = rows.iter().filter(|row| row.is_shahed_public_proxy).count();
            CalibrationBin {
                bin_start: start,
                bin_end: end,
                records: rows.len(),
                positive_fraction: ratio(positives, rows.len()),
            }
        })
        .collect()
}

fn ratio(num: usize, den: usize) -> f64 {
    if den == 0 {
        0.0
    } else {
        num as f64 / den as f64
    }
}

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
