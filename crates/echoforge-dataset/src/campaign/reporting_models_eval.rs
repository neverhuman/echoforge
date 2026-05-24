//! Model artifact writing and evaluation metrics.
//! Extracted from reporting_models.rs for LOC compliance.

use std::collections::BTreeMap;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::campaign::types::{
    CalibrationBin, CampaignRecordSummary, CurvePoint, ModelEvaluationReport,
};
use crate::export::write_json_pretty;
use crate::monte_carlo::DatasetError;

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

pub fn write_model_artifacts(
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
