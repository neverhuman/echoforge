pub(super) use super::reporting_models::{
    build_campaign_benchmark_report, build_runtime_report, campaign_dataset_card,
    campaign_guardrails, write_model_artifacts,
};

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use serde::Serialize;

use super::types::{
    CampaignBucket, CampaignConfig, CampaignRecordSummary, ClassBalanceReport,
    FirstTriggerEvent, ModelPredictionRow,
};
use crate::monte_carlo::DatasetError;

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

