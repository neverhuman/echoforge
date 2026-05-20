//! Report and metadata generation: quality, normalization, runtime, and
//! manifest generation. Dataset card, schemas, guardrails, feature-family
//! availability, and per-tier metrics live in the sibling `report_card` module.

#[path = "report_sources.rs"]
mod report_sources;

pub(super) use super::report_card::{
    aggregate_per_tier_metrics, dataset_card, feature_family_availability, feature_schema,
    guardrails, label_schema,
};

use std::collections::BTreeMap;
use std::time::Duration;

use echoforge_radar::RuntimePlan;
use serde_json::json;

use crate::monte_carlo::StageTiming;
use crate::split::SplitKind;

use super::config::{MlTrainingDataConfig, NEUTRAL_OBJECT_ID};
use super::types::{
    ColumnStats, DatasetManifest, ExternalCalibrationSource, FeatureFamily,
    MlFeatureSummaryRow, MlRecordSummary, NormalizationStats,
    QualityReport, RuntimeReport,
};

pub(super) fn quality_report(
    config: &MlTrainingDataConfig,
    records: &[MlRecordSummary],
    features: &[MlFeatureSummaryRow],
) -> QualityReport {
    let positive_records = records
        .iter()
        .filter(|record| record.is_public_proxy_positive)
        .count();
    let hard_negative_family_counts = records
        .iter()
        .filter(|record| record.is_hard_negative)
        .fold(BTreeMap::<String, usize>::new(), |mut acc, record| {
            *acc.entry(record.hard_negative_family.clone()).or_insert(0) += 1;
            acc
        });
    QualityReport {
        dataset_id: config.dataset.clone(),
        records: records.len(),
        positive_records,
        positive_fraction_actual: if records.is_empty() {
            0.0
        } else {
            positive_records as f64 / records.len() as f64
        },
        hard_negative_family_coverage: hard_negative_family_counts.len(),
        hard_negative_family_counts,
        minimum_hard_negative_families: 10.min(records.len().saturating_sub(positive_records)),
        feature_families_checked: feature_families()
            .into_iter()
            .map(|family| family.id)
            .collect(),
        all_records_have_required_artifacts: records.iter().all(|record| {
            !record.tensor_dir.is_empty()
                && !record.micro_doppler_dir.is_empty()
                && !record.multi_view_dir.is_empty()
                && !record.learned_windows_dir.is_empty()
        }),
        finite_feature_values: features.iter().all(feature_summary_is_finite),
        split_counts: split_counts(records),
        limitations: vec![
            "Synthetic public-proxy signatures only; no measured object traces are copied into the output."
                .to_string(),
            "GPU readiness selects the runtime plan, while the v1 radar kernel remains CPU-backed and records that recovery path."
                .to_string(),
            "Hard negatives are robustness and false-alarm stressors.".to_string(),
        ],
    }
}

pub(super) fn feature_summary_is_finite(row: &MlFeatureSummaryRow) -> bool {
    [
        row.mean_snr_db,
        row.max_snr_db,
        row.mean_doppler_scr,
        row.mean_rfi_pressure,
        row.dropout_fraction,
        row.mean_micro_doppler_energy,
        row.micro_doppler_peak_hz_proxy,
        row.micro_doppler_bandwidth_hz_proxy,
        row.mean_track_score,
        row.cfar_detection_fraction,
    ]
    .iter()
    .all(|value| value.is_finite())
}

pub(super) fn normalization_stats(
    dataset_id: &str,
    rows: &[MlFeatureSummaryRow],
) -> NormalizationStats {
    let columns = BTreeMap::from([
        (
            "mean_snr_db".to_string(),
            column_stats(rows, |row| row.mean_snr_db as f64),
        ),
        (
            "max_snr_db".to_string(),
            column_stats(rows, |row| row.max_snr_db as f64),
        ),
        (
            "mean_doppler_scr".to_string(),
            column_stats(rows, |row| row.mean_doppler_scr as f64),
        ),
        (
            "mean_rfi_pressure".to_string(),
            column_stats(rows, |row| row.mean_rfi_pressure as f64),
        ),
        (
            "dropout_fraction".to_string(),
            column_stats(rows, |row| row.dropout_fraction as f64),
        ),
        (
            "mean_micro_doppler_energy".to_string(),
            column_stats(rows, |row| row.mean_micro_doppler_energy as f64),
        ),
        (
            "micro_doppler_peak_hz_proxy".to_string(),
            column_stats(rows, |row| row.micro_doppler_peak_hz_proxy as f64),
        ),
        (
            "mean_track_score".to_string(),
            column_stats(rows, |row| row.mean_track_score as f64),
        ),
        (
            "cfar_detection_fraction".to_string(),
            column_stats(rows, |row| row.cfar_detection_fraction as f64),
        ),
    ]);
    NormalizationStats {
        dataset_id: dataset_id.to_string(),
        source: "features.csv record-level training feature summaries".to_string(),
        columns,
    }
}

fn column_stats<F>(rows: &[MlFeatureSummaryRow], select: F) -> ColumnStats
where
    F: Fn(&MlFeatureSummaryRow) -> f64,
{
    if rows.is_empty() {
        return ColumnStats {
            mean: 0.0,
            stddev: 0.0,
            min: 0.0,
            max: 0.0,
        };
    }
    let values = rows.iter().map(select).collect::<Vec<_>>();
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let variance = values
        .iter()
        .map(|value| {
            let diff = *value - mean;
            diff * diff
        })
        .sum::<f64>()
        / values.len() as f64;
    ColumnStats {
        mean,
        stddev: variance.sqrt(),
        min: values.iter().copied().fold(f64::INFINITY, f64::min),
        max: values.iter().copied().fold(f64::NEG_INFINITY, f64::max),
    }
}

pub(super) fn split_counts(records: &[MlRecordSummary]) -> BTreeMap<SplitKind, usize> {
    let mut counts = BTreeMap::new();
    for record in records {
        *counts.entry(record.split).or_insert(0) += 1;
    }
    counts
}

pub(super) fn runtime_report(
    config: &MlTrainingDataConfig,
    runtime: &RuntimePlan,
    worker_count: usize,
    frame_count: usize,
    stage_timings: &[StageTiming],
    elapsed: Duration,
) -> RuntimeReport {
    let rf = crate::guard::runtime_report_fields(runtime, config.records, elapsed);
    RuntimeReport {
        requested_backend: rf.requested_backend,
        selected_backend: rf.selected_backend,
        logical_cores: rf.logical_cores,
        recommended_worker_budget: rf.recommended_worker_budget,
        kernel_backend: rf.kernel_backend,
        gpu_available: rf.gpu.available,
        gpu_usable: rf.gpu.usable,
        gpu_constrained: rf.gpu.constrained,
        gpu_stage_recovery: rf.gpu.stage_recovery,
        gpu_min_free_memory_mb: rf.gpu.min_free_memory_mb,
        gpu_free_memory_mb: rf.gpu.free_memory_mb,
        worker_count,
        worker_cap: 40,
        frame_count,
        records: config.records,
        stage_timings: stage_timings.to_vec(),
        total_elapsed_ns: rf.total_elapsed_ns,
        throughput_records_per_sec: rf.throughput_records_per_sec,
    }
}

pub(super) fn dataset_manifest(
    config: &MlTrainingDataConfig,
    records: Vec<MlRecordSummary>,
    frame_count: usize,
    external_calibration_sources: Vec<ExternalCalibrationSource>,
) -> DatasetManifest {
    let artifacts = BTreeMap::from([
        ("records".to_string(), "records.csv".to_string()),
        ("features".to_string(), "features.csv".to_string()),
        ("splits".to_string(), "split_manifest.csv".to_string()),
        ("dataset_card".to_string(), "dataset_card.json".to_string()),
        (
            "feature_schema".to_string(),
            "feature_schema.json".to_string(),
        ),
        ("label_schema".to_string(), "label_schema.json".to_string()),
        (
            "normalization_stats".to_string(),
            "normalization_stats.json".to_string(),
        ),
        (
            "quality_report".to_string(),
            "quality_report.json".to_string(),
        ),
        (
            "runtime_report".to_string(),
            "runtime_report.json".to_string(),
        ),
    ]);
    DatasetManifest {
        manifest_version: "1".to_string(),
        dataset_id: config.dataset.clone(),
        neutral_object_id: NEUTRAL_OBJECT_ID.to_string(),
        generated_at: config.generated_at.clone(),
        root_seed: config.seed,
        records,
        frame_count,
        frame_rate_hz: config.frame_rate_hz,
        time_window_s: config.time_window_s,
        positive_fraction: config.positive_fraction,
        split_policy: "scenario/object seed grouped; 70% train, 15% validation, 15% test"
            .to_string(),
        feature_families: feature_families(),
        artifacts,
        external_calibration_sources,
        guardrails: guardrails(),
    }
}

pub(super) fn feature_families() -> Vec<FeatureFamily> {
    vec![
        FeatureFamily {
            id: "coherent_range_doppler".to_string(),
            description: "Proxy IQ, integrated range profile, range-Doppler tensor, CFAR/TBD labels."
                .to_string(),
            artifact_pattern: "records/<record_id>/products/*".to_string(),
        },
        FeatureFamily {
            id: "clutter_interference".to_string(),
            description:
                "Local noise floor, Doppler signal-to-clutter ratio, RFI, dropout, and impairment summaries."
                    .to_string(),
            artifact_pattern: "records/<record_id>/streaming_features.csv".to_string(),
        },
        FeatureFamily {
            id: "micro_doppler".to_string(),
            description:
                "STFT spectrogram, weighted spectrum, cepstrum, cadence-velocity, and descriptors."
                    .to_string(),
            artifact_pattern: "records/<record_id>/micro_doppler/*".to_string(),
        },
        FeatureFamily {
            id: "multi_view_tensors".to_string(),
            description:
                "Range-time, Doppler-time, range-Doppler-time tensors plus future angle schema (pending)."
                    .to_string(),
            artifact_pattern: "records/<record_id>/multi_view/*".to_string(),
        },
        FeatureFamily {
            id: "learned_windows".to_string(),
            description: "Normalized 8/16/32-frame tensors for CNN, GRU, and contrastive pretraining."
                .to_string(),
            artifact_pattern: "records/<record_id>/learned_windows/*".to_string(),
        },
    ]
}

pub(super) fn external_calibration_sources() -> Vec<ExternalCalibrationSource> {
    report_sources::external_calibration_sources()
}

pub(super) fn local_external_data_config(
    sources: &[ExternalCalibrationSource],
) -> serde_json::Value {
    json!({
        "policy": "optional local paths only; no external data are copied by default",
        "sources": sources
            .iter()
            .map(|source| {
                json!({
                    "id": source.id,
                    "enabled": false,
                    "local_path": null,
                    "license_review_complete": false
                })
            })
            .collect::<Vec<_>>()
    })
}

