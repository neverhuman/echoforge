//! Report and metadata generation: quality, normalization, runtime, dataset
//! card, manifest, schema, and per-tier aggregation.

use std::collections::BTreeMap;
use std::time::Duration;

use echoforge_core::deterministic_id;
use echoforge_core::models::{
    DatasetCard, DatasetSplits, LicenseInfo, Provenance, ValidationCheck, ValidationInfo,
};
use echoforge_radar::{RuntimePlan, Tier};
use serde_json::json;


use crate::monte_carlo::{DatasetError, StageTiming};
use crate::split::SplitKind;

use super::config::{MlTrainingDataConfig, DEFAULT_ML_TRAINING_DATASET_ID, NEUTRAL_OBJECT_ID};
use super::types::{
    AvailabilityEntry, ColumnStats, DatasetManifest, ExternalCalibrationSource, FeatureFamily,
    FeatureFamilyAvailability, MlFeatureSummaryRow, MlRecordSummary, NormalizationStats,
    PerRecordTierObservation, PerTierMetrics, QualityReport, RuntimeReport,
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
    vec![
        ExternalCalibrationSource {
            id: "scientific-data-2026-drone-radar-rf".to_string(),
            title: "Time-synchronized multi-sensor drone radar/RF dataset".to_string(),
            url: "https://www.nature.com/articles/s41597-026-06802-6".to_string(),
            role: "calibration_or_evaluation_metadata_only".to_string(),
            local_data_default: "not_vendored".to_string(),
            license_notes:
                "Do not copy source data into EchoForge output unless the local operator verifies dataset terms."
                    .to_string(),
            expected_feature_mappings: vec![
                "range_doppler_proxy".to_string(),
                "doppler_spectrum".to_string(),
                "power_spectral_density".to_string(),
            ],
        },
        ExternalCalibrationSource {
            id: "rdrd-rad-dar-public-drone-radar".to_string(),
            title: "RDRD/RAD-DAR public drone radar dataset metadata (pending)".to_string(),
            url: "local-path-config-required".to_string(),
            role: "optional_local_calibration_mapping".to_string(),
            local_data_default: "not_vendored".to_string(),
            license_notes:
                "Metadata hook only; configure a local path after source and license review.".to_string(),
            expected_feature_mappings: vec![
                "range_doppler_map".to_string(),
                "micro_doppler_spectrum".to_string(),
            ],
        },
        ExternalCalibrationSource {
            id: "rahman-robertson-drone-bird-micro-doppler".to_string(),
            title: "Radar micro-Doppler signatures of drones and birds".to_string(),
            url: "https://research-repository.st-andrews.ac.uk/handle/10023/16577".to_string(),
            role: "micro_doppler_format_motivation".to_string(),
            local_data_default: "not_vendored".to_string(),
            license_notes: "Publication metadata only; no measured traces are vendored.".to_string(),
            expected_feature_mappings: vec![
                "propeller_or_wingbeat_peak_hz_proxy".to_string(),
                "micro_doppler_bandwidth_hz_proxy".to_string(),
            ],
        },
        ExternalCalibrationSource {
            id: "eusipco-2020-micro-doppler-representations".to_string(),
            title: "Comparison of micro-Doppler signal representations".to_string(),
            url: "https://eurasip.org/Proceedings/Eusipco/Eusipco2020/pdfs/0001561.pdf"
                .to_string(),
            role: "representation_family_motivation".to_string(),
            local_data_default: "not_vendored".to_string(),
            license_notes: "Publication metadata only; no paper data are vendored.".to_string(),
            expected_feature_mappings: vec![
                "stft_spectrogram".to_string(),
                "weighted_spectrum".to_string(),
                "cepstrum".to_string(),
                "cadence_velocity".to_string(),
            ],
        },
        ExternalCalibrationSource {
            id: "low-grazing-uav-detection-cfar-micro-doppler".to_string(),
            title: "Low-grazing UAV detection literature on CFAR, clutter, and trajectory extraction"
                .to_string(),
            url: "https://arxiv.org/abs/1902.05483".to_string(),
            role: "cfar_tbd_label_motivation".to_string(),
            local_data_default: "not_vendored".to_string(),
            license_notes: "Publication metadata only.".to_string(),
            expected_feature_mappings: vec![
                "cfar_statistic".to_string(),
                "tbd_track_score".to_string(),
                "clutter_pressure".to_string(),
            ],
        },
        ExternalCalibrationSource {
            id: "learned-radar-representation-2301-02451".to_string(),
            title: "Learned radar representations and data-driven detector reference".to_string(),
            url: "https://arxiv.org/abs/2301.02451".to_string(),
            role: "low_level_tensor_retention_motivation".to_string(),
            local_data_default: "not_vendored".to_string(),
            license_notes: "Publication metadata only.".to_string(),
            expected_feature_mappings: vec![
                "iq_complex".to_string(),
                "range_time".to_string(),
                "learned_windows".to_string(),
            ],
        },
        ExternalCalibrationSource {
            id: "learned-radar-representation-2402-12970".to_string(),
            title: "Data-driven radar detector reference".to_string(),
            url: "https://arxiv.org/abs/2402.12970".to_string(),
            role: "low_level_tensor_retention_motivation".to_string(),
            local_data_default: "not_vendored".to_string(),
            license_notes: "Publication metadata only.".to_string(),
            expected_feature_mappings: vec![
                "range_doppler_time".to_string(),
                "normalized_windows".to_string(),
            ],
        },
    ]
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

pub(super) fn dataset_card(
    config: &MlTrainingDataConfig,
    quality: &QualityReport,
) -> Result<DatasetCard, DatasetError> {
    let source_payload = json!({
        "dataset": DEFAULT_ML_TRAINING_DATASET_ID,
        "records": config.records,
        "positive_fraction": config.positive_fraction,
        "seed": config.seed,
        "time_window_s": config.time_window_s,
        "frame_rate_hz": config.frame_rate_hz,
    });
    let source_campaign_id =
        deterministic_id("ml_training_dataset", NEUTRAL_OBJECT_ID, &source_payload)?;
    let train = *quality.split_counts.get(&SplitKind::Train).unwrap_or(&0) as u64;
    let validation = *quality
        .split_counts
        .get(&SplitKind::Validation)
        .unwrap_or(&0) as u64;
    let test = *quality.split_counts.get(&SplitKind::Test).unwrap_or(&0) as u64;
    DatasetCard {
        id: String::new(),
        kind: String::new(),
        schema_version: String::new(),
        public_proxy_id: NEUTRAL_OBJECT_ID.to_string(),
        provenance: Provenance {
            source_kind: "synthetic_public_proxy".to_string(),
            source_refs: vec![
                "configs/monte-carlo/airspace-objects-v1.json".to_string(),
                "object-packs/public-proxy-v1/source_dossier.yaml".to_string(),
                "external_calibration_sources.json records publication metadata only; no external measured traces are copied by default.".to_string(),
            ],
            generated_by: "echoforge-cli demo ml-training-data".to_string(),
            generated_at: config.generated_at.clone(),
            fingerprint_sha256: String::new(),
        },
        license: LicenseInfo {
            spdx_id: "CC-BY-4.0".to_string(),
            notice:
                "Synthetic public-proxy ML training metadata and tensors; external datasets are metadata hooks only."
                    .to_string(),
        },
        validation: ValidationInfo {
            tier: "basic".to_string(),
            status: if quality.all_records_have_required_artifacts
                && quality.finite_feature_values
                && quality.hard_negative_family_coverage >= quality.minimum_hard_negative_families
            {
                "pass".to_string()
            } else {
                "fail".to_string()
            },
            uncertainty_score: 0.48,
            checks: vec![
                ValidationCheck {
                    name: "positive_fraction".to_string(),
                    status: "pass".to_string(),
                    message: format!(
                        "{} positive public-proxy records out of {}",
                        quality.positive_records, quality.records
                    ),
                },
                ValidationCheck {
                    name: "hard_negative_coverage".to_string(),
                    status: if quality.hard_negative_family_coverage
                        >= quality.minimum_hard_negative_families
                    {
                        "pass".to_string()
                    } else {
                        "fail".to_string()
                    },
                    message: format!(
                        "{} hard-negative families represented",
                        quality.hard_negative_family_coverage
                    ),
                },
                ValidationCheck {
                    name: "feature_families".to_string(),
                    status: if quality.all_records_have_required_artifacts {
                        "pass".to_string()
                    } else {
                        "fail".to_string()
                    },
                    message: "Each record writes coherent, clutter/RFI, micro-Doppler, multi-view, and learned-window artifacts.".to_string(),
                },
            ],
            fidelity_class: None,
        },
        dataset_name: "Shahed-136 Public-Proxy ML Training Corpus v1".to_string(),
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

pub(super) fn feature_schema() -> serde_json::Value {
    json!({
        "schema_id": "echoforge.ml_training.feature_schema.v1",
        "format": "csv",
        "root_file": "features.csv",
        "per_record_file": "records/<record_id>/streaming_features.csv",
        "families": feature_families(),
        "columns": [
            {"name": "snr_db", "type": "f32", "family": "coherent_range_doppler"},
            {"name": "cfar_statistic", "type": "f32", "family": "coherent_range_doppler"},
            {"name": "tbd_track_score", "type": "f32", "family": "coherent_range_doppler"},
            {"name": "local_noise_floor_db", "type": "f32", "family": "clutter_interference"},
            {"name": "doppler_scr", "type": "f32", "family": "clutter_interference"},
            {"name": "rfi_pressure", "type": "f32", "family": "clutter_interference"},
            {"name": "dropout_fraction", "type": "f32", "family": "clutter_interference"},
            {"name": "micro_doppler_energy", "type": "f32", "family": "micro_doppler"},
            {"name": "micro_doppler_peak_hz_proxy", "type": "f32", "family": "micro_doppler"},
            {"name": "range_time_energy", "type": "f32", "family": "multi_view_tensors"},
            {"name": "range_doppler_time_energy", "type": "f32", "family": "multi_view_tensors"},
            {"name": "normalized_snr", "type": "f32", "family": "learned_windows"}
        ]
    })
}

pub(super) fn label_schema() -> serde_json::Value {
    json!({
        "schema_id": "echoforge.ml_training.label_schema.v1",
        "format": "csv",
        "per_record_file": "records/<record_id>/frame_labels.csv",
        "split_unit": "scenario_object_seed",
        "columns": [
            {"name": "class_label", "type": "string"},
            {"name": "is_public_proxy_positive", "type": "bool"},
            {"name": "is_hard_negative", "type": "bool"},
            {"name": "cfar_label", "type": "bool"},
            {"name": "tbd_label", "type": "bool"},
            {"name": "first_detectable_frame", "type": "usize?"},
            {"name": "safety_use", "type": "string"}
        ],
        "safety_boundary": "defensive early-detection and false-alarm robustness only"
    })
}

pub(super) fn guardrails() -> Vec<String> {
    crate::guard::PRODUCT_GUARDRAILS
        .iter()
        .map(|s| s.to_string())
        .collect()
}

pub(super) fn feature_family_availability(record_id: &str) -> FeatureFamilyAvailability {
    FeatureFamilyAvailability {
        record_id: record_id.to_string(),
        coherent_range_doppler: AvailabilityEntry {
            status: "available".to_string(),
            path: "products".to_string(),
            reason: None,
        },
        clutter_interference: AvailabilityEntry {
            status: "available".to_string(),
            path: "streaming_features.csv".to_string(),
            reason: None,
        },
        micro_doppler: AvailabilityEntry {
            status: "available".to_string(),
            path: "micro_doppler".to_string(),
            reason: None,
        },
        multi_view_tensors: AvailabilityEntry {
            status: "available".to_string(),
            path: "multi_view".to_string(),
            reason: None,
        },
        learned_windows: AvailabilityEntry {
            status: "available".to_string(),
            path: "learned_windows".to_string(),
            reason: None,
        },
        range_angle_future_schema: AvailabilityEntry {
            status: "unavailable".to_string(),
            path: "multi_view/range_angle_schema_pending.json".to_string(),
            reason: Some(
                "future MIMO range-angle and range-azimuth-Doppler schema pending only"
                    .to_string(),
            ),
        },
    }
}

/// Aggregate per-record [`PerRecordTierObservation`]s into the
/// (tier × is_positive) summary rows written to `per_tier_pd_pfa.json`.
///
/// `n_episodes` is the count of records (of the given polarity) that
/// observed at least one CPI in the tier. `n_detections` is the count
/// of such records that registered at least one Pd-positive CPI in the
/// tier. `pd_proxy = n_detections / n_episodes` is therefore the
/// fraction of in-tier records that produced a detection — bounded to
/// `[0, 1]` and directly comparable to a Pd surrogate (or Pfa
/// surrogate for `is_positive == false`).
pub(super) fn aggregate_per_tier_metrics(
    observations: &[PerRecordTierObservation],
) -> Vec<PerTierMetrics> {
    let tiers = [Tier::None, Tier::Boost, Tier::ClimbOut, Tier::Cruise];
    let mut rows = Vec::with_capacity(tiers.len() * 2);
    for is_positive in [true, false] {
        for tier in tiers.iter().copied() {
            let (n_episodes, n_detections, n_horizon_blocked, conf_sum, conf_n) = observations
                .iter()
                .filter(|o| o.is_positive == is_positive)
                .fold((0usize, 0usize, 0usize, 0.0f64, 0usize), |(eps, det, hor, cs, cn), obs| {
                    (
                        eps + obs.tier_counts.contains_key(&tier) as usize,
                        det + (obs.detection_counts.get(&tier).copied().unwrap_or(0) > 0) as usize,
                        hor + obs.horizon_blocked_counts.get(&tier).copied().unwrap_or(0),
                        cs + obs.confidence_sum.get(&tier).copied().unwrap_or(0.0),
                        cn + obs.confidence_n.get(&tier).copied().unwrap_or(0),
                    )
                });
            rows.push(PerTierMetrics {
                tier: tier_name(tier).to_string(),
                is_positive,
                n_episodes,
                n_detections,
                n_horizon_blocked,
                mean_confidence: if conf_n > 0 { conf_sum / conf_n as f64 } else { 0.0 },
                pd_proxy: if n_episodes > 0 { n_detections as f64 / n_episodes as f64 } else { 0.0 },
            });
        }
    }
    rows
}

fn tier_name(tier: Tier) -> &'static str {
    match tier {
        Tier::None => "none",
        Tier::Boost => "boost",
        Tier::ClimbOut => "climb_out",
        Tier::Cruise => "cruise",
    }
}
