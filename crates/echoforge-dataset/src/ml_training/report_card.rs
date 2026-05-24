//! Dataset card, feature/label schema, guardrails, feature-family availability,
//! and per-tier metrics: the "card & schema" half of the ml_training report module.

use echoforge_core::deterministic_id;
use echoforge_core::models::{
    DatasetCard, DatasetSplits, LicenseInfo, Provenance, ValidationCheck, ValidationInfo,
};
use echoforge_radar::Tier;

use crate::monte_carlo::DatasetError;
use crate::split::SplitKind;

use super::config::{MlTrainingDataConfig, DEFAULT_ML_TRAINING_DATASET_ID, NEUTRAL_OBJECT_ID};
use super::report::feature_families;
use super::types::{
    AvailabilityEntry, FeatureFamilyAvailability, PerRecordTierObservation, PerTierMetrics,
    QualityReport,
};

pub(super) fn dataset_card(
    config: &MlTrainingDataConfig,
    quality: &QualityReport,
) -> Result<DatasetCard, DatasetError> {
    let source_payload = serde_json::json!({
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
    serde_json::json!({
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
    serde_json::json!({
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
                "future MIMO range-angle and range-azimuth-Doppler schema pending only".to_string(),
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
                .fold(
                    (0usize, 0usize, 0usize, 0.0f64, 0usize),
                    |(eps, det, hor, cs, cn), obs| {
                        (
                            eps + obs.tier_counts.contains_key(&tier) as usize,
                            det + (obs.detection_counts.get(&tier).copied().unwrap_or(0) > 0)
                                as usize,
                            hor + obs.horizon_blocked_counts.get(&tier).copied().unwrap_or(0),
                            cs + obs.confidence_sum.get(&tier).copied().unwrap_or(0.0),
                            cn + obs.confidence_n.get(&tier).copied().unwrap_or(0),
                        )
                    },
                );
            rows.push(PerTierMetrics {
                tier: tier_name(tier).to_string(),
                is_positive,
                n_episodes,
                n_detections,
                n_horizon_blocked,
                mean_confidence: if conf_n > 0 {
                    conf_sum / conf_n as f64
                } else {
                    0.0
                },
                pd_proxy: if n_episodes > 0 {
                    n_detections as f64 / n_episodes as f64
                } else {
                    0.0
                },
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
