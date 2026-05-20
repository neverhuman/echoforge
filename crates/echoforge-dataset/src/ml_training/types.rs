//! Internal data types shared across ml_training sub-modules.

use std::collections::{BTreeMap, HashMap};
use std::path::PathBuf;

use echoforge_radar::{PhaseTieredDecision, RuntimePlan, Tier};
use serde::{Deserialize, Serialize};

use crate::split::SplitKind;

// ---------------------------------------------------------------------------
// Public-facing types (re-exported from mod.rs)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MlTrainingDataReport {
    pub output_dir: PathBuf,
    pub dataset_id: String,
    pub records: usize,
    pub positive_records: usize,
    pub hard_negative_families: usize,
    pub frame_count: usize,
    pub split_counts: BTreeMap<SplitKind, usize>,
    pub worker_count: usize,
    pub runtime: RuntimePlan,
    pub dataset_manifest_path: PathBuf,
    pub dataset_card_path: PathBuf,
    pub split_manifest_path: PathBuf,
    pub records_path: PathBuf,
    pub features_path: PathBuf,
    pub label_schema_path: PathBuf,
    pub feature_schema_path: PathBuf,
    pub normalization_stats_path: PathBuf,
    pub quality_report_path: PathBuf,
    pub runtime_report_path: PathBuf,
    /// V3 unified-path report: per-tier (boost/climb/cruise) Pd/Pfa
    /// proxy summary derived from `PhaseTieredDetector::evaluate_cpi`
    /// against each synthesised episode.
    pub per_tier_pd_pfa_path: PathBuf,
}

/// V3 per-tier metrics row (one per `Tier` × `is_positive` slot).
/// Aggregated across episodes; `pd_proxy` is `n_detections /
/// n_episodes` for positive records (Pd surrogate), and
/// `n_detections / n_episodes` for confusers (Pfa surrogate). The
/// numerator counts CPIs where the phase-tiered detector returned a
/// non-`None` tier with confidence >= 0.5 and was not horizon-blocked.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerTierMetrics {
    pub tier: String,
    pub is_positive: bool,
    pub n_episodes: usize,
    pub n_detections: usize,
    pub n_horizon_blocked: usize,
    pub mean_confidence: f64,
    pub pd_proxy: f64,
}

// ---------------------------------------------------------------------------
// Internal types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct MlClass {
    pub class_id: String,
    pub display_name: String,
    pub target_family: String,
    pub hard_negative_family: String,
    pub is_public_proxy_positive: bool,
    pub is_hard_negative: bool,
}

#[derive(Debug, Clone)]
pub(super) struct MlRecordPlan {
    pub record_index: usize,
    pub record_id: String,
    pub scenario_seed: u64,
    pub object_seed: u64,
    pub split: SplitKind,
    pub class: MlClass,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct MlRecordSummary {
    pub record_id: String,
    pub record_index: usize,
    pub split: SplitKind,
    pub class_id: String,
    pub target_family: String,
    pub is_public_proxy_positive: bool,
    pub is_hard_negative: bool,
    pub hard_negative_family: String,
    pub scenario_seed: u64,
    pub object_seed: u64,
    pub frame_count: usize,
    pub cpi_pulses: usize,
    pub tensor_dir: String,
    pub streaming_features_path: String,
    pub frame_labels_path: String,
    pub truth_metadata_path: String,
    pub detector_events_path: String,
    pub feature_family_availability_path: String,
    pub micro_doppler_dir: String,
    pub multi_view_dir: String,
    pub learned_windows_dir: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct MlFeatureSummaryRow {
    pub record_id: String,
    pub split: SplitKind,
    pub target_family: String,
    pub hard_negative_family: String,
    pub is_public_proxy_positive: bool,
    pub mean_snr_db: f32,
    pub max_snr_db: f32,
    pub mean_doppler_scr: f32,
    pub mean_rfi_pressure: f32,
    pub dropout_fraction: f32,
    pub mean_micro_doppler_energy: f32,
    pub micro_doppler_peak_hz_proxy: f32,
    pub micro_doppler_bandwidth_hz_proxy: f32,
    pub mean_track_score: f32,
    pub cfar_detection_fraction: f32,
    pub first_detectable_frame: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct MlFrameFeatureRow {
    pub record_id: String,
    pub frame_index: usize,
    pub time_s: f64,
    pub cpi_pulses: usize,
    pub range_m: f64,
    pub radial_velocity_mps: f64,
    pub altitude_m: f64,
    pub snr_db: f32,
    pub cfar_statistic: f32,
    pub cfar_threshold: f32,
    pub cfar_detected: bool,
    pub tbd_track_score: f32,
    pub local_noise_floor_db: f32,
    pub doppler_scr: f32,
    pub rfi_pressure: f32,
    pub dropout_fraction: f32,
    pub phase_impairment_rad: f32,
    pub amplitude_impairment: f32,
    pub micro_doppler_energy: f32,
    pub micro_doppler_peak_hz_proxy: f32,
    pub micro_doppler_bandwidth_hz_proxy: f32,
    pub stft_energy: f32,
    pub weighted_spectrum_peak: f32,
    pub cepstrum_peak: f32,
    pub cadence_velocity_peak: f32,
    pub range_time_energy: f32,
    pub doppler_time_energy: f32,
    pub range_doppler_time_energy: f32,
    pub normalized_snr: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct MlFrameLabelRow {
    pub record_id: String,
    pub frame_index: usize,
    pub time_s: f64,
    pub split: SplitKind,
    pub class_label: String,
    pub is_public_proxy_positive: bool,
    pub is_hard_negative: bool,
    pub hard_negative_family: String,
    pub cfar_label: bool,
    pub tbd_label: bool,
    pub first_detectable_frame: Option<usize>,
    pub safety_use: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct MlDetectorEvent {
    pub record_id: String,
    pub detector_id: String,
    pub frame_index: usize,
    pub time_s: f64,
    pub score: f32,
    pub threshold: f32,
    pub event_kind: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct MlTruthMetadata {
    pub record_id: String,
    pub neutral_object_id: String,
    pub dataset_id: String,
    pub split: SplitKind,
    pub class_id: String,
    pub hard_negative_family: String,
    pub target_family: String,
    pub is_public_proxy_positive: bool,
    pub is_hard_negative: bool,
    pub scenario_seed: u64,
    pub object_seed: u64,
    pub dimensions_m: DimensionsSample,
    pub rcs_dbsm_proxy: f64,
    pub speed_mps_proxy: f64,
    pub time_window_s: f64,
    pub frame_rate_hz: f64,
    pub frame_count: usize,
    pub guardrails: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(super) struct DimensionsSample {
    pub length: f64,
    pub wingspan: f64,
    pub height: f64,
}

#[derive(Debug, Clone)]
pub(super) struct MlEnvelope {
    pub dimensions_m: DimensionsSample,
    pub rcs_dbsm: f64,
    pub speed_mps: f64,
    pub initial_range_m: f64,
    pub radial_velocity_mps: f64,
    pub altitude_m: f64,
    pub base_snr_db: f32,
    pub micro_peak_hz: f32,
    pub micro_bandwidth_hz: f32,
    pub clutter_pressure: f32,
    pub rfi_pressure: f32,
    pub dropout_probability: f32,
    pub phase_impairment_rad: f32,
    pub amplitude_impairment: f32,
}

#[derive(Debug, Clone)]
pub(super) struct RecordOutput {
    pub summary: MlRecordSummary,
    pub features: MlFeatureSummaryRow,
    pub split: SplitManifestRow,
    /// V3 unified-path per-record aggregate from
    /// `PhaseTieredDetector::evaluate_cpi`. Records the tier counts /
    /// detection counts that feed the per-tier Pd/Pfa report.
    pub per_tier_observation: PerRecordTierObservation,
}

/// Per-record summary of `PhaseTieredDetector` evaluations over the
/// synthesised episode. Aggregated across records to produce
/// `per_tier_pd_pfa.json`. `HashMap` is used because `Tier` derives
/// `Hash` (not `Ord`) in the radar crate.
#[derive(Debug, Clone)]
pub(super) struct PerRecordTierObservation {
    pub is_positive: bool,
    /// One entry per CPI per tier classification. `Tier::None` is
    /// included so reviewers see how often the arbiter never latched.
    pub tier_counts: HashMap<Tier, usize>,
    pub detection_counts: HashMap<Tier, usize>,
    pub horizon_blocked_counts: HashMap<Tier, usize>,
    pub confidence_sum: HashMap<Tier, f64>,
    pub confidence_n: HashMap<Tier, usize>,
}

impl PerRecordTierObservation {
    pub fn new(is_positive: bool) -> Self {
        Self {
            is_positive,
            tier_counts: HashMap::new(),
            detection_counts: HashMap::new(),
            horizon_blocked_counts: HashMap::new(),
            confidence_sum: HashMap::new(),
            confidence_n: HashMap::new(),
        }
    }

    pub fn record(&mut self, decision: &PhaseTieredDecision) {
        *self.tier_counts.entry(decision.tier).or_insert(0) += 1;
        if decision.horizon_blocked {
            *self
                .horizon_blocked_counts
                .entry(decision.tier)
                .or_insert(0) += 1;
        }
        // Pd-proxy: detector latched a tier (not `None`) and confidence
        // >= 0.5 and not horizon-blocked.
        if !matches!(decision.tier, Tier::None)
            && decision.confidence >= 0.5
            && !decision.horizon_blocked
        {
            *self.detection_counts.entry(decision.tier).or_insert(0) += 1;
        }
        *self.confidence_sum.entry(decision.tier).or_insert(0.0) += decision.confidence as f64;
        *self.confidence_n.entry(decision.tier).or_insert(0) += 1;
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct SplitManifestRow {
    pub record_id: String,
    pub split: SplitKind,
    pub split_key_kind: String,
    pub split_key: String,
    pub scenario_seed: u64,
    pub object_seed: u64,
    pub class_id: String,
    pub target_family: String,
    pub hard_negative_family: String,
}

// Re-export shared PRNG from crate-level module
pub(super) use crate::rng::SplitMix64;

// Re-export manifest/report types from types_ext (split for LOC compliance).
pub(super) use super::types_ext::{
    AvailabilityEntry, ColumnStats, DatasetManifest, ExternalCalibrationSource, FeatureFamily,
    FeatureFamilyAvailability, LearnedWindowEntry, LearnedWindowManifest, MicroDopplerDescriptors,
    NormalizationStats, QualityReport, RuntimeReport,
};
