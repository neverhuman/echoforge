//! Extended manifest/report type definitions — split from types.rs for LOC compliance.
//! All structs here are `pub(super)` and accessible to ml_training sub-modules.

use std::collections::BTreeMap;

use echoforge_radar::BackendMode;
use serde::{Deserialize, Serialize};

use crate::monte_carlo::StageTiming;
use crate::split::SplitKind;
use super::types::MlRecordSummary;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct DatasetManifest {
    pub manifest_version: String,
    pub dataset_id: String,
    pub neutral_object_id: String,
    pub generated_at: String,
    pub root_seed: u64,
    pub records: Vec<MlRecordSummary>,
    pub frame_count: usize,
    pub frame_rate_hz: f64,
    pub time_window_s: f64,
    pub positive_fraction: f64,
    pub split_policy: String,
    pub feature_families: Vec<FeatureFamily>,
    pub artifacts: BTreeMap<String, String>,
    pub external_calibration_sources: Vec<ExternalCalibrationSource>,
    pub guardrails: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct FeatureFamily {
    pub id: String,
    pub description: String,
    pub artifact_pattern: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct ExternalCalibrationSource {
    pub id: String,
    pub title: String,
    pub url: String,
    pub role: String,
    pub local_data_default: String,
    pub license_notes: String,
    pub expected_feature_mappings: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct FeatureFamilyAvailability {
    pub record_id: String,
    pub coherent_range_doppler: AvailabilityEntry,
    pub clutter_interference: AvailabilityEntry,
    pub micro_doppler: AvailabilityEntry,
    pub multi_view_tensors: AvailabilityEntry,
    pub learned_windows: AvailabilityEntry,
    pub range_angle_future_schema: AvailabilityEntry,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct AvailabilityEntry {
    pub status: String,
    pub path: String,
    pub reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct MicroDopplerDescriptors {
    pub record_id: String,
    pub peak_hz_proxy: f32,
    pub bandwidth_hz_proxy: f32,
    pub weighted_spectrum_entropy: f32,
    pub cepstrum_peak: f32,
    pub cadence_velocity_peak: f32,
    pub representation_note: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct LearnedWindowManifest {
    pub record_id: String,
    pub windows: Vec<LearnedWindowEntry>,
    pub feature_order: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct LearnedWindowEntry {
    pub window_frames: usize,
    pub path: String,
    pub shape: Vec<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct RuntimeReport {
    pub requested_backend: BackendMode,
    pub selected_backend: String,
    pub kernel_backend: String,
    pub gpu_stage_recovery: Option<String>,
    pub gpu_available: bool,
    pub gpu_usable: bool,
    pub gpu_constrained: bool,
    pub gpu_min_free_memory_mb: u64,
    pub gpu_free_memory_mb: Option<u64>,
    pub logical_cores: usize,
    pub recommended_worker_budget: usize,
    pub worker_count: usize,
    pub worker_cap: usize,
    pub records: usize,
    pub frame_count: usize,
    pub stage_timings: Vec<StageTiming>,
    pub total_elapsed_ns: u64,
    pub throughput_records_per_sec: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct QualityReport {
    pub dataset_id: String,
    pub records: usize,
    pub positive_records: usize,
    pub positive_fraction_actual: f64,
    pub hard_negative_family_counts: BTreeMap<String, usize>,
    pub hard_negative_family_coverage: usize,
    pub minimum_hard_negative_families: usize,
    pub feature_families_checked: Vec<String>,
    pub all_records_have_required_artifacts: bool,
    pub finite_feature_values: bool,
    pub split_counts: BTreeMap<SplitKind, usize>,
    pub limitations: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct NormalizationStats {
    pub dataset_id: String,
    pub source: String,
    pub columns: BTreeMap<String, ColumnStats>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct ColumnStats {
    pub mean: f64,
    pub stddev: f64,
    pub min: f64,
    pub max: f64,
}
