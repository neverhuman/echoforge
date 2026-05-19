use std::collections::BTreeMap;
use std::path::PathBuf;

use echoforge_radar::{BackendMode, RuntimePlan};
use serde::{Deserialize, Serialize};

use super::{DEFAULT_CAMPAIGN_OUTPUT, DEFAULT_CAMPAIGN_REQUEST_ID};

// ── Config ────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct CampaignConfig {
    pub campaign: String,
    pub records: usize,
    pub shahed_min: usize,
    pub shahed_target: usize,
    pub time_window_s: f64,
    pub frame_rate_hz: f64,
    pub backend: BackendMode,
    pub workers: Option<usize>,
    pub trigger_confidence: f32,
    pub seed: u64,
    pub generated_at: String,
    pub output_dir: PathBuf,
    pub progress: bool,
}

impl CampaignConfig {
    pub fn shahed_public_proxy_default() -> Self {
        Self {
            campaign: DEFAULT_CAMPAIGN_REQUEST_ID.to_string(),
            records: 1_000,
            shahed_min: 50,
            shahed_target: 80,
            time_window_s: 90.0,
            frame_rate_hz: 2.0,
            backend: BackendMode::Auto,
            workers: Some(40),
            trigger_confidence: 0.80,
            seed: 20_260_518_136,
            generated_at: "2026-05-18T00:00:00Z".to_string(),
            output_dir: PathBuf::from(DEFAULT_CAMPAIGN_OUTPUT),
            progress: true,
        }
    }
}

// ── Report ────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CampaignReport {
    pub output_dir: PathBuf,
    pub campaign_request_id: String,
    pub neutral_campaign_id: String,
    pub records: usize,
    pub shahed_positive_records: usize,
    pub worker_count: usize,
    pub runtime: RuntimePlan,
    pub progress_enabled: bool,
    pub campaign_manifest_path: PathBuf,
    pub class_balance_path: PathBuf,
    pub dataset_card_path: PathBuf,
    pub runtime_report_path: PathBuf,
    pub benchmark_report_path: PathBuf,
}

// ── Class taxonomy ────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CampaignBucket {
    PositiveOwaDeltaPusher,
    SmallUav,
    Biological,
    WindborneDebris,
    InfrastructureTerrain,
    GroundMoversMultipath,
    WeatherRfiSensorArtifacts,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CampaignClass {
    pub id: String,
    pub display_name: String,
    pub target_family: String,
    pub bucket: CampaignBucket,
    pub hard_negative_family: String,
    pub is_shahed_public_proxy: bool,
    pub is_hard_negative: bool,
}

// ── Record planning ───────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CampaignRecordPlan {
    pub index: usize,
    pub record_id: String,
    pub seed: u64,
    pub class: CampaignClass,
}

// ── Per-frame data ────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrameFeature {
    pub frame_index: usize,
    pub time_s: f64,
    pub cpi_pulses: usize,
    pub range_m: f64,
    pub range_rate_mps: f64,
    pub radial_velocity_mps: f64,
    pub altitude_m: f64,
    pub snr_db: f32,
    pub snr_trend_db: f32,
    pub doppler_spread_hz: f32,
    pub blob_area_bins: f32,
    pub micro_doppler_modulation: f32,
    pub track_persistence_s: f32,
    pub clutter_pressure: f32,
    pub rfi_pressure: f32,
    pub receiver_dropout: bool,
    pub phase_noise_rad: f32,
    pub amplitude_scintillation: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrameLabel {
    pub frame_index: usize,
    pub time_s: f64,
    pub target_family: String,
    pub is_shahed_public_proxy: bool,
    pub is_hard_negative: bool,
    pub hard_negative_family: String,
    pub scenario_phase: String,
    pub first_detectable_frame: Option<usize>,
    pub first_model_trigger_frame: Option<usize>,
    pub confidence_threshold: f32,
}

// ── Detection state ───────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DetectionState {
    pub model_id: String,
    pub frame_index: usize,
    pub confidence: f32,
    pub class_probabilities: BTreeMap<String, f32>,
    pub first_trigger_event: Option<FirstTriggerEvent>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FirstTriggerEvent {
    pub model_id: String,
    pub frame_index: usize,
    pub time_s: f64,
    pub confidence: f32,
    pub threshold: f32,
    pub consecutive_frames: usize,
}

// ── Internal record types ─────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct ModelPredictionRow {
    pub record_id: String,
    pub model_id: String,
    pub frame_index: usize,
    pub time_s: f64,
    pub confidence: f32,
    pub class_probabilities_json: String,
    pub triggered_on_this_frame: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct RecordTruthMetadata {
    pub record_id: String,
    pub neutral_object_id: String,
    pub target_family: String,
    pub is_shahed_public_proxy: bool,
    pub is_hard_negative: bool,
    pub hard_negative_family: String,
    pub source_dossier_ref: String,
    pub dimensions_m: DimensionsSample,
    pub rcs_dbsm_proxy: f64,
    pub cruise_speed_mps: f64,
    pub cpi_pulses: usize,
    pub frame_count: usize,
    pub time_window_s: f64,
    pub first_detectable_frame: Option<usize>,
    pub confidence_threshold: f32,
    pub guardrails: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct DimensionsSample {
    pub length: f64,
    pub wingspan: f64,
    pub height: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct CampaignRecordSummary {
    pub record_id: String,
    pub record_index: usize,
    pub target_family: String,
    pub class_id: String,
    pub bucket: CampaignBucket,
    pub is_shahed_public_proxy: bool,
    pub is_hard_negative: bool,
    pub hard_negative_family: String,
    pub first_detectable_frame: Option<usize>,
    pub first_model_trigger_frame: Option<usize>,
    pub cpi_pulses: usize,
    pub tensor_dir: String,
    pub frame_labels_path: String,
    pub truth_metadata_path: String,
    pub model_predictions_path: String,
    pub detector_events_path: String,
    pub max_confidence_by_model: BTreeMap<String, f32>,
    pub first_trigger_by_model: BTreeMap<String, Option<usize>>,
}

#[derive(Debug, Clone)]
pub(super) struct CampaignRecordOutput {
    pub summary: CampaignRecordSummary,
    pub events: Vec<FirstTriggerEvent>,
    pub predictions: Vec<ModelPredictionRow>,
}

// ── Manifest (written to disk at end of campaign) ─────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct CampaignManifest {
    pub manifest_version: String,
    pub campaign_request_id: String,
    pub neutral_campaign_id: String,
    pub generated_at: String,
    pub root_seed: u64,
    pub records: Vec<CampaignRecordSummary>,
    pub frame_count: usize,
    pub frame_rate_hz: f64,
    pub time_window_s: f64,
    pub class_balance: ClassBalanceReport,
    pub runtime_report_path: String,
    pub benchmark_report_path: String,
    pub dataset_card_path: String,
    pub source_dossier_ref: String,
    pub guardrails: Vec<String>,
}

// ── Evaluation types ──────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ClassBalanceReport {
    pub total_records: usize,
    pub shahed_positive_records: usize,
    pub shahed_min_required: usize,
    pub meets_shahed_min: bool,
    pub bucket_counts: BTreeMap<CampaignBucket, usize>,
    pub target_family_counts: BTreeMap<String, usize>,
    pub hard_negative_family_counts: BTreeMap<String, usize>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelEvaluationReport {
    pub model_id: String,
    pub positive_records: usize,
    pub negative_records: usize,
    pub pd: f64,
    pub pfa: f64,
    pub missed_positive_records: Vec<String>,
    pub false_alarm_by_hard_negative_family: BTreeMap<String, usize>,
    pub mean_first_detection_latency_frames: Option<f64>,
    pub roc_points: Vec<CurvePoint>,
    pub pr_points: Vec<CurvePoint>,
    pub confidence_calibration_bins: Vec<CalibrationBin>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CurvePoint {
    pub threshold: f32,
    pub x: f64,
    pub y: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CalibrationBin {
    pub bin_start: f32,
    pub bin_end: f32,
    pub records: usize,
    pub positive_fraction: f64,
}
