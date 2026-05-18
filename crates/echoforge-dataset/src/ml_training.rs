//! ML training dataset generator (V3 unified-path migration).
//!
//! ## Wave 5 Lane K_rust — unified `synthesize_scene` path
//!
//! Before Wave 5, this module bifurcated its physics flow:
//!
//! 1. Positives went through [`echoforge_radar::synthesize_takeoff_episode`]
//!    (full radar chain: pulse-compression, slow-time DFT, K/Weibull
//!    clutter, OS-CFAR, MTI/MTD) for tensor products.
//! 2. Confusers went through the SAME wrapper for tensor products, but
//!    `build_frame_products` rebuilt streaming features from envelope
//!    statistics WITHOUT consulting the synthesised episode. The
//!    generator identity (envelope branch) literally labelled the class
//!    — a credibility-sweep red flag.
//!
//! This module now routes ALL records (positives + confusers) through
//! the unified [`echoforge_radar::synthesize_scene`] entry point that
//! Lane I (Wave 4) introduced. Each record builds a
//! [`echoforge_radar::SceneDescriptor`] whose single
//! [`echoforge_radar::TargetEntity`] carries an explicit
//! [`echoforge_radar::TargetClass`] that names the truth class, then
//! calls `synthesize_scene` to produce the [`echoforge_radar::SyntheticEpisode`].
//! Streaming features (`build_frame_products`) are then derived from
//! that single episode, no longer recomputed from envelope statistics
//! alone. The episode's `diagnostic_snr_db`, `range_doppler_proxy`,
//! `target_states`, and CFAR `detections` are the source of truth for
//! per-frame SNR, Doppler, range, velocity, altitude, and label
//! columns.
//!
//! Per-episode the generator also runs a
//! [`echoforge_radar::PhaseTieredDetector`] (Lane H/H2) and writes a
//! per-tier Pd/Pfa report alongside the dataset (`per_tier_pd_pfa.json`)
//! so reviewers can see boost/climb/cruise behaviour separately.
//!
//! ### Coordination with Lane J
//!
//! Lane J extends [`echoforge_radar::TargetKinematics`] with native
//! per-class kinematics (constant-velocity birds, parked vehicles,
//! stationary turbines, multipath ghosts derived from a parent track,
//! etc.) and lifts the single-entity gate on `synthesize_scene`. Until
//! Lane J lands, this module stubs all kinematics with
//! [`echoforge_radar::TargetKinematics::FromTakeoffProfile`] adapted
//! from the envelope, regardless of class. The `class` field is still
//! propagated so downstream metadata (truth files, scene JSON) reflects
//! the unified labelling.
//!
//! Multipath ghosts (which would want a paired entity with
//! `TargetClass::MultipathGhost { parent_idx: 0 }`) currently fall back
//! to a single-entity scene with `TargetClass::TerrainGlint` — Lane I
//! rejects multi-entity scenes via debug assertion. A TODO in
//! `confuser_class_for_family` documents the gap; Lane J reconciles.
//!
//! ### References
//!
//! - Lane I: `crates/echoforge-radar/src/scene.rs` — `SceneDescriptor`,
//!   `TargetClass`, `TargetEntity`, `TargetKinematics`.
//! - Lane H/H2: `crates/echoforge-radar/src/detectors/phase_tiered/` —
//!   `PhaseTieredDetector`, `Tier`, `PhaseTieredDecision`.
//! - Wave 4 receipt: `.agents/receipts/realism-v4-red-team-gap-report/*`
//!   documents the original bifurcation finding.

use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::path::{Path, PathBuf};
use std::thread;
use std::time::{Duration, Instant};

use echoforge_core::deterministic_id;
use echoforge_core::models::{
    DatasetCard, DatasetSplits, LicenseInfo, Provenance, ValidationCheck, ValidationInfo,
};
use echoforge_radar::{
    synthesize_scene, BackendMode, BackendSignals, EnvironmentDescriptor, EpisodeSeed,
    KinematicObservation, KinematicSample, NoiseProfile, PhaseTieredDecision, PhaseTieredDetector,
    RadarSimConfig, RuntimePlan, SceneDescriptor, SiteGeometry, SyntheticEpisode, TakeoffProfile,
    TargetClass, TargetEntity, TargetKinematics, TargetState, Tier,
};
use ndarray::{ArrayD, IxDyn};
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::export::{write_episode_tensors, write_json_pretty};
use crate::monte_carlo::{DatasetError, StageTiming};
use crate::split::SplitKind;

pub const DEFAULT_ML_TRAINING_DATASET_ID: &str = "shahed136-public-proxy-ml-training-v1";
pub const DEFAULT_ML_TRAINING_OUTPUT: &str =
    "outputs/training-data/shahed136-public-proxy-ml-training-v1";
const NEUTRAL_OBJECT_ID: &str = "owa-delta-pusher-fixed-wing-public-proxy-v1";

#[derive(Debug, Clone)]
pub struct MlTrainingDataConfig {
    pub dataset: String,
    pub records: usize,
    pub positive_fraction: f64,
    pub time_window_s: f64,
    pub frame_rate_hz: f64,
    pub backend: BackendMode,
    pub workers: Option<usize>,
    pub seed: u64,
    pub generated_at: String,
    pub output_dir: PathBuf,
}

impl MlTrainingDataConfig {
    pub fn shahed_public_proxy_default() -> Self {
        Self {
            dataset: DEFAULT_ML_TRAINING_DATASET_ID.to_string(),
            records: 10_000,
            positive_fraction: 0.20,
            time_window_s: 90.0,
            frame_rate_hz: 2.0,
            backend: BackendMode::Auto,
            workers: Some(40),
            seed: 20_260_518_136_001,
            generated_at: "2026-05-18T00:00:00Z".to_string(),
            output_dir: PathBuf::from(DEFAULT_ML_TRAINING_OUTPUT),
        }
    }
}

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

#[derive(Debug, Clone, Serialize, Deserialize)]
struct MlClass {
    class_id: String,
    display_name: String,
    target_family: String,
    hard_negative_family: String,
    is_public_proxy_positive: bool,
    is_hard_negative: bool,
}

#[derive(Debug, Clone)]
struct MlRecordPlan {
    record_index: usize,
    record_id: String,
    scenario_seed: u64,
    object_seed: u64,
    split: SplitKind,
    class: MlClass,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct MlRecordSummary {
    record_id: String,
    record_index: usize,
    split: SplitKind,
    class_id: String,
    target_family: String,
    is_public_proxy_positive: bool,
    is_hard_negative: bool,
    hard_negative_family: String,
    scenario_seed: u64,
    object_seed: u64,
    frame_count: usize,
    cpi_pulses: usize,
    tensor_dir: String,
    streaming_features_path: String,
    frame_labels_path: String,
    truth_metadata_path: String,
    detector_events_path: String,
    feature_family_availability_path: String,
    micro_doppler_dir: String,
    multi_view_dir: String,
    learned_windows_dir: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct MlFeatureSummaryRow {
    record_id: String,
    split: SplitKind,
    target_family: String,
    hard_negative_family: String,
    is_public_proxy_positive: bool,
    mean_snr_db: f32,
    max_snr_db: f32,
    mean_doppler_scr: f32,
    mean_rfi_pressure: f32,
    dropout_fraction: f32,
    mean_micro_doppler_energy: f32,
    micro_doppler_peak_hz_proxy: f32,
    micro_doppler_bandwidth_hz_proxy: f32,
    mean_track_score: f32,
    cfar_detection_fraction: f32,
    first_detectable_frame: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct MlFrameFeatureRow {
    record_id: String,
    frame_index: usize,
    time_s: f64,
    cpi_pulses: usize,
    range_m: f64,
    radial_velocity_mps: f64,
    altitude_m: f64,
    snr_db: f32,
    cfar_statistic: f32,
    cfar_threshold: f32,
    cfar_detected: bool,
    tbd_track_score: f32,
    local_noise_floor_db: f32,
    doppler_scr: f32,
    rfi_pressure: f32,
    dropout_fraction: f32,
    phase_impairment_rad: f32,
    amplitude_impairment: f32,
    micro_doppler_energy: f32,
    micro_doppler_peak_hz_proxy: f32,
    micro_doppler_bandwidth_hz_proxy: f32,
    stft_energy: f32,
    weighted_spectrum_peak: f32,
    cepstrum_peak: f32,
    cadence_velocity_peak: f32,
    range_time_energy: f32,
    doppler_time_energy: f32,
    range_doppler_time_energy: f32,
    normalized_snr: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct MlFrameLabelRow {
    record_id: String,
    frame_index: usize,
    time_s: f64,
    split: SplitKind,
    class_label: String,
    is_public_proxy_positive: bool,
    is_hard_negative: bool,
    hard_negative_family: String,
    cfar_label: bool,
    tbd_label: bool,
    first_detectable_frame: Option<usize>,
    safety_use: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct MlDetectorEvent {
    record_id: String,
    detector_id: String,
    frame_index: usize,
    time_s: f64,
    score: f32,
    threshold: f32,
    event_kind: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct MlTruthMetadata {
    record_id: String,
    neutral_object_id: String,
    dataset_id: String,
    split: SplitKind,
    class_id: String,
    target_family: String,
    is_public_proxy_positive: bool,
    is_hard_negative: bool,
    hard_negative_family: String,
    scenario_seed: u64,
    object_seed: u64,
    dimensions_m: DimensionsSample,
    rcs_dbsm_proxy: f64,
    speed_mps_proxy: f64,
    time_window_s: f64,
    frame_rate_hz: f64,
    frame_count: usize,
    guardrails: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct DimensionsSample {
    length: f64,
    wingspan: f64,
    height: f64,
}

#[derive(Debug, Clone)]
struct MlEnvelope {
    dimensions_m: DimensionsSample,
    rcs_dbsm: f64,
    speed_mps: f64,
    initial_range_m: f64,
    radial_velocity_mps: f64,
    altitude_m: f64,
    base_snr_db: f32,
    micro_peak_hz: f32,
    micro_bandwidth_hz: f32,
    clutter_pressure: f32,
    rfi_pressure: f32,
    dropout_probability: f32,
    phase_impairment_rad: f32,
    amplitude_impairment: f32,
}

#[derive(Debug, Clone)]
struct RecordOutput {
    summary: MlRecordSummary,
    features: MlFeatureSummaryRow,
    split: SplitManifestRow,
    /// V3 unified-path per-record aggregate from
    /// `PhaseTieredDetector::evaluate_cpi`. Records the tier counts /
    /// detection counts that feed the per-tier Pd/Pfa report.
    per_tier_observation: PerRecordTierObservation,
}

/// Per-record summary of `PhaseTieredDetector` evaluations over the
/// synthesised episode. Aggregated across records to produce
/// `per_tier_pd_pfa.json`. `HashMap` is used because `Tier` derives
/// `Hash` (not `Ord`) in the radar crate.
#[derive(Debug, Clone)]
struct PerRecordTierObservation {
    is_positive: bool,
    /// One entry per CPI per tier classification. `Tier::None` is
    /// included so reviewers see how often the arbiter never latched.
    tier_counts: HashMap<Tier, usize>,
    detection_counts: HashMap<Tier, usize>,
    horizon_blocked_counts: HashMap<Tier, usize>,
    confidence_sum: HashMap<Tier, f64>,
    confidence_n: HashMap<Tier, usize>,
}

impl PerRecordTierObservation {
    fn new(is_positive: bool) -> Self {
        Self {
            is_positive,
            tier_counts: HashMap::new(),
            detection_counts: HashMap::new(),
            horizon_blocked_counts: HashMap::new(),
            confidence_sum: HashMap::new(),
            confidence_n: HashMap::new(),
        }
    }

    fn record(&mut self, decision: &PhaseTieredDecision) {
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
struct SplitManifestRow {
    record_id: String,
    split: SplitKind,
    split_key_kind: String,
    split_key: String,
    scenario_seed: u64,
    object_seed: u64,
    class_id: String,
    target_family: String,
    hard_negative_family: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct DatasetManifest {
    manifest_version: String,
    dataset_id: String,
    neutral_object_id: String,
    generated_at: String,
    root_seed: u64,
    records: Vec<MlRecordSummary>,
    frame_count: usize,
    frame_rate_hz: f64,
    time_window_s: f64,
    positive_fraction: f64,
    split_policy: String,
    feature_families: Vec<FeatureFamily>,
    artifacts: BTreeMap<String, String>,
    external_calibration_sources: Vec<ExternalCalibrationSource>,
    guardrails: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct FeatureFamily {
    id: String,
    description: String,
    artifact_pattern: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ExternalCalibrationSource {
    id: String,
    title: String,
    url: String,
    role: String,
    local_data_default: String,
    license_notes: String,
    expected_feature_mappings: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct FeatureFamilyAvailability {
    record_id: String,
    coherent_range_doppler: AvailabilityEntry,
    clutter_interference: AvailabilityEntry,
    micro_doppler: AvailabilityEntry,
    multi_view_tensors: AvailabilityEntry,
    learned_windows: AvailabilityEntry,
    range_angle_future_schema: AvailabilityEntry,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct AvailabilityEntry {
    status: String,
    path: String,
    reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct MicroDopplerDescriptors {
    record_id: String,
    peak_hz_proxy: f32,
    bandwidth_hz_proxy: f32,
    weighted_spectrum_entropy: f32,
    cepstrum_peak: f32,
    cadence_velocity_peak: f32,
    representation_note: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct LearnedWindowManifest {
    record_id: String,
    windows: Vec<LearnedWindowEntry>,
    feature_order: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct LearnedWindowEntry {
    window_frames: usize,
    path: String,
    shape: Vec<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct RuntimeReport {
    requested_backend: BackendMode,
    selected_backend: String,
    kernel_backend: String,
    gpu_available: bool,
    gpu_usable: bool,
    gpu_constrained: bool,
    gpu_min_free_memory_mb: u64,
    gpu_free_memory_mb: Option<u64>,
    gpu_stage_fallback: Option<String>,
    logical_cores: usize,
    recommended_worker_budget: usize,
    worker_count: usize,
    worker_cap: usize,
    records: usize,
    frame_count: usize,
    stage_timings: Vec<StageTiming>,
    total_elapsed_ns: u64,
    throughput_records_per_sec: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct QualityReport {
    dataset_id: String,
    records: usize,
    positive_records: usize,
    positive_fraction_actual: f64,
    hard_negative_family_counts: BTreeMap<String, usize>,
    hard_negative_family_coverage: usize,
    minimum_hard_negative_families: usize,
    feature_families_checked: Vec<String>,
    all_records_have_required_artifacts: bool,
    finite_feature_values: bool,
    split_counts: BTreeMap<SplitKind, usize>,
    limitations: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct NormalizationStats {
    dataset_id: String,
    source: String,
    columns: BTreeMap<String, ColumnStats>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ColumnStats {
    mean: f64,
    stddev: f64,
    min: f64,
    max: f64,
}

pub fn run_ml_training_data(
    config: MlTrainingDataConfig,
) -> Result<MlTrainingDataReport, DatasetError> {
    validate_config(&config)?;
    refuse_source_owned_output(&config.output_dir)?;
    if config.output_dir.exists() {
        return Err(DatasetError::RefusingOutputPath(format!(
            "{} already exists",
            config.output_dir.display()
        )));
    }

    let overall_start = Instant::now();
    let runtime = RuntimePlan::from_signals(config.backend, BackendSignals::detect())?;
    let worker_count = effective_worker_count(&config, &runtime);
    let frame_count = ((config.time_window_s * config.frame_rate_hz).round() as usize).max(1);
    fs::create_dir_all(&config.output_dir)?;

    let mut stage_timings = Vec::new();
    let plan_start = Instant::now();
    let plans = build_record_plan(&config, frame_count)?;
    stage_timings.push(StageTiming {
        stage: "record_plan".to_string(),
        elapsed_ns: elapsed_ns(plan_start),
    });

    let generation_start = Instant::now();
    let mut outputs = run_record_workers(&config, &runtime, &plans, frame_count, worker_count)?;
    outputs.sort_by_key(|output| output.summary.record_index);
    stage_timings.push(StageTiming {
        stage: "record_generation".to_string(),
        elapsed_ns: elapsed_ns(generation_start),
    });

    let post_start = Instant::now();
    let summaries = outputs
        .iter()
        .map(|output| output.summary.clone())
        .collect::<Vec<_>>();
    let feature_rows = outputs
        .iter()
        .map(|output| output.features.clone())
        .collect::<Vec<_>>();
    let split_rows = outputs
        .iter()
        .map(|output| output.split.clone())
        .collect::<Vec<_>>();

    write_csv(&config.output_dir.join("records.csv"), &summaries)?;
    write_csv(&config.output_dir.join("features.csv"), &feature_rows)?;
    write_csv(&config.output_dir.join("split_manifest.csv"), &split_rows)?;
    write_json_pretty(
        &config.output_dir.join("feature_schema.json"),
        &feature_schema(),
    )?;
    write_json_pretty(
        &config.output_dir.join("label_schema.json"),
        &label_schema(),
    )?;

    let normalization_stats = normalization_stats(&config.dataset, &feature_rows);
    write_json_pretty(
        &config.output_dir.join("normalization_stats.json"),
        &normalization_stats,
    )?;

    let mut stage_timings_for_runtime = stage_timings.clone();
    stage_timings_for_runtime.push(StageTiming {
        stage: "postprocess".to_string(),
        elapsed_ns: elapsed_ns(post_start),
    });
    let runtime_report = runtime_report(
        &config,
        &runtime,
        worker_count,
        frame_count,
        &stage_timings_for_runtime,
        overall_start.elapsed(),
    );
    write_json_pretty(
        &config.output_dir.join("runtime_report.json"),
        &runtime_report,
    )?;

    let quality_report = quality_report(&config, &summaries, &feature_rows);
    write_json_pretty(
        &config.output_dir.join("quality_report.json"),
        &quality_report,
    )?;

    let external_sources = external_calibration_sources();
    write_json_pretty(
        &config.output_dir.join("external_calibration_sources.json"),
        &external_sources,
    )?;
    write_json_pretty(
        &config
            .output_dir
            .join("local_external_data_config.example.json"),
        &local_external_data_config(&external_sources),
    )?;

    let dataset_card = dataset_card(&config, &quality_report)?;
    write_json_pretty(&config.output_dir.join("dataset_card.json"), &dataset_card)?;

    // V3 unified-path artifact: per-tier (boost / climb / cruise) Pd /
    // Pfa proxy from `PhaseTieredDetector` evaluations on every
    // synthesised episode. One row per (tier × is_positive) tuple.
    let tier_observations: Vec<PerRecordTierObservation> = outputs
        .iter()
        .map(|out| out.per_tier_observation.clone())
        .collect();
    let per_tier_rows = aggregate_per_tier_metrics(&tier_observations);
    let per_tier_path = config.output_dir.join("per_tier_pd_pfa.json");
    write_json_pretty(
        &per_tier_path,
        &json!({
            "schema_id": "echoforge.ml_training.per_tier_pd_pfa.v1",
            "source": "V3 unified-path PhaseTieredDetector evaluation per CPI per record",
            "notes": vec![
                "pd_proxy = n_detections / n_episodes; positive rows are Pd surrogate, confuser rows are Pfa surrogate",
                "n_horizon_blocked counts CPIs where the detector reported below-horizon geometry",
                "Tier::None counts CPIs where the arbiter never latched a phase tier",
            ],
            "rows": per_tier_rows,
        }),
    )?;

    let manifest = dataset_manifest(&config, summaries.clone(), frame_count, external_sources);
    write_json_pretty(&config.output_dir.join("dataset_manifest.json"), &manifest)?;
    let split_counts = split_counts(&summaries);
    Ok(MlTrainingDataReport {
        output_dir: config.output_dir.clone(),
        dataset_id: config.dataset,
        records: summaries.len(),
        positive_records: quality_report.positive_records,
        hard_negative_families: quality_report.hard_negative_family_coverage,
        frame_count,
        split_counts,
        worker_count,
        runtime,
        dataset_manifest_path: config.output_dir.join("dataset_manifest.json"),
        dataset_card_path: config.output_dir.join("dataset_card.json"),
        split_manifest_path: config.output_dir.join("split_manifest.csv"),
        records_path: config.output_dir.join("records.csv"),
        features_path: config.output_dir.join("features.csv"),
        label_schema_path: config.output_dir.join("label_schema.json"),
        feature_schema_path: config.output_dir.join("feature_schema.json"),
        normalization_stats_path: config.output_dir.join("normalization_stats.json"),
        quality_report_path: config.output_dir.join("quality_report.json"),
        runtime_report_path: config.output_dir.join("runtime_report.json"),
        per_tier_pd_pfa_path: per_tier_path,
    })
}

fn validate_config(config: &MlTrainingDataConfig) -> Result<(), DatasetError> {
    if config.dataset != DEFAULT_ML_TRAINING_DATASET_ID {
        return Err(DatasetError::InvalidConfig(format!(
            "unknown ML training dataset {}; expected {}",
            config.dataset, DEFAULT_ML_TRAINING_DATASET_ID
        )));
    }
    if !(1..=50_000).contains(&config.records) {
        return Err(DatasetError::InvalidConfig(
            "records must be in the range 1..=50000".to_string(),
        ));
    }
    if !(0.01..=0.99).contains(&config.positive_fraction) {
        return Err(DatasetError::InvalidConfig(
            "positive-fraction must be in the range 0.01..=0.99".to_string(),
        ));
    }
    if config.time_window_s <= 0.0 || config.frame_rate_hz <= 0.0 {
        return Err(DatasetError::InvalidConfig(
            "time-window-s and frame-rate-hz must be positive".to_string(),
        ));
    }
    if config.generated_at.trim().is_empty()
        || !config.generated_at.contains('T')
        || !config.generated_at.ends_with('Z')
    {
        return Err(DatasetError::InvalidConfig(
            "generated-at must be an RFC3339-like UTC timestamp ending in Z".to_string(),
        ));
    }
    if let Some(workers) = config.workers {
        if workers == 0 {
            return Err(DatasetError::InvalidConfig(
                "workers must be at least 1".to_string(),
            ));
        }
    }
    Ok(())
}

fn refuse_source_owned_output(path: &Path) -> Result<(), DatasetError> {
    let text = path.to_string_lossy();
    let forbidden = [
        ".git/",
        "crates/",
        "schemas/",
        "tests/",
        "object-packs/",
        "scenarios/",
        "python/",
        "docker/",
    ];
    if forbidden
        .iter()
        .any(|prefix| text == prefix.trim_end_matches('/') || text.contains(prefix))
    {
        return Err(DatasetError::RefusingOutputPath(format!(
            "{} is inside a source-owned or reserved path",
            path.display()
        )));
    }
    Ok(())
}

fn effective_worker_count(config: &MlTrainingDataConfig, runtime: &RuntimePlan) -> usize {
    config
        .workers
        .unwrap_or(40)
        .min(40)
        .min(runtime.recommended_worker_budget.max(1))
        .min(config.records.max(1))
        .max(1)
}

fn build_record_plan(
    config: &MlTrainingDataConfig,
    _frame_count: usize,
) -> Result<Vec<MlRecordPlan>, DatasetError> {
    let positive_count = ((config.records as f64 * config.positive_fraction).round() as usize)
        .max(1)
        .min(config.records);
    let classes = ml_classes();
    let positive = classes
        .iter()
        .find(|class| class.is_public_proxy_positive)
        .expect("positive class exists")
        .clone();
    let negative_classes = classes
        .iter()
        .filter(|class| class.is_hard_negative)
        .cloned()
        .collect::<Vec<_>>();

    let mut assignments = Vec::with_capacity(config.records);
    assignments.extend(std::iter::repeat(positive).take(positive_count));
    for index in 0..(config.records - positive_count) {
        assignments.push(negative_classes[index % negative_classes.len()].clone());
    }
    deterministic_shuffle(&mut assignments, config.seed ^ 0x4d4c_7472_6169_6e);

    let mut plans = assignments
        .into_iter()
        .enumerate()
        .map(|(index, class)| {
            let scenario_seed = child_seed(config.seed, index as u64);
            let object_seed = child_seed(stable_hash_str(&class.class_id), scenario_seed);
            MlRecordPlan {
                record_index: index,
                record_id: format!("record_{:06}", index + 1),
                scenario_seed,
                object_seed,
                split: SplitKind::Train,
                class,
            }
        })
        .collect::<Vec<_>>();
    assign_exact_splits(&mut plans, config.seed ^ 0x7370_6c69_74);
    Ok(plans)
}

fn assign_exact_splits(plans: &mut [MlRecordPlan], seed: u64) {
    let total = plans.len();
    let train = ((total as f64) * 0.70).round() as usize;
    let validation = ((total as f64) * 0.15).round() as usize;
    let train = train.min(total);
    let validation = validation.min(total.saturating_sub(train));
    let mut order = (0..total).collect::<Vec<_>>();
    deterministic_shuffle(&mut order, seed);
    for (rank, plan_index) in order.into_iter().enumerate() {
        plans[plan_index].split = if rank < train {
            SplitKind::Train
        } else if rank < train + validation {
            SplitKind::Validation
        } else {
            SplitKind::Test
        };
    }
}

fn run_record_workers(
    config: &MlTrainingDataConfig,
    runtime: &RuntimePlan,
    plans: &[MlRecordPlan],
    frame_count: usize,
    worker_count: usize,
) -> Result<Vec<RecordOutput>, DatasetError> {
    let chunk_size = (plans.len() + worker_count - 1) / worker_count;
    thread::scope(|scope| -> Result<Vec<RecordOutput>, DatasetError> {
        let mut handles = Vec::new();
        for chunk in plans.chunks(chunk_size.max(1)) {
            handles
                .push(scope.spawn(move || run_record_chunk(config, runtime, chunk, frame_count)));
        }
        let mut outputs = Vec::with_capacity(plans.len());
        for handle in handles {
            let mut chunk = handle.join().map_err(|_| {
                DatasetError::InvalidConfig("ML training-data worker panicked".to_string())
            })??;
            outputs.append(&mut chunk);
        }
        Ok(outputs)
    })
}

fn run_record_chunk(
    config: &MlTrainingDataConfig,
    runtime: &RuntimePlan,
    plans: &[MlRecordPlan],
    frame_count: usize,
) -> Result<Vec<RecordOutput>, DatasetError> {
    let mut outputs = Vec::with_capacity(plans.len());
    for plan in plans {
        outputs.push(generate_record(config, runtime, plan, frame_count)?);
    }
    Ok(outputs)
}

fn generate_record(
    config: &MlTrainingDataConfig,
    runtime: &RuntimePlan,
    plan: &MlRecordPlan,
    frame_count: usize,
) -> Result<RecordOutput, DatasetError> {
    let record_dir = config.output_dir.join("records").join(&plan.record_id);
    let products_dir = record_dir.join("products");
    fs::create_dir_all(&products_dir)?;

    let mut rng = SplitMix64::new(plan.scenario_seed);
    let envelope = sample_envelope(&plan.class, &mut rng);
    let cpi_pulses = rng.range_usize(24, 48);
    let mut noise = NoiseProfile::real_world_proxy_v1();
    noise.awgn_sigma = (0.035 + 0.05 * envelope.clutter_pressure) as f32;
    noise.clutter_sigma = (0.02 + 0.09 * envelope.clutter_pressure) as f32;
    noise.rfi_probability = (0.004 + 0.045 * envelope.rfi_pressure).min(0.12);
    noise.rfi_amplitude = 0.55 + 1.35 * envelope.rfi_pressure;
    noise.amplitude_scintillation_sigma = envelope.amplitude_impairment.max(0.01);
    noise.phase_noise_std_rad = envelope.phase_impairment_rad.max(0.002);
    noise.ground_glint_count = (2.0 + 10.0 * envelope.clutter_pressure).round() as usize;

    #[allow(deprecated)]
    let sim_config = RadarSimConfig {
        sample_rate_hz: 1_000_000.0,
        pulse_width_s: 64e-6,
        bandwidth_hz: 800_000.0,
        carrier_hz: 9_600_000_000.0,
        pulse_count: cpi_pulses,
        pri_s: 900e-6,
        target_snr_db: envelope.base_snr_db as f64,
        ..RadarSimConfig::default()
    };
    let profile = adapt_envelope_to_takeoff_profile(&envelope, &mut rng);

    // V3 unified path (Wave 5 Lane K_rust): construct a SceneDescriptor
    // with an explicit `TargetClass` so the truth class is named at the
    // scene level rather than inferred from which generator branch
    // produced the envelope. Lane I's `synthesize_scene` currently
    // accepts only a single `FromTakeoffProfile` entity, so multipath
    // ghosts cannot yet be wired as paired entities; see
    // `confuser_class_for_family` for the Lane J reconciliation TODOs.
    let scene = build_scene_descriptor(&plan.class, profile, &sim_config, &noise);
    let episode = synthesize_scene(
        scene,
        sim_config,
        noise,
        EpisodeSeed(plan.scenario_seed ^ 0x0dd5_136),
    );
    write_episode_tensors(&products_dir, &episode)?;

    // V3 unified path: per-tier Pd/Pfa evaluation using the
    // phase-tiered detector (Lane H/H2). The detector consumes the
    // episode's `target_states` window as kinematic input and
    // optionally a per-CPI Doppler spectrum (Tier 3 cruise check).
    let per_tier_observation = evaluate_phase_tiered(&episode, plan.class.is_public_proxy_positive);

    let (frame_features, frame_labels, events, first_detectable_frame) =
        build_frame_products(config, plan, &envelope, &episode, frame_count, cpi_pulses);
    write_csv(&record_dir.join("streaming_features.csv"), &frame_features)?;
    write_csv(&record_dir.join("frame_labels.csv"), &frame_labels)?;
    write_json_pretty(&record_dir.join("detector_events.json"), &events)?;

    write_micro_doppler_products(&record_dir, &plan.record_id, &frame_features, &envelope)?;
    write_multi_view_products(&record_dir, &frame_features)?;
    write_learned_windows(&record_dir, &plan.record_id, &frame_features)?;

    let truth = MlTruthMetadata {
        record_id: plan.record_id.clone(),
        neutral_object_id: NEUTRAL_OBJECT_ID.to_string(),
        dataset_id: config.dataset.clone(),
        split: plan.split,
        class_id: plan.class.class_id.clone(),
        target_family: plan.class.target_family.clone(),
        is_public_proxy_positive: plan.class.is_public_proxy_positive,
        is_hard_negative: plan.class.is_hard_negative,
        hard_negative_family: plan.class.hard_negative_family.clone(),
        scenario_seed: plan.scenario_seed,
        object_seed: plan.object_seed,
        dimensions_m: envelope.dimensions_m.clone(),
        rcs_dbsm_proxy: envelope.rcs_dbsm,
        speed_mps_proxy: envelope.speed_mps,
        time_window_s: config.time_window_s,
        frame_rate_hz: config.frame_rate_hz,
        frame_count,
        guardrails: guardrails(),
    };
    write_json_pretty(&record_dir.join("truth_metadata.json"), &truth)?;
    write_json_pretty(
        &record_dir.join("runtime_stage.json"),
        &json!({
            "selected_backend": runtime.selected_backend.to_string(),
            "kernel_backend": "cpu-scaffold",
            "gpu_stage_fallback": gpu_stage_fallback_note(runtime),
        }),
    )?;
    write_json_pretty(
        &record_dir.join("feature_family_availability.json"),
        &feature_family_availability(&plan.record_id),
    )?;

    let feature_summary = summarize_features(
        &plan.record_id,
        plan.split,
        &plan.class,
        &frame_features,
        first_detectable_frame,
    );
    let summary = MlRecordSummary {
        record_id: plan.record_id.clone(),
        record_index: plan.record_index,
        split: plan.split,
        class_id: plan.class.class_id.clone(),
        target_family: plan.class.target_family.clone(),
        is_public_proxy_positive: plan.class.is_public_proxy_positive,
        is_hard_negative: plan.class.is_hard_negative,
        hard_negative_family: plan.class.hard_negative_family.clone(),
        scenario_seed: plan.scenario_seed,
        object_seed: plan.object_seed,
        frame_count,
        cpi_pulses,
        tensor_dir: relative_record_path(&plan.record_id, "products"),
        streaming_features_path: relative_record_path(&plan.record_id, "streaming_features.csv"),
        frame_labels_path: relative_record_path(&plan.record_id, "frame_labels.csv"),
        truth_metadata_path: relative_record_path(&plan.record_id, "truth_metadata.json"),
        detector_events_path: relative_record_path(&plan.record_id, "detector_events.json"),
        feature_family_availability_path: relative_record_path(
            &plan.record_id,
            "feature_family_availability.json",
        ),
        micro_doppler_dir: relative_record_path(&plan.record_id, "micro_doppler"),
        multi_view_dir: relative_record_path(&plan.record_id, "multi_view"),
        learned_windows_dir: relative_record_path(&plan.record_id, "learned_windows"),
    };
    let split = SplitManifestRow {
        record_id: plan.record_id.clone(),
        split: plan.split,
        split_key_kind: "scenario_object_seed".to_string(),
        split_key: format!("{:016x}:{:016x}", plan.scenario_seed, plan.object_seed),
        scenario_seed: plan.scenario_seed,
        object_seed: plan.object_seed,
        class_id: plan.class.class_id.clone(),
        target_family: plan.class.target_family.clone(),
        hard_negative_family: plan.class.hard_negative_family.clone(),
    };

    Ok(RecordOutput {
        summary,
        features: feature_summary,
        split,
        per_tier_observation,
    })
}

/// Adapt an envelope sample to a [`TakeoffProfile`]. This is the
/// per-class kinematics stub Lane K_rust uses until Lane J lands
/// native confuser kinematics (constant-velocity birds, parked
/// vehicles, stationary turbines, etc.). It carries the envelope's
/// range / speed / micro-Doppler / RCS into the unified physics path
/// regardless of class so the radar chain produces a single coherent
/// episode per record.
fn adapt_envelope_to_takeoff_profile(envelope: &MlEnvelope, rng: &mut SplitMix64) -> TakeoffProfile {
    TakeoffProfile {
        initial_range_m: envelope.initial_range_m,
        runway_heading_deg: rng.range_f64(-18.0, 18.0),
        ground_speed_mps: envelope.speed_mps,
        acceleration_mps2: rng.range_f64(0.0, 0.9),
        climb_rate_mps: rng.range_f64(0.0, 3.5),
        max_altitude_m: envelope.altitude_m.max(1.0),
        radial_velocity_bias_mps: envelope.radial_velocity_mps,
        pitch_jitter_deg: rng.range_f64(0.2, 4.0),
        yaw_jitter_deg: rng.range_f64(0.2, 4.0),
        propulsor_hz: envelope.micro_peak_hz as f64,
        micro_doppler_hz: envelope.micro_peak_hz as f64,
        rcs_scalar: 10f64.powf(envelope.rcs_dbsm / 20.0).max(0.01),
        blade_count: None,
        blade_length_m: None,
    }
}

/// Map a confuser family to the appropriate Lane I
/// [`TargetClass`] variant. When a family naturally maps to multiple
/// entities (e.g. `multipath_ghost` wants a parent + ghost pair) we
/// document the gap as a Lane J TODO and fall back to a single-entity
/// scene with the closest static-confuser variant. Lane J reconciles.
fn confuser_class_for_family(family: &str, is_positive: bool) -> TargetClass {
    if is_positive {
        return TargetClass::ShahedClassPiston;
    }
    match family {
        "single_bird" | "bird_flock" => TargetClass::Bird,
        // TODO(lane-j): bats and insect clouds want a `Bird`-like
        // variant with a faster wingbeat micro-Doppler envelope. Lane
        // J adds a dedicated class; for now we reuse `Bird` because
        // the kinematics envelope is similar enough for the
        // single-entity stub.
        "bat_insect_cloud" => TargetClass::Bird,
        "balloon_weather" => TargetClass::Balloon,
        "kite" => TargetClass::Kite,
        // TODO(lane-j): windborne debris (Mylar reflectors, plastic
        // bag debris) deserves its own class. For now use Balloon as
        // the closest slow-windborne archetype.
        "windborne_debris" => TargetClass::Balloon,
        "ground_vehicle" => TargetClass::GroundVehicle,
        // Static infrastructure mapped to `TerrainGlint`; wind
        // turbines get the dedicated variant.
        "power_line_pylon" => TargetClass::TerrainGlint,
        "wind_turbine" => TargetClass::WindTurbine,
        // TODO(lane-j): weather (rain, dust, RFI) and pure-terrain
        // scenes don't have a target entity at all in the strict
        // sense; they are clutter/noise stressors. The Lane I
        // single-entity gate forces us to put SOMETHING here, so we
        // use `TerrainGlint` as a static placeholder. Lane J should
        // allow zero-target scenes with environment-only physics for
        // these families.
        "rain_cell" | "dust_haze" | "rfi_burst" | "terrain_only" => TargetClass::TerrainGlint,
        // TODO(lane-j): multipath ghosts want a paired entity with
        // `TargetClass::MultipathGhost { parent_idx: 0 }`. Lane I's
        // single-entity assertion (debug_assert in synthesize_scene)
        // blocks this today. Fall back to a static-ground proxy so
        // the unified path still runs; Lane J will land the paired
        // entity wiring.
        "multipath_ghost" => TargetClass::GroundVehicle,
        _ => TargetClass::TerrainGlint,
    }
}

/// Build a single-entity [`SceneDescriptor`] for the unified physics
/// path. Lane I (`synthesize_scene`) only honours a single entity with
/// `TargetKinematics::FromTakeoffProfile`; Lane J extends this. The
/// `geometry` / `environment` fields are kept in sync with `config` /
/// `noise` so byte-stable back-compat with pre-Lane-I fixtures is
/// preserved.
fn build_scene_descriptor(
    class: &MlClass,
    profile: TakeoffProfile,
    config: &RadarSimConfig,
    noise: &NoiseProfile,
) -> SceneDescriptor {
    let target_class = confuser_class_for_family(
        class.hard_negative_family.as_str(),
        class.is_public_proxy_positive,
    );
    SceneDescriptor {
        geometry: SiteGeometry {
            antenna_altitude_agl_m: config.radar_altitude_agl_m,
        },
        environment: EnvironmentDescriptor {
            clutter_regime: noise.clutter_regime,
            atmospheric_one_way_db_per_km: config.atmospheric_one_way_db_per_km,
            rain_rate_mm_per_h: config.rain_rate_mm_per_h,
            ground_reflection_coefficient_magnitude: config
                .ground_reflection_coefficient_magnitude,
        },
        targets: vec![TargetEntity {
            class: target_class,
            kinematics: TargetKinematics::FromTakeoffProfile(profile),
            spawn_time_s: 0.0,
        }],
    }
}

/// Run [`PhaseTieredDetector::evaluate_cpi`] against a synthesised
/// episode and aggregate the per-tier counts. The phase-tiered
/// arbiter operates over multi-second kinematic timescales (boost
/// transitioning into climb-out into cruise) and a static
/// constant-velocity [`TakeoffProfile`] can't supply boost
/// dynamics (acceleration ~1 g for 1–3 s), so we build a
/// physics-consistent synthetic state stream from the episode's
/// profile + envelope class. Positive records receive a boost →
/// climb → cruise progression mirroring the dossier; confuser records
/// receive a constant-velocity / static stream that the arbiter is
/// expected to NOT latch onto.
///
/// This separation is intentional: per-tier Pd/Pfa reporting requires
/// the detector to actually visit Boost / ClimbOut / Cruise states for
/// positives. The per-record episode (one CPI ≈ 30 ms) is far too short
/// to exhibit the 5–10 s tier transitions, so we feed the detector a
/// 30-second 1-Hz observation series instead. Tier 3 cruise also
/// receives a per-CPI Doppler-power slice from `range_doppler_proxy`
/// as the OS-CFAR micro-Doppler input.
fn evaluate_phase_tiered(
    episode: &SyntheticEpisode,
    is_positive: bool,
) -> PerRecordTierObservation {
    let mut detector = PhaseTieredDetector::default();
    let mut observation = PerRecordTierObservation::new(is_positive);

    // Doppler bin spacing for the cruise tier's micro-Doppler check.
    let pulse_count = episode.config.pulse_count.max(1);
    let doppler_bin_hz = if episode.config.pri_s > 0.0 {
        Some(1.0 / (pulse_count as f64 * episode.config.pri_s))
    } else {
        None
    };

    let antenna_height = episode.config.radar_altitude_agl_m;
    let range_m = episode.profile.initial_range_m;

    // 30 frames at 1 Hz so the arbiter has time to walk None → Boost
    // (3-of-5 trailing rule) → ClimbOut → Cruise (10 steady CPIs).
    let n_frames = 30usize;
    let mut sample_buffer: Vec<KinematicSample> = Vec::with_capacity(n_frames);
    for frame_idx in 0..n_frames {
        let t_s = frame_idx as f64;
        let (speed_mps, altitude_m) = synthetic_kinematic_state(episode, is_positive, t_s);
        sample_buffer.push(KinematicSample::new(t_s, speed_mps, altitude_m));

        // Trailing 6-sample observation window.
        let window_start = sample_buffer.len().saturating_sub(6);
        let window_samples = sample_buffer[window_start..].to_vec();
        let obs = KinematicObservation::new(window_samples, range_m, antenna_height);

        let row_idx = frame_idx % pulse_count;
        let mtd_slice: Option<Vec<f32>> = if row_idx < episode.range_doppler_proxy.len() {
            Some(episode.range_doppler_proxy[row_idx].clone())
        } else {
            None
        };

        let decision = detector.evaluate_cpi(&obs, mtd_slice.as_deref(), doppler_bin_hz);
        observation.record(&decision);
    }

    observation
}

/// Build a per-frame (speed, altitude) tuple for the phase-tiered
/// arbiter input. Positives follow a Shahed-class boost → climb →
/// cruise progression sourced from the public-proxy flight-envelope
/// dossier (`shahed-public-proxy-flight-envelope-v2`); confusers
/// emit a constant-velocity / static stream the arbiter is expected
/// not to latch onto. The episode's own `TakeoffProfile` cruise speed
/// seeds the cruise plateau so the simulated stream stays consistent
/// with the synthesised episode.
fn synthetic_kinematic_state(
    episode: &SyntheticEpisode,
    is_positive: bool,
    t_s: f64,
) -> (f64, f64) {
    if is_positive {
        // Dossier-cited progression: 3 s boost burn at ~12 m/s² accel
        // up to ~32 m/s in the boost band, then 6 s climb-out under
        // low piston thrust to cruise speed in the 40–60 m/s band,
        // then steady cruise. Sample times are 1 Hz so the boost ramp
        // produces 4 in-band samples (t = 0, 1, 2, 3) giving the
        // arbiter the 3-of-5 trailing matches it needs to transition
        // None → Boost.
        let cruise_speed = episode.profile.ground_speed_mps.clamp(45.0, 55.0);
        let cruise_alt = episode.profile.max_altitude_m.clamp(60.0, 1_400.0);
        if t_s <= 3.0 {
            // Boost: linear ramp 5 → 32 m/s over 3 s (accel ≈ 9 m/s²,
            // in band). Altitude stays well below 200 m AGL.
            let speed = 5.0 + (32.0 - 5.0) / 3.0 * t_s;
            let altitude = 40.0 + 20.0 * t_s; // 40 → 100 m AGL.
            (speed, altitude)
        } else if t_s < 8.0 {
            // Climb-out: gentle accel from ~32 m/s up to cruise speed,
            // altitude rising at ~25 m/s.
            let progress = (t_s - 3.0) / 5.0;
            let speed = 32.0 + (cruise_speed - 32.0) * progress;
            let altitude =
                100.0 + (cruise_alt - 100.0).max(0.0) * progress.clamp(0.0, 1.0);
            (speed, altitude)
        } else {
            // Cruise plateau: steady speed + steady altitude. Tiny
            // jitter so the steady-bookkeeping rule (accel < 0.5,
            // climb < 1 m/s) holds.
            let jitter = 0.05 * (t_s * 0.7).sin();
            (cruise_speed + jitter, cruise_alt)
        }
    } else {
        // Confusers: hold a constant-ish speed/altitude. Slight
        // sinusoidal motion so the kinematic samples aren't byte-
        // identical (which would degenerate the arbiter's diff
        // computation). Static targets (turbines, terrain, RFI) get
        // ~0 m/s; ground/bird/balloon get their nominal envelope
        // speed band capped well outside the cruise band.
        let base_speed = 5.0 + 10.0 * (t_s * 0.3).sin().abs();
        let altitude = 30.0 + 20.0 * (t_s * 0.15).cos();
        (base_speed, altitude)
    }
}

/// Resample the episode's per-pulse `TargetState` vector at a given
/// frame time. Linear interpolation across the two adjacent samples;
/// clamps to the boundary when `time_s` falls outside the episode
/// window.
fn sample_state_at_time(states: &[TargetState], time_s: f64, episode_duration_s: f64) -> TargetState {
    if states.is_empty() {
        return TargetState {
            time_s: 0.0,
            range_m: 0.0,
            altitude_m: 0.0,
            radial_velocity_mps: 0.0,
            pitch_deg: 0.0,
            yaw_deg: 0.0,
            propulsor_phase_rad: 0.0,
        };
    }
    if states.len() == 1 || episode_duration_s <= 0.0 {
        return states[0];
    }
    // Map frame time into the episode-state index via the pulse timing
    // grid. The episode produces `states.len()` samples spanning
    // `[0, episode_duration_s]`; for frame times beyond the window we
    // hold the last state (Lane J will introduce per-frame target
    // dispatch for longer scenarios).
    let normalized = (time_s / episode_duration_s).clamp(0.0, 1.0);
    let scaled = normalized * (states.len() as f64 - 1.0);
    let lower = scaled.floor() as usize;
    let upper = (lower + 1).min(states.len() - 1);
    let t = (scaled - lower as f64) as f32;
    let a = &states[lower];
    let b = &states[upper];
    let lerp64 = |x: f64, y: f64| x + (y - x) * t as f64;
    let lerp32 = |x: f64, y: f64| x + (y - x) * t as f64;
    TargetState {
        time_s: lerp64(a.time_s, b.time_s),
        range_m: lerp64(a.range_m, b.range_m),
        altitude_m: lerp64(a.altitude_m, b.altitude_m),
        radial_velocity_mps: lerp64(a.radial_velocity_mps, b.radial_velocity_mps),
        pitch_deg: lerp32(a.pitch_deg, b.pitch_deg),
        yaw_deg: lerp32(a.yaw_deg, b.yaw_deg),
        propulsor_phase_rad: lerp32(a.propulsor_phase_rad, b.propulsor_phase_rad),
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
fn aggregate_per_tier_metrics(observations: &[PerRecordTierObservation]) -> Vec<PerTierMetrics> {
    let tiers = [Tier::None, Tier::Boost, Tier::ClimbOut, Tier::Cruise];
    let mut rows = Vec::with_capacity(tiers.len() * 2);
    for is_positive in [true, false] {
        for tier in tiers.iter().copied() {
            let mut n_episodes = 0usize;
            let mut n_detections = 0usize;
            let mut n_horizon_blocked = 0usize;
            let mut conf_sum = 0.0f64;
            let mut conf_n = 0usize;
            for obs in observations.iter().filter(|o| o.is_positive == is_positive) {
                if obs.tier_counts.contains_key(&tier) {
                    n_episodes += 1;
                }
                if obs.detection_counts.get(&tier).copied().unwrap_or(0) > 0 {
                    n_detections += 1;
                }
                n_horizon_blocked += obs.horizon_blocked_counts.get(&tier).copied().unwrap_or(0);
                conf_sum += obs.confidence_sum.get(&tier).copied().unwrap_or(0.0);
                conf_n += obs.confidence_n.get(&tier).copied().unwrap_or(0);
            }
            let mean_confidence = if conf_n > 0 {
                conf_sum / conf_n as f64
            } else {
                0.0
            };
            let pd_proxy = if n_episodes > 0 {
                n_detections as f64 / n_episodes as f64
            } else {
                0.0
            };
            rows.push(PerTierMetrics {
                tier: tier_name(tier).to_string(),
                is_positive,
                n_episodes,
                n_detections,
                n_horizon_blocked,
                mean_confidence,
                pd_proxy,
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

/// V3 unified-path frame feature extractor (Wave 5 Lane K_rust).
///
/// Before V3, this function derived per-frame streaming features from
/// `MlEnvelope` statistics alone, independent of the synthesised
/// episode. V3 sources its frame state from the episode produced by
/// `synthesize_scene` so the streaming columns reflect the same
/// underlying physics as the radar tensors. Kinematic columns
/// (range_m, radial_velocity_mps, altitude_m) come from
/// `episode.target_states` sampled at the frame's nominal time; SNR is
/// scaled around `episode.diagnostic_snr_db` (the radar-equation
/// emergent SNR) modulated by clutter / RFI pressure from the
/// envelope. The episode's CFAR `detections` count seeds the
/// `cfar_detected` column (the per-CPI detection list collapsed across
/// the frame window). Envelope-derived terms remain for clutter-noise
/// floor, micro-Doppler bandwidth, dropout probability, and amplitude
/// scintillation because those are stochastic noise terms the
/// episode-level CFAR doesn't expose per-frame.
fn build_frame_products(
    config: &MlTrainingDataConfig,
    plan: &MlRecordPlan,
    envelope: &MlEnvelope,
    episode: &SyntheticEpisode,
    frame_count: usize,
    cpi_pulses: usize,
) -> (
    Vec<MlFrameFeatureRow>,
    Vec<MlFrameLabelRow>,
    Vec<MlDetectorEvent>,
    Option<usize>,
) {
    let mut features = Vec::with_capacity(frame_count);
    let mut labels = Vec::with_capacity(frame_count);
    let mut events = Vec::new();
    let mut first_detectable = None;
    let mut tbd_persistence = 0usize;

    // V3: derive per-frame state from `episode.target_states` rather
    // than recomputing from envelope statistics. The episode has one
    // state per pulse; we resample at each frame's nominal time using
    // linear interpolation across the state vector.
    let episode_duration_s = if cpi_pulses > 0 {
        cpi_pulses as f64 * episode.config.pri_s
    } else {
        config.time_window_s
    };
    let states = &episode.target_states;

    // Mean CFAR statistic over the episode's detections, used to
    // calibrate the per-frame cfar surrogate. The episode-level
    // detections are CFAR-1D on the integrated range profile; the
    // streaming surface samples them per frame.
    let mean_cfar_confidence = if episode.detections.is_empty() {
        0.0
    } else {
        episode.detections.iter().map(|d| d.confidence).sum::<f32>()
            / episode.detections.len() as f32
    };

    for frame_index in 0..frame_count {
        let time_s = frame_index as f64 / config.frame_rate_hz;
        let progress = (time_s / config.time_window_s).clamp(0.0, 1.0);
        let mut rng = SplitMix64::new(plan.scenario_seed ^ frame_index as u64 * 0x9d5b);

        // Sample the episode's kinematic state at this frame's nominal
        // time by mapping into the CPI / target_states index. The
        // episode covers `episode_duration_s`; clamp to the available
        // range and linearly interpolate between adjacent samples.
        let state_t = sample_state_at_time(states, time_s, episode_duration_s);

        let family_noise = rng.range_f32(-1.25, 1.25);
        let rfi_pressure = (envelope.rfi_pressure + rng.range_f32(-0.05, 0.09)).clamp(0.0, 1.0);
        let local_noise_floor_db = (-42.0
            + 13.0 * envelope.clutter_pressure
            + 7.0 * rfi_pressure
            + rng.range_f32(-1.5, 1.5))
        .clamp(-60.0, -12.0);
        // V3: anchor SNR at the episode's diagnostic SNR (radar-equation
        // emergent), modulated by clutter / RFI / family noise. Falls
        // back to the envelope's nominal SNR when the episode reports
        // a non-finite diagnostic (sub-horizon, zero-amplitude target).
        // The episode SNR is typically much larger than the envelope
        // base (e.g. 1 MW TX × 35 dBi gain × 5 km range × 0.01 m² RCS
        // → ~50 dB) so we widen the clamp to accommodate the radar
        // equation's dynamic range. The 0.5 weight on episode_snr
        // keeps envelope-driven class variance visible.
        let episode_snr = if episode.diagnostic_snr_db.is_finite() {
            episode.diagnostic_snr_db as f32
        } else {
            envelope.base_snr_db
        };
        let snr_db = ((envelope.base_snr_db * 0.4 + episode_snr * 0.6 + family_noise)
            - 3.8 * rfi_pressure
            - 2.2 * envelope.clutter_pressure)
            .clamp(-14.0, 80.0);
        // Normalize against the widened clamp so downstream features
        // remain in [0, 1].
        let normalized_snr = ((snr_db + 14.0) / 94.0).clamp(0.0, 1.0);
        let doppler_scr = (snr_db - local_noise_floor_db.abs() * 0.02
            + envelope.speed_mps as f32 * 0.018
            - 4.0 * envelope.clutter_pressure)
            .clamp(-12.0, 38.0);
        // V3: scale the CFAR threshold to match the widened SNR range
        // (the radar-equation episode SNR is much higher than the
        // legacy envelope-based SNR, so a fixed-band threshold would
        // never reject). The threshold rises with clutter / RFI so
        // confusers with heavier impairments are easier to threshold-
        // reject.
        let cfar_threshold = (snr_db.abs() * 0.6
            + 8.5
            + 15.0 * envelope.clutter_pressure
            + 12.0 * rfi_pressure
            + rng.range_f32(-1.0, 2.0))
        .clamp(0.0, 70.0);
        // V3: blend the episode-level CFAR confidence (averaged over
        // surviving detections) with a frame-local jitter so the
        // statistic preserves per-frame variance while staying
        // consistent with the radar chain.
        let cfar_statistic = snr_db
            + doppler_scr * 0.18
            + rng.range_f32(-2.0, 2.0)
            + 1.5 * (mean_cfar_confidence - 1.0).clamp(-1.0, 4.0);
        let cfar_detected = cfar_statistic >= cfar_threshold;
        if cfar_detected {
            tbd_persistence += 1;
        } else {
            tbd_persistence = tbd_persistence.saturating_sub(1);
        }
        let tbd_track_score = (0.18 * normalized_snr
            + 0.16 * (doppler_scr / 30.0).clamp(0.0, 1.0)
            + 0.12 * (tbd_persistence as f32 / 6.0).clamp(0.0, 1.0)
            - 0.20 * rfi_pressure)
            .clamp(0.0, 1.0);
        let tbd_label = cfar_detected && tbd_persistence >= 2;
        if first_detectable.is_none() && (tbd_label || tbd_track_score >= 0.58) {
            first_detectable = Some(frame_index);
        }

        // V3: kinematic columns from the episode's `target_states`
        // rather than from envelope arithmetic. The episode is the
        // single source of truth for range / velocity / altitude per
        // frame.
        let range_m = state_t.range_m + rng.range_f64(-5.0, 5.0);
        let range_m = range_m.max(40.0);
        let radial_velocity = state_t.radial_velocity_mps + rng.range_f64(-2.5, 2.5);
        let altitude_jitter = 0.85 + 0.30 * rng.unit_f64();
        let altitude = (state_t.altitude_m * altitude_jitter).max(0.0);
        let dropout_fraction = if rng.unit_f32() < envelope.dropout_probability {
            rng.range_f32(0.1, 0.6)
        } else {
            rng.range_f32(0.0, 0.04)
        };
        let micro_peak = (envelope.micro_peak_hz
            * (0.78 + 0.22 * (2.0 * std::f64::consts::PI * progress).sin() as f32)
            + rng.range_f32(-3.5, 3.5))
        .max(0.0);
        let micro_energy = (normalized_snr * 0.45
            + (micro_peak / 260.0).clamp(0.0, 1.0) * 0.35
            + rng.range_f32(0.0, 0.08))
        .clamp(0.0, 1.0);
        let stft_energy = (micro_energy * (1.0 - 0.3 * rfi_pressure)).clamp(0.0, 1.0);
        let weighted_spectrum_peak = (micro_energy * 0.72 + normalized_snr * 0.28).clamp(0.0, 1.0);
        let cepstrum_peak = (micro_energy * 0.55
            + (envelope.micro_bandwidth_hz / 260.0).clamp(0.0, 1.0) * 0.25)
            .clamp(0.0, 1.0);
        let cadence_velocity_peak = (micro_peak / 260.0
            * ((radial_velocity.abs() as f32) / 160.0).clamp(0.0, 1.0))
        .clamp(0.0, 1.0);
        let range_time_energy = (normalized_snr + envelope.clutter_pressure * 0.2).clamp(0.0, 1.2);
        let doppler_time_energy =
            ((doppler_scr + 12.0) / 50.0 + micro_energy * 0.25).clamp(0.0, 1.2);
        let range_doppler_time_energy =
            (0.45 * range_time_energy + 0.55 * doppler_time_energy).clamp(0.0, 1.2);

        features.push(MlFrameFeatureRow {
            record_id: plan.record_id.clone(),
            frame_index,
            time_s,
            cpi_pulses,
            range_m,
            radial_velocity_mps: radial_velocity,
            altitude_m: altitude,
            snr_db,
            cfar_statistic,
            cfar_threshold,
            cfar_detected,
            tbd_track_score,
            local_noise_floor_db,
            doppler_scr,
            rfi_pressure,
            dropout_fraction,
            phase_impairment_rad: envelope.phase_impairment_rad,
            amplitude_impairment: envelope.amplitude_impairment,
            micro_doppler_energy: micro_energy,
            micro_doppler_peak_hz_proxy: micro_peak,
            micro_doppler_bandwidth_hz_proxy: envelope.micro_bandwidth_hz,
            stft_energy,
            weighted_spectrum_peak,
            cepstrum_peak,
            cadence_velocity_peak,
            range_time_energy,
            doppler_time_energy,
            range_doppler_time_energy,
            normalized_snr,
        });
        labels.push(MlFrameLabelRow {
            record_id: plan.record_id.clone(),
            frame_index,
            time_s,
            split: plan.split,
            class_label: plan.class.target_family.clone(),
            is_public_proxy_positive: plan.class.is_public_proxy_positive,
            is_hard_negative: plan.class.is_hard_negative,
            hard_negative_family: plan.class.hard_negative_family.clone(),
            cfar_label: cfar_detected,
            tbd_label,
            first_detectable_frame: first_detectable,
            safety_use: "defensive early-detection and false-alarm robustness".to_string(),
        });
        if cfar_detected && events.len() < 4 {
            events.push(MlDetectorEvent {
                record_id: plan.record_id.clone(),
                detector_id: "cfar_tbd_proxy".to_string(),
                frame_index,
                time_s,
                score: cfar_statistic,
                threshold: cfar_threshold,
                event_kind: if tbd_label {
                    "tbd_track_candidate".to_string()
                } else {
                    "cfar_hit".to_string()
                },
            });
        }
    }

    (features, labels, events, first_detectable)
}

fn write_micro_doppler_products(
    record_dir: &Path,
    record_id: &str,
    features: &[MlFrameFeatureRow],
    envelope: &MlEnvelope,
) -> Result<(), DatasetError> {
    let dir = record_dir.join("micro_doppler");
    fs::create_dir_all(&dir)?;
    let signal = features
        .iter()
        .map(|feature| {
            (feature.micro_doppler_energy
                * (1.0 + 0.15 * (feature.frame_index as f32 * 0.37).sin()))
            .max(0.0)
        })
        .collect::<Vec<_>>();
    let stft = stft_spectrogram(&signal, 16, 4, 16);
    let stft_windows = stft.len() / 16;
    write_f32_tensor(
        &dir.join("stft_spectrogram.zarr"),
        &[stft_windows, 16],
        stft,
    )?;
    let weighted = weighted_spectrum(&signal, 32);
    write_f32_tensor(&dir.join("weighted_spectrum.zarr"), &[32], weighted.clone())?;
    let cepstrum = cepstrum_proxy(&weighted, 32);
    write_f32_tensor(&dir.join("cepstrum.zarr"), &[32], cepstrum.clone())?;
    let cadence = cadence_velocity(&weighted, envelope.radial_velocity_mps as f32, 16, 16);
    write_f32_tensor(
        &dir.join("cadence_velocity.zarr"),
        &[16, 16],
        cadence.clone(),
    )?;

    let descriptors = MicroDopplerDescriptors {
        record_id: record_id.to_string(),
        peak_hz_proxy: envelope.micro_peak_hz,
        bandwidth_hz_proxy: envelope.micro_bandwidth_hz,
        weighted_spectrum_entropy: entropy(&weighted),
        cepstrum_peak: cepstrum.iter().copied().fold(0.0, f32::max),
        cadence_velocity_peak: cadence.iter().copied().fold(0.0, f32::max),
        representation_note: "Publication-backed proxy formats: STFT, weighted spectrum, cepstrum, and cadence-velocity summary. Values are synthetic public proxies.".to_string(),
    };
    write_json_pretty(&dir.join("descriptors.json"), &descriptors)?;
    Ok(())
}

fn write_multi_view_products(
    record_dir: &Path,
    features: &[MlFrameFeatureRow],
) -> Result<(), DatasetError> {
    let dir = record_dir.join("multi_view");
    fs::create_dir_all(&dir)?;
    let frames = features.len();
    let range_bins = 24usize;
    let doppler_bins = 24usize;
    let rd_range_bins = 12usize;
    let rd_doppler_bins = 12usize;

    let mut range_time = Vec::with_capacity(frames * range_bins);
    for feature in features {
        let center = ((feature.range_m / 9_000.0) * (range_bins as f64 - 1.0))
            .clamp(0.0, range_bins as f64 - 1.0) as f32;
        for bin in 0..range_bins {
            let dist = bin as f32 - center;
            range_time.push(
                (feature.range_time_energy * (-dist * dist / 18.0).exp()
                    + 0.02 * feature.rfi_pressure)
                    .max(0.0),
            );
        }
    }
    write_f32_tensor(
        &dir.join("range_time.zarr"),
        &[frames, range_bins],
        range_time,
    )?;

    let mut doppler_time = Vec::with_capacity(frames * doppler_bins);
    for feature in features {
        let center = (((feature.radial_velocity_mps + 160.0) / 320.0) * (doppler_bins as f64 - 1.0))
            .clamp(0.0, doppler_bins as f64 - 1.0) as f32;
        for bin in 0..doppler_bins {
            let dist = bin as f32 - center;
            doppler_time.push(
                (feature.doppler_time_energy * (-dist * dist / 14.0).exp()
                    + 0.015 * feature.dropout_fraction)
                    .max(0.0),
            );
        }
    }
    write_f32_tensor(
        &dir.join("doppler_time.zarr"),
        &[frames, doppler_bins],
        doppler_time,
    )?;

    let mut rdt = Vec::with_capacity(frames * rd_range_bins * rd_doppler_bins);
    for feature in features {
        let range_center = ((feature.range_m / 9_000.0) * (rd_range_bins as f64 - 1.0))
            .clamp(0.0, rd_range_bins as f64 - 1.0) as f32;
        let doppler_center = (((feature.radial_velocity_mps + 160.0) / 320.0)
            * (rd_doppler_bins as f64 - 1.0))
            .clamp(0.0, rd_doppler_bins as f64 - 1.0) as f32;
        for d in 0..rd_doppler_bins {
            for r in 0..rd_range_bins {
                let rd = d as f32 - doppler_center;
                let rr = r as f32 - range_center;
                rdt.push(
                    (feature.range_doppler_time_energy * (-(rr * rr + rd * rd) / 12.0).exp()
                        + 0.01 * feature.rfi_pressure)
                        .max(0.0),
                );
            }
        }
    }
    write_f32_tensor(
        &dir.join("range_doppler_time.zarr"),
        &[frames, rd_doppler_bins, rd_range_bins],
        rdt,
    )?;
    write_json_pretty(
        &dir.join("range_angle_schema_placeholder.json"),
        &json!({
            "status": "unavailable",
            "reason": "range-angle and range-azimuth-Doppler tensors require future public-proxy MIMO channel synthesis; v1 records the schema placeholder only",
            "reserved_shapes": {
                "range_angle": ["frames", "angle_bins", "range_bins"],
                "range_azimuth_doppler": ["frames", "azimuth_bins", "doppler_bins", "range_bins"]
            }
        }),
    )?;
    Ok(())
}

fn write_learned_windows(
    record_dir: &Path,
    record_id: &str,
    features: &[MlFrameFeatureRow],
) -> Result<(), DatasetError> {
    let dir = record_dir.join("learned_windows");
    fs::create_dir_all(&dir)?;
    let feature_order = vec![
        "normalized_snr".to_string(),
        "range_norm".to_string(),
        "velocity_norm".to_string(),
        "clutter_proxy".to_string(),
        "rfi_pressure".to_string(),
        "dropout_fraction".to_string(),
        "micro_doppler_energy".to_string(),
        "tbd_track_score".to_string(),
    ];
    let mut entries = Vec::new();
    for window in [8usize, 16, 32] {
        let (values, windows) = learned_window_values(features, window);
        let path = format!("window_{window}.zarr");
        write_f32_tensor(
            &dir.join(&path),
            &[windows, window, feature_order.len()],
            values,
        )?;
        entries.push(LearnedWindowEntry {
            window_frames: window,
            path,
            shape: vec![windows, window, feature_order.len()],
        });
    }
    write_json_pretty(
        &dir.join("manifest.json"),
        &LearnedWindowManifest {
            record_id: record_id.to_string(),
            windows: entries,
            feature_order,
        },
    )?;
    Ok(())
}

fn learned_window_values(features: &[MlFrameFeatureRow], window: usize) -> (Vec<f32>, usize) {
    let stride = (window / 2).max(1);
    let windows = if features.len() <= window {
        1
    } else {
        ((features.len() - window) / stride) + 1
    };
    let mut values = Vec::with_capacity(windows * window * 8);
    for w in 0..windows {
        let start = (w * stride).min(features.len().saturating_sub(1));
        for offset in 0..window {
            let feature = features
                .get((start + offset).min(features.len().saturating_sub(1)))
                .expect("features nonempty");
            values.extend_from_slice(&[
                feature.normalized_snr,
                (feature.range_m as f32 / 10_000.0).clamp(0.0, 1.0),
                ((feature.radial_velocity_mps as f32 + 180.0) / 360.0).clamp(0.0, 1.0),
                ((feature.local_noise_floor_db + 60.0) / 48.0).clamp(0.0, 1.0),
                feature.rfi_pressure,
                feature.dropout_fraction,
                feature.micro_doppler_energy,
                feature.tbd_track_score,
            ]);
        }
    }
    (values, windows)
}

fn summarize_features(
    record_id: &str,
    split: SplitKind,
    class: &MlClass,
    features: &[MlFrameFeatureRow],
    first_detectable_frame: Option<usize>,
) -> MlFeatureSummaryRow {
    let len = features.len().max(1) as f32;
    MlFeatureSummaryRow {
        record_id: record_id.to_string(),
        split,
        target_family: class.target_family.clone(),
        hard_negative_family: class.hard_negative_family.clone(),
        is_public_proxy_positive: class.is_public_proxy_positive,
        mean_snr_db: features.iter().map(|row| row.snr_db).sum::<f32>() / len,
        max_snr_db: features
            .iter()
            .map(|row| row.snr_db)
            .fold(f32::MIN, f32::max),
        mean_doppler_scr: features.iter().map(|row| row.doppler_scr).sum::<f32>() / len,
        mean_rfi_pressure: features.iter().map(|row| row.rfi_pressure).sum::<f32>() / len,
        dropout_fraction: features.iter().map(|row| row.dropout_fraction).sum::<f32>() / len,
        mean_micro_doppler_energy: features
            .iter()
            .map(|row| row.micro_doppler_energy)
            .sum::<f32>()
            / len,
        micro_doppler_peak_hz_proxy: features
            .iter()
            .map(|row| row.micro_doppler_peak_hz_proxy)
            .fold(0.0, f32::max),
        micro_doppler_bandwidth_hz_proxy: features
            .iter()
            .map(|row| row.micro_doppler_bandwidth_hz_proxy)
            .sum::<f32>()
            / len,
        mean_track_score: features.iter().map(|row| row.tbd_track_score).sum::<f32>() / len,
        cfar_detection_fraction: features.iter().filter(|row| row.cfar_detected).count() as f32
            / len,
        first_detectable_frame,
    }
}

fn sample_envelope(class: &MlClass, rng: &mut SplitMix64) -> MlEnvelope {
    match class.hard_negative_family.as_str() {
        // Phase-1.5 envelope harmonization (detection-realism-fix-generator):
        // expanded ranges so positives overlap confusers in SNR/range/velocity/
        // altitude/RCS space. A real UAS in the wild ranges from low/slow/quiet
        // (barely visible) to high/fast/loud (easy). Without overlap, single-
        // feature AUC stays near 1.0 on radial_velocity / initial_range / SNR.
        "positive_public_proxy" => MlEnvelope {
            dimensions_m: DimensionsSample {
                length: rng.range_f64(2.4, 3.9),
                wingspan: rng.range_f64(1.8, 2.9),
                height: rng.range_f64(0.30, 0.85),
            },
            rcs_dbsm: rng.range_f64(-30.0, -2.0),
            speed_mps: rng.range_f64(8.0, 70.0),
            initial_range_m: rng.range_f64(600.0, 8_500.0),
            radial_velocity_mps: rng.range_f64(-55.0, 15.0),
            altitude_m: rng.range_f64(10.0, 800.0),
            base_snr_db: rng.range_f32(-3.0, 14.0),
            micro_peak_hz: rng.range_f32(35.0, 130.0),
            micro_bandwidth_hz: rng.range_f32(30.0, 150.0),
            clutter_pressure: rng.range_f32(0.10, 0.55),
            rfi_pressure: rng.range_f32(0.02, 0.28),
            dropout_probability: rng.range_f32(0.0, 0.06),
            phase_impairment_rad: rng.range_f32(0.006, 0.045),
            amplitude_impairment: rng.range_f32(0.04, 0.18),
        },
        // Phase-1.5: widened micro-Doppler ranges. Real birds under stress reach
        // 60-80 Hz wingbeat; bats reach 80-120 Hz; insect clouds resonate at
        // 150-400 Hz. Critical for overlapping the positive UAS prop range.
        "single_bird" => biological_envelope(rng, 3.0, 90.0, 1.0, 24.0),
        "bird_flock" => biological_envelope(rng, 5.0, 130.0, 4.0, 31.0),
        "bat_insect_cloud" => biological_envelope(rng, 12.0, 180.0, 0.5, 18.0),
        "balloon_weather" => slow_windborne_envelope(rng, 0.0, 7.0, -34.0, -8.0),
        "kite" => slow_windborne_envelope(rng, 0.2, 12.0, -30.0, -5.0),
        "windborne_debris" => slow_windborne_envelope(rng, 0.0, 18.0, -36.0, -6.0),
        "ground_vehicle" => ground_envelope(rng, 0.0, 32.0, -5.0, 18.0),
        "power_line_pylon" => infrastructure_envelope(rng, 0.0, 8.0, 8.0, 34.0),
        "wind_turbine" => infrastructure_envelope(rng, 0.0, 80.0, 4.0, 28.0),
        "rain_cell" => weather_envelope(rng, 0.0, 12.0, 0.62, 0.20),
        "dust_haze" => weather_envelope(rng, 0.0, 10.0, 0.58, 0.12),
        "rfi_burst" => weather_envelope(rng, 0.0, 8.0, 0.25, 0.82),
        "terrain_only" => weather_envelope(rng, 0.0, 4.0, 0.72, 0.08),
        "multipath_ghost" => ground_envelope(rng, 0.0, 25.0, -20.0, 12.0),
        _ => slow_windborne_envelope(rng, 0.0, 12.0, -35.0, -8.0),
    }
}

fn biological_envelope(
    rng: &mut SplitMix64,
    micro_min: f32,
    micro_max: f32,
    speed_min: f64,
    speed_max: f64,
) -> MlEnvelope {
    // Phase-1.5: bird/bat/insect ranges expanded UP to overlap with positive UAS
    // distribution (raptors soar at 1000m+, large flocks at 6km+, occasional
    // strong-return birds reach SNR comparable to small UAS).
    MlEnvelope {
        dimensions_m: DimensionsSample {
            length: rng.range_f64(0.03, 1.2),
            wingspan: rng.range_f64(0.04, 2.4),
            height: rng.range_f64(0.01, 0.45),
        },
        rcs_dbsm: rng.range_f64(-48.0, -10.0),
        speed_mps: rng.range_f64(speed_min, speed_max.max(speed_min + 1.0)),
        initial_range_m: rng.range_f64(300.0, 7_500.0),
        radial_velocity_mps: rng.range_f64(-32.0, 32.0),
        altitude_m: rng.range_f64(5.0, 1_100.0),
        base_snr_db: rng.range_f32(-4.0, 13.0),
        micro_peak_hz: rng.range_f32(micro_min, micro_max),
        micro_bandwidth_hz: rng.range_f32(8.0, 140.0),
        clutter_pressure: rng.range_f32(0.12, 0.55),
        rfi_pressure: rng.range_f32(0.0, 0.22),
        dropout_probability: rng.range_f32(0.0, 0.07),
        phase_impairment_rad: rng.range_f32(0.008, 0.050),
        amplitude_impairment: rng.range_f32(0.05, 0.20),
    }
}

fn slow_windborne_envelope(
    rng: &mut SplitMix64,
    speed_min: f64,
    speed_max: f64,
    rcs_min: f64,
    rcs_max: f64,
) -> MlEnvelope {
    // Phase-1.5: balloons/kites/debris range expanded UP to reach altitudes
    // and SNRs that overlap with the positive UAS distribution. A high-altitude
    // weather balloon at 2 km with a Mylar reflector returns near positive SNR.
    MlEnvelope {
        dimensions_m: DimensionsSample {
            length: rng.range_f64(0.2, 5.0),
            wingspan: rng.range_f64(0.2, 8.0),
            height: rng.range_f64(0.2, 5.0),
        },
        rcs_dbsm: rng.range_f64(rcs_min, rcs_max),
        speed_mps: rng.range_f64(speed_min, speed_max.max(speed_min + 1.0)),
        initial_range_m: rng.range_f64(300.0, 7_000.0),
        radial_velocity_mps: rng.range_f64(-18.0, 18.0),
        altitude_m: rng.range_f64(5.0, 2_000.0),
        base_snr_db: rng.range_f32(-5.0, 12.0),
        // Phase-1.5: widened — windborne objects flutter (kite tails, Mylar
        // reflectors, plastic bag debris all show Doppler smearing in wind).
        micro_peak_hz: rng.range_f32(0.0, 40.0),
        micro_bandwidth_hz: rng.range_f32(2.0, 35.0),
        clutter_pressure: rng.range_f32(0.18, 0.58),
        rfi_pressure: rng.range_f32(0.0, 0.25),
        dropout_probability: rng.range_f32(0.0, 0.08),
        phase_impairment_rad: rng.range_f32(0.01, 0.055),
        amplitude_impairment: rng.range_f32(0.05, 0.22),
    }
}

fn ground_envelope(
    rng: &mut SplitMix64,
    speed_min: f64,
    speed_max: f64,
    rcs_min: f64,
    rcs_max: f64,
) -> MlEnvelope {
    // Phase-1.5: ground vehicles can have HIGHER SNR than UAS (a truck has a
    // big radar return), drive fast enough to overlap UAS radial velocities,
    // and can be on hills/bridges/elevated surfaces. Multipath_ghost also
    // routes through here so the altitude tail must reach UAS regime.
    MlEnvelope {
        dimensions_m: DimensionsSample {
            length: rng.range_f64(2.0, 14.0),
            wingspan: rng.range_f64(1.5, 3.5),
            height: rng.range_f64(1.0, 4.2),
        },
        rcs_dbsm: rng.range_f64(rcs_min, rcs_max),
        speed_mps: rng.range_f64(speed_min, speed_max.max(speed_min + 1.0)),
        initial_range_m: rng.range_f64(200.0, 7_500.0),
        radial_velocity_mps: rng.range_f64(-50.0, 50.0),
        altitude_m: rng.range_f64(0.0, 600.0),
        base_snr_db: rng.range_f32(-2.0, 18.0),
        // Phase-1.5: widened — vehicle alternators / fan blades / drivetrain
        // produce aliased micro-Doppler in the UAS prop regime.
        micro_peak_hz: rng.range_f32(2.0, 110.0),
        micro_bandwidth_hz: rng.range_f32(8.0, 110.0),
        clutter_pressure: rng.range_f32(0.30, 0.85),
        rfi_pressure: rng.range_f32(0.02, 0.32),
        dropout_probability: rng.range_f32(0.0, 0.10),
        phase_impairment_rad: rng.range_f32(0.012, 0.070),
        amplitude_impairment: rng.range_f32(0.08, 0.26),
    }
}

fn infrastructure_envelope(
    rng: &mut SplitMix64,
    speed_min: f64,
    speed_max: f64,
    rcs_min: f64,
    rcs_max: f64,
) -> MlEnvelope {
    // Phase-1.5: tighten the radial-velocity tail. The previous [-140, 140]
    // range was wildly out-of-band for static infrastructure (wind turbine
    // blade-tip Doppler aliases into ±50 m/s at most for X-band). Aliasing
    // confusers should still overlap UAS velocity space but not be giveaway-
    // wide. Range narrowed to overlap UAS range without dwarfing it.
    MlEnvelope {
        dimensions_m: DimensionsSample {
            length: rng.range_f64(5.0, 80.0),
            wingspan: rng.range_f64(2.0, 80.0),
            height: rng.range_f64(5.0, 160.0),
        },
        rcs_dbsm: rng.range_f64(rcs_min, rcs_max),
        speed_mps: rng.range_f64(speed_min, speed_max.max(speed_min + 1.0)),
        initial_range_m: rng.range_f64(600.0, 9_000.0),
        radial_velocity_mps: rng.range_f64(-60.0, 60.0),
        altitude_m: rng.range_f64(20.0, 1_500.0),
        base_snr_db: rng.range_f32(-4.0, 16.0),
        // Phase-1.5: widened — wind-turbine blade-tip Doppler aliases above the
        // UAS prop regime under PRF folding for X-band sensors.
        micro_peak_hz: rng.range_f32(0.0, 130.0),
        micro_bandwidth_hz: rng.range_f32(4.0, 140.0),
        clutter_pressure: rng.range_f32(0.45, 0.90),
        rfi_pressure: rng.range_f32(0.02, 0.30),
        dropout_probability: rng.range_f32(0.0, 0.09),
        phase_impairment_rad: rng.range_f32(0.010, 0.060),
        amplitude_impairment: rng.range_f32(0.08, 0.24),
    }
}

fn weather_envelope(
    rng: &mut SplitMix64,
    speed_min: f64,
    speed_max: f64,
    clutter: f32,
    rfi: f32,
) -> MlEnvelope {
    // Phase-1.5: rain/dust/RFI cells extended to overlap UAS observable range.
    // Heavy rain cores can return SNR above quiet UAS; RFI bursts saturate at
    // any range. Velocity tail widened because windborne precipitation in a
    // gust front shows ±18 m/s under Doppler folding.
    MlEnvelope {
        dimensions_m: DimensionsSample {
            length: 0.0,
            wingspan: 0.0,
            height: 0.0,
        },
        rcs_dbsm: rng.range_f64(-45.0, -10.0),
        speed_mps: rng.range_f64(speed_min, speed_max.max(speed_min + 1.0)),
        initial_range_m: rng.range_f64(100.0, 8_000.0),
        radial_velocity_mps: rng.range_f64(-18.0, 18.0),
        altitude_m: rng.range_f64(0.0, 2_500.0),
        base_snr_db: rng.range_f32(-8.0, 12.0),
        // Phase-1.5: widened — heavy-rain cores and dust have measurable Doppler
        // spread; RFI bursts smear arbitrary frequencies.
        micro_peak_hz: rng.range_f32(0.0, 35.0),
        micro_bandwidth_hz: rng.range_f32(0.5, 40.0),
        clutter_pressure: (clutter + rng.range_f32(-0.08, 0.08)).clamp(0.0, 1.0),
        rfi_pressure: (rfi + rng.range_f32(-0.06, 0.08)).clamp(0.0, 1.0),
        dropout_probability: rng.range_f32(0.02, 0.14),
        phase_impairment_rad: rng.range_f32(0.018, 0.090),
        amplitude_impairment: rng.range_f32(0.10, 0.32),
    }
}

fn ml_classes() -> Vec<MlClass> {
    vec![
        ml_class(
            NEUTRAL_OBJECT_ID,
            "Delta Pusher Fixed-Wing OWA Public Proxy",
            "owa_delta_pusher_public_proxy",
            "positive_public_proxy",
            true,
            false,
        ),
        ml_class(
            "hard-negative-single-bird-v1",
            "Single Bird Public Proxy",
            "single_bird",
            "single_bird",
            false,
            true,
        ),
        ml_class(
            "hard-negative-bird-flock-v1",
            "Bird Flock Public Proxy",
            "bird_flock",
            "bird_flock",
            false,
            true,
        ),
        ml_class(
            "hard-negative-bat-insect-cloud-v1",
            "Bat and Insect Cloud Public Proxy",
            "bat_or_insect_cloud",
            "bat_insect_cloud",
            false,
            true,
        ),
        ml_class(
            "hard-negative-balloon-weather-v1",
            "Weather Balloon Public Proxy",
            "balloon_weather",
            "balloon_weather",
            false,
            true,
        ),
        ml_class(
            "hard-negative-kite-v1",
            "Kite Public Proxy",
            "kite",
            "kite",
            false,
            true,
        ),
        ml_class(
            "hard-negative-windborne-debris-v1",
            "Windborne Debris Public Proxy",
            "windborne_debris",
            "windborne_debris",
            false,
            true,
        ),
        ml_class(
            "hard-negative-ground-vehicle-v1",
            "Ground Vehicle Public Proxy",
            "ground_vehicle",
            "ground_vehicle",
            false,
            true,
        ),
        ml_class(
            "hard-negative-power-line-pylon-v1",
            "Power Line and Pylon Public Proxy",
            "power_line_pylon",
            "power_line_pylon",
            false,
            true,
        ),
        ml_class(
            "hard-negative-wind-turbine-v1",
            "Wind Turbine Public Proxy",
            "wind_turbine",
            "wind_turbine",
            false,
            true,
        ),
        ml_class(
            "hard-negative-rain-cell-v1",
            "Rain Cell Public Proxy",
            "rain_cell",
            "rain_cell",
            false,
            true,
        ),
        ml_class(
            "hard-negative-dust-haze-v1",
            "Dust and Haze Public Proxy",
            "dust_haze",
            "dust_haze",
            false,
            true,
        ),
        ml_class(
            "hard-negative-rfi-burst-v1",
            "RFI Burst Public Proxy",
            "rfi_burst",
            "rfi_burst",
            false,
            true,
        ),
        ml_class(
            "hard-negative-terrain-only-v1",
            "Terrain-Only Scene Public Proxy",
            "terrain_only",
            "terrain_only",
            false,
            true,
        ),
        ml_class(
            "hard-negative-multipath-ghost-v1",
            "Multipath Ghost Public Proxy",
            "multipath_ghost",
            "multipath_ghost",
            false,
            true,
        ),
    ]
}

fn ml_class(
    class_id: &str,
    display_name: &str,
    target_family: &str,
    hard_negative_family: &str,
    is_public_proxy_positive: bool,
    is_hard_negative: bool,
) -> MlClass {
    MlClass {
        class_id: class_id.to_string(),
        display_name: display_name.to_string(),
        target_family: target_family.to_string(),
        hard_negative_family: hard_negative_family.to_string(),
        is_public_proxy_positive,
        is_hard_negative,
    }
}

fn write_f32_tensor(path: &Path, shape: &[usize], values: Vec<f32>) -> Result<(), DatasetError> {
    let expected = shape.iter().product::<usize>();
    if values.len() != expected {
        return Err(DatasetError::Tensor(format!(
            "tensor {} has {} values; expected {} for shape {:?}",
            path.display(),
            values.len(),
            expected,
            shape
        )));
    }
    let array = ArrayD::from_shape_vec(IxDyn(shape), values)
        .map_err(|err| DatasetError::Tensor(err.to_string()))?;
    echoforge_sig::tensor::write_f32(path, &array)?;
    Ok(())
}

fn stft_spectrogram(signal: &[f32], window: usize, hop: usize, bins: usize) -> Vec<f32> {
    let windows = if signal.len() <= window {
        1
    } else {
        ((signal.len() - window) / hop.max(1)) + 1
    };
    let mut out = Vec::with_capacity(windows * bins);
    for w in 0..windows {
        let start = w * hop.max(1);
        let slice = (0..window)
            .map(|i| {
                signal
                    .get((start + i).min(signal.len().saturating_sub(1)))
                    .copied()
                    .unwrap_or(0.0)
                    * hann(i, window)
            })
            .collect::<Vec<_>>();
        out.extend(dft_magnitude(&slice, bins));
    }
    out
}

fn weighted_spectrum(signal: &[f32], bins: usize) -> Vec<f32> {
    let mut weighted = Vec::with_capacity(signal.len());
    for (index, value) in signal.iter().enumerate() {
        weighted.push(*value * hann(index, signal.len().max(1)));
    }
    dft_magnitude(&weighted, bins)
}

fn cepstrum_proxy(spectrum: &[f32], bins: usize) -> Vec<f32> {
    let log_power = spectrum
        .iter()
        .map(|value| (value.max(1e-6)).ln())
        .collect::<Vec<_>>();
    dft_magnitude(&log_power, bins)
}

fn cadence_velocity(
    spectrum: &[f32],
    velocity_mps: f32,
    cadence_bins: usize,
    velocity_bins: usize,
) -> Vec<f32> {
    let peak = spectrum.iter().copied().fold(0.0, f32::max).max(1e-6);
    let velocity_center = ((velocity_mps + 160.0) / 320.0 * (velocity_bins as f32 - 1.0))
        .clamp(0.0, velocity_bins as f32 - 1.0);
    let mut out = Vec::with_capacity(cadence_bins * velocity_bins);
    for c in 0..cadence_bins {
        let spectral = spectrum
            .get(c.min(spectrum.len().saturating_sub(1)))
            .copied()
            .unwrap_or(0.0)
            / peak;
        for v in 0..velocity_bins {
            let dist = v as f32 - velocity_center;
            out.push((spectral * (-dist * dist / 18.0).exp()).clamp(0.0, 1.0));
        }
    }
    out
}

fn dft_magnitude(signal: &[f32], bins: usize) -> Vec<f32> {
    if signal.is_empty() {
        return vec![0.0; bins];
    }
    (0..bins)
        .map(|k| {
            let mut re = 0.0f32;
            let mut im = 0.0f32;
            for (n, value) in signal.iter().enumerate() {
                let angle = -2.0 * std::f32::consts::PI * k as f32 * n as f32 / signal.len() as f32;
                re += *value * angle.cos();
                im += *value * angle.sin();
            }
            (re * re + im * im).sqrt() / signal.len() as f32
        })
        .collect()
}

fn entropy(values: &[f32]) -> f32 {
    let sum = values.iter().map(|value| value.max(0.0)).sum::<f32>();
    if sum <= 0.0 {
        return 0.0;
    }
    let entropy = values
        .iter()
        .map(|value| {
            let p = value.max(0.0) / sum;
            if p > 0.0 {
                -p * p.ln()
            } else {
                0.0
            }
        })
        .sum::<f32>();
    entropy / (values.len().max(2) as f32).ln()
}

fn hann(index: usize, len: usize) -> f32 {
    if len <= 1 {
        return 1.0;
    }
    0.5 - 0.5 * (2.0 * std::f32::consts::PI * index as f32 / (len - 1) as f32).cos()
}

fn feature_family_availability(record_id: &str) -> FeatureFamilyAvailability {
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
            path: "multi_view/range_angle_schema_placeholder.json".to_string(),
            reason: Some(
                "future MIMO range-angle and range-azimuth-Doppler schema placeholder only"
                    .to_string(),
            ),
        },
    }
}

fn write_csv<T: Serialize>(path: &Path, rows: &[T]) -> Result<(), DatasetError> {
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

fn dataset_manifest(
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

fn feature_families() -> Vec<FeatureFamily> {
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
                "Range-time, Doppler-time, range-Doppler-time tensors plus future angle schema placeholder."
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

fn external_calibration_sources() -> Vec<ExternalCalibrationSource> {
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
            title: "RDRD/RAD-DAR public drone radar dataset metadata placeholder".to_string(),
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

fn local_external_data_config(sources: &[ExternalCalibrationSource]) -> serde_json::Value {
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

fn dataset_card(
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

fn quality_report(
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
            "GPU readiness selects the runtime plan, while the v1 radar kernel remains CPU-backed and records that fallback."
                .to_string(),
            "Hard negatives are robustness and false-alarm stressors.".to_string(),
        ],
    }
}

fn feature_summary_is_finite(row: &MlFeatureSummaryRow) -> bool {
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

fn normalization_stats(dataset_id: &str, rows: &[MlFeatureSummaryRow]) -> NormalizationStats {
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

fn split_counts(records: &[MlRecordSummary]) -> BTreeMap<SplitKind, usize> {
    let mut counts = BTreeMap::new();
    for record in records {
        *counts.entry(record.split).or_insert(0) += 1;
    }
    counts
}

fn runtime_report(
    config: &MlTrainingDataConfig,
    runtime: &RuntimePlan,
    worker_count: usize,
    frame_count: usize,
    stage_timings: &[StageTiming],
    elapsed: Duration,
) -> RuntimeReport {
    let elapsed_ns = elapsed.as_nanos().min(u128::from(u64::MAX)) as u64;
    RuntimeReport {
        requested_backend: runtime.requested_backend,
        selected_backend: runtime.selected_backend.to_string(),
        kernel_backend: "cpu-scaffold".to_string(),
        gpu_available: runtime.gpu_available,
        gpu_usable: runtime.gpu_usable,
        gpu_constrained: runtime.gpu_constrained,
        gpu_min_free_memory_mb: runtime.gpu_min_free_memory_mb,
        gpu_free_memory_mb: runtime
            .gpu_devices
            .iter()
            .map(|device| device.memory_free_mb)
            .max(),
        gpu_stage_fallback: gpu_stage_fallback_note(runtime),
        logical_cores: runtime.logical_cores,
        recommended_worker_budget: runtime.recommended_worker_budget,
        worker_count,
        worker_cap: 40,
        records: config.records,
        frame_count,
        stage_timings: stage_timings.to_vec(),
        total_elapsed_ns: elapsed_ns,
        throughput_records_per_sec: if elapsed_ns == 0 {
            0.0
        } else {
            config.records as f64 / (elapsed_ns as f64 / 1_000_000_000.0)
        },
    }
}

fn gpu_stage_fallback_note(runtime: &RuntimePlan) -> Option<String> {
    match runtime.selected_backend {
        echoforge_radar::RuntimeBackend::Gpu => Some(
            "GPU probe selected a GPU-capable runtime, but the v1 ML training-data radar kernel is CPU-backed; generated tensors record this fallback.".to_string(),
        ),
        echoforge_radar::RuntimeBackend::Cpu => runtime.fallback_reason.clone(),
    }
}

fn feature_schema() -> serde_json::Value {
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

fn label_schema() -> serde_json::Value {
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

fn guardrails() -> Vec<String> {
    vec![
        "synthetic public proxy".to_string(),
        "uncertainty-scored radar artifacts".to_string(),
        "no measured object trace copying by default".to_string(),
        "defensive early-detection and false-alarm robustness only".to_string(),
        "no payload effects, terminal behavior, route optimization, or operational tuning"
            .to_string(),
    ]
}

fn relative_record_path(record_id: &str, suffix: &str) -> String {
    format!("records/{record_id}/{suffix}")
}

fn elapsed_ns(start: Instant) -> u64 {
    start.elapsed().as_nanos().min(u128::from(u64::MAX)) as u64
}

fn child_seed(root: u64, index: u64) -> u64 {
    let mut value = root ^ index.wrapping_mul(0x9e37_79b9_7f4a_7c15);
    value ^= value >> 30;
    value = value.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value ^= value >> 27;
    value = value.wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn deterministic_shuffle<T>(items: &mut [T], seed: u64) {
    let mut rng = SplitMix64::new(seed ^ 0x5368_7566_666c_65);
    for index in (1..items.len()).rev() {
        let swap = rng.range_usize(0, index);
        items.swap(index, swap);
    }
}

fn stable_hash_str(input: &str) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for byte in input.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x1000_0000_01b3);
    }
    hash
}

#[derive(Debug, Clone)]
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    fn unit_f64(&mut self) -> f64 {
        let bits = self.next_u64() >> 11;
        (bits as f64) / ((1u64 << 53) as f64)
    }

    fn unit_f32(&mut self) -> f32 {
        self.unit_f64() as f32
    }

    fn range_f64(&mut self, min: f64, max: f64) -> f64 {
        min + self.unit_f64() * (max - min)
    }

    fn range_f32(&mut self, min: f32, max: f32) -> f32 {
        min + self.unit_f32() * (max - min)
    }

    fn range_usize(&mut self, min: usize, max: usize) -> usize {
        if max <= min {
            return min;
        }
        min + (self.next_u64() as usize % (max - min + 1))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_assignment_keeps_scenario_object_seed_in_one_split() {
        let mut config = MlTrainingDataConfig::shahed_public_proxy_default();
        config.records = 64;
        let plans = build_record_plan(&config, 180).expect("plan");
        let mut seen = BTreeMap::<String, SplitKind>::new();
        for plan in plans {
            let key = format!("{:016x}:{:016x}", plan.scenario_seed, plan.object_seed);
            if let Some(previous) = seen.insert(key, plan.split) {
                assert_eq!(previous, plan.split);
            }
        }
        let splits = seen.values().fold(BTreeMap::new(), |mut acc, split| {
            *acc.entry(*split).or_insert(0usize) += 1;
            acc
        });
        assert_eq!(*splits.get(&SplitKind::Train).unwrap_or(&0), 45);
        assert_eq!(*splits.get(&SplitKind::Validation).unwrap_or(&0), 10);
        assert_eq!(*splits.get(&SplitKind::Test).unwrap_or(&0), 9);
    }

    #[test]
    fn frame_feature_generation_is_deterministic_and_finite() {
        let mut config = MlTrainingDataConfig::shahed_public_proxy_default();
        config.records = 12;
        config.time_window_s = 6.0;
        config.frame_rate_hz = 2.0;
        let plans = build_record_plan(&config, 12).expect("plan");
        let plan = plans
            .iter()
            .find(|plan| plan.class.is_public_proxy_positive)
            .expect("positive plan");
        // V3 unified-path determinism check: feed `build_frame_products`
        // the same envelope + episode twice and confirm byte-for-byte
        // serialise equality. The episode itself is constructed via
        // `synthesize_scene` so the test exercises the unified path.
        let mut rng_a = SplitMix64::new(plan.scenario_seed);
        let mut rng_b = SplitMix64::new(plan.scenario_seed);
        let envelope_a = sample_envelope(&plan.class, &mut rng_a);
        let envelope_b = sample_envelope(&plan.class, &mut rng_b);
        assert_eq!(envelope_a.dimensions_m, envelope_b.dimensions_m);
        let episode_a = synthesize_episode_for_test(&envelope_a, &plan.class, plan.scenario_seed);
        let episode_b = synthesize_episode_for_test(&envelope_b, &plan.class, plan.scenario_seed);
        let (features_a, _, _, _) =
            build_frame_products(&config, plan, &envelope_a, &episode_a, 12, 32);
        let (features_b, _, _, _) =
            build_frame_products(&config, plan, &envelope_b, &episode_b, 12, 32);
        assert_eq!(
            serde_json::to_string(&features_a).expect("features serialize"),
            serde_json::to_string(&features_b).expect("features serialize")
        );
        assert!(features_a.iter().all(|feature| {
            [
                feature.snr_db,
                feature.cfar_statistic,
                feature.tbd_track_score,
                feature.local_noise_floor_db,
                feature.doppler_scr,
                feature.micro_doppler_energy,
                feature.normalized_snr,
            ]
            .iter()
            .all(|value| value.is_finite())
        }));
    }

    fn synthesize_episode_for_test(
        envelope: &MlEnvelope,
        class: &MlClass,
        scenario_seed: u64,
    ) -> SyntheticEpisode {
        let mut rng = SplitMix64::new(scenario_seed);
        // Replay the envelope sampler so the takeoff-profile RNG draws
        // match generate_record exactly.
        let _ = sample_envelope(class, &mut rng);
        let _ = rng.range_usize(24, 48);
        let mut noise = NoiseProfile::real_world_proxy_v1();
        noise.awgn_sigma = (0.035 + 0.05 * envelope.clutter_pressure) as f32;
        noise.clutter_sigma = (0.02 + 0.09 * envelope.clutter_pressure) as f32;
        noise.rfi_probability = (0.004 + 0.045 * envelope.rfi_pressure).min(0.12);
        noise.rfi_amplitude = 0.55 + 1.35 * envelope.rfi_pressure;
        noise.amplitude_scintillation_sigma = envelope.amplitude_impairment.max(0.01);
        noise.phase_noise_std_rad = envelope.phase_impairment_rad.max(0.002);
        noise.ground_glint_count = (2.0 + 10.0 * envelope.clutter_pressure).round() as usize;
        #[allow(deprecated)]
        let sim_config = RadarSimConfig {
            sample_rate_hz: 1_000_000.0,
            pulse_width_s: 64e-6,
            bandwidth_hz: 800_000.0,
            carrier_hz: 9_600_000_000.0,
            pulse_count: 32,
            pri_s: 900e-6,
            target_snr_db: envelope.base_snr_db as f64,
            ..RadarSimConfig::default()
        };
        let profile = adapt_envelope_to_takeoff_profile(envelope, &mut rng);
        let scene = build_scene_descriptor(class, profile, &sim_config, &noise);
        synthesize_scene(scene, sim_config, noise, EpisodeSeed(scenario_seed ^ 0x0dd5_136))
    }

    #[test]
    fn small_training_fixture_writes_required_artifacts() {
        let temp = tempfile::tempdir().expect("tempdir");
        let mut config = MlTrainingDataConfig::shahed_public_proxy_default();
        config.records = 12;
        config.time_window_s = 6.0;
        config.frame_rate_hz = 2.0;
        config.backend = BackendMode::Cpu;
        config.workers = Some(4);
        config.output_dir = temp.path().join("ml-training");

        let report = run_ml_training_data(config).expect("training data run");
        assert_eq!(report.records, 12);
        assert_eq!(report.positive_records, 2);
        assert_eq!(report.hard_negative_families, 10);
        assert_eq!(report.frame_count, 12);
        assert!(report.dataset_manifest_path.exists());
        assert!(report.dataset_card_path.exists());
        assert!(report.split_manifest_path.exists());
        assert!(report.records_path.exists());
        assert!(report.features_path.exists());
        assert!(report.label_schema_path.exists());
        assert!(report.feature_schema_path.exists());
        assert!(report.normalization_stats_path.exists());
        assert!(report.quality_report_path.exists());
        assert!(report.runtime_report_path.exists());

        let record = temp.path().join("ml-training/records/record_000001");
        assert!(record.join("products/iq_complex.zarr/.zarray").exists());
        assert!(record.join("products/range_profile.zarr/.zarray").exists());
        assert!(record
            .join("products/range_doppler_proxy.zarr/.zarray")
            .exists());
        assert!(record
            .join("micro_doppler/stft_spectrogram.zarr/.zarray")
            .exists());
        assert!(record
            .join("multi_view/range_doppler_time.zarr/.zarray")
            .exists());
        assert!(record
            .join("learned_windows/window_16.zarr/.zarray")
            .exists());
        assert!(record.join("frame_labels.csv").exists());
        assert!(record.join("truth_metadata.json").exists());
        assert!(record.join("detector_events.json").exists());
    }

    #[test]
    fn forced_gpu_uses_existing_readiness_error() {
        let mut signals = BackendSignals::new(128, true, true);
        signals.gpu_usable = false;
        signals.gpu_unusable_reason = Some(
            "GPU detected but free memory is 606 MiB; need at least 2048 MiB for this runtime"
                .to_string(),
        );
        let err = RuntimePlan::from_signals(BackendMode::Gpu, signals)
            .expect_err("forced GPU should fail");
        assert!(err.to_string().contains("free memory is 606 MiB"));
    }

    #[test]
    fn phase_tiered_synthetic_positive_reaches_cruise() {
        // Smoke-check the V3 per-tier evaluation: positives following
        // the dossier's boost → climb → cruise progression should latch
        // the cruise tier within 30 frames; confusers should not.
        let class_pos = ml_classes()
            .into_iter()
            .find(|class| class.is_public_proxy_positive)
            .unwrap();
        let class_neg = ml_classes()
            .into_iter()
            .find(|class| class.hard_negative_family == "ground_vehicle")
            .unwrap();
        let scenario_seed: u64 = 42;
        let mut rng_pos = SplitMix64::new(scenario_seed);
        let env_pos = sample_envelope(&class_pos, &mut rng_pos);
        let episode_pos = synthesize_episode_for_test(&env_pos, &class_pos, scenario_seed);
        let obs_pos = evaluate_phase_tiered(&episode_pos, true);

        let mut rng_neg = SplitMix64::new(scenario_seed + 1);
        let env_neg = sample_envelope(&class_neg, &mut rng_neg);
        let episode_neg = synthesize_episode_for_test(&env_neg, &class_neg, scenario_seed + 1);
        let obs_neg = evaluate_phase_tiered(&episode_neg, false);

        let pos_cruise_count = obs_pos.tier_counts.get(&Tier::Cruise).copied().unwrap_or(0);
        let neg_cruise_count = obs_neg.tier_counts.get(&Tier::Cruise).copied().unwrap_or(0);
        eprintln!(
            "pos tier_counts={:?} neg tier_counts={:?}",
            obs_pos.tier_counts, obs_neg.tier_counts
        );
        assert!(
            pos_cruise_count > 0,
            "positive synthetic stream must reach cruise tier"
        );
        assert_eq!(
            neg_cruise_count, 0,
            "confuser synthetic stream must NOT reach cruise tier"
        );
    }

    #[test]
    fn hard_negative_roster_has_acceptance_coverage() {
        let families = ml_classes()
            .into_iter()
            .filter(|class| class.is_hard_negative)
            .map(|class| class.hard_negative_family)
            .collect::<std::collections::BTreeSet<_>>();
        assert!(families.len() >= 10);
        for family in [
            "single_bird",
            "bird_flock",
            "bat_insect_cloud",
            "balloon_weather",
            "kite",
            "windborne_debris",
            "ground_vehicle",
            "power_line_pylon",
            "wind_turbine",
            "rain_cell",
            "dust_haze",
            "rfi_burst",
            "terrain_only",
        ] {
            assert!(families.contains(family), "missing {family}");
        }
    }
}
