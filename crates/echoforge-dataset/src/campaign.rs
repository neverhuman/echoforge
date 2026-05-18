use std::collections::{BTreeMap, VecDeque};
use std::fs;
use std::io::{self, IsTerminal};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use echoforge_core::deterministic_id;
use echoforge_core::models::{
    DatasetCard, DatasetSplits, LicenseInfo, Provenance, ValidationCheck, ValidationInfo,
};
use echoforge_radar::{
    apply_clutter_to_profile, apply_receiver_impairments, apply_rfi_to_profile,
    sample_clutter_frame, sample_receiver_impairments, sample_rfi_frame,
    synthesize_takeoff_episode, BackendMode, BackendSignals, ClutterProfile, EpisodeSeed,
    NoiseProfile, RadarSimConfig, ReceiverImpairmentProfile, RfiProfile, RuntimePlan,
    SyntheticEpisode, TakeoffProfile,
};
use indicatif::{ProgressBar, ProgressStyle};
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::export::{write_episode_tensors, write_json_pretty};
use crate::monte_carlo::{DatasetError, StageTiming};

pub const DEFAULT_CAMPAIGN_REQUEST_ID: &str = "shahed136-public-proxy-early-detection-v1";
pub const NEUTRAL_CAMPAIGN_ID: &str = "owa-delta-pusher-public-proxy-early-detection-v1";
pub const OWA_DELTA_OBJECT_ID: &str = "owa-delta-pusher-fixed-wing-public-proxy-v1";
pub const DEFAULT_CAMPAIGN_OUTPUT: &str =
    "outputs/campaigns/shahed136-public-proxy-early-detection-v1";

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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CampaignRecordPlan {
    pub index: usize,
    pub record_id: String,
    pub seed: u64,
    pub class: CampaignClass,
}

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

pub trait StreamingDetector {
    fn model_id(&self) -> &'static str;
    fn update(&mut self, frame: &FrameFeature) -> DetectionState;
}

#[derive(Debug, Clone)]
struct CfarTrackerBaseline {
    threshold: f32,
    consecutive: usize,
    first_trigger: Option<FirstTriggerEvent>,
}

impl CfarTrackerBaseline {
    fn new(threshold: f32) -> Self {
        Self {
            threshold,
            consecutive: 0,
            first_trigger: None,
        }
    }
}

impl StreamingDetector for CfarTrackerBaseline {
    fn model_id(&self) -> &'static str {
        "cfar_tracker_baseline"
    }

    fn update(&mut self, frame: &FrameFeature) -> DetectionState {
        let snr_term = ((frame.snr_db - 5.0) / 14.0).clamp(0.0, 1.0);
        let persistence = (frame.track_persistence_s / 4.0).clamp(0.0, 1.0);
        let clutter_penalty = 0.35 * frame.clutter_pressure + 0.25 * frame.rfi_pressure;
        let confidence =
            (0.12 + 0.58 * snr_term + 0.35 * persistence - clutter_penalty).clamp(0.0, 1.0);
        detection_update(
            self.model_id(),
            frame,
            confidence,
            self.threshold,
            &mut self.consecutive,
            &mut self.first_trigger,
        )
    }
}

#[derive(Debug, Clone)]
struct FeatureTreeClassifier {
    threshold: f32,
    consecutive: usize,
    first_trigger: Option<FirstTriggerEvent>,
}

impl FeatureTreeClassifier {
    fn new(threshold: f32) -> Self {
        Self {
            threshold,
            consecutive: 0,
            first_trigger: None,
        }
    }
}

impl StreamingDetector for FeatureTreeClassifier {
    fn model_id(&self) -> &'static str {
        "feature_tree_classifier"
    }

    fn update(&mut self, frame: &FrameFeature) -> DetectionState {
        let speed_like = if (35.0..=75.0).contains(&frame.range_rate_mps.abs()) {
            0.28
        } else {
            0.04
        };
        let micro_like = if (25.0..=130.0).contains(&frame.micro_doppler_modulation) {
            0.24
        } else {
            0.05
        };
        let area_like = if (2.0..=18.0).contains(&frame.blob_area_bins) {
            0.18
        } else {
            0.04
        };
        let snr_like = ((frame.snr_db - 4.0) / 18.0).clamp(0.0, 0.28);
        let clutter_penalty = 0.24 * frame.clutter_pressure + 0.18 * frame.rfi_pressure;
        let confidence = (0.08 + speed_like + micro_like + area_like + snr_like - clutter_penalty)
            .clamp(0.0, 1.0);
        detection_update(
            self.model_id(),
            frame,
            confidence,
            self.threshold,
            &mut self.consecutive,
            &mut self.first_trigger,
        )
    }
}

#[derive(Debug, Clone)]
struct TemporalTinyModel {
    threshold: f32,
    window: VecDeque<f32>,
    consecutive: usize,
    first_trigger: Option<FirstTriggerEvent>,
}

impl TemporalTinyModel {
    fn new(threshold: f32) -> Self {
        Self {
            threshold,
            window: VecDeque::with_capacity(16),
            consecutive: 0,
            first_trigger: None,
        }
    }
}

impl StreamingDetector for TemporalTinyModel {
    fn model_id(&self) -> &'static str {
        "temporal_tiny_model"
    }

    fn update(&mut self, frame: &FrameFeature) -> DetectionState {
        let instantaneous = (0.42 * ((frame.snr_db - 3.0) / 18.0).clamp(0.0, 1.0)
            + 0.28 * (frame.track_persistence_s / 6.0).clamp(0.0, 1.0)
            + 0.18 * (((frame.range_rate_mps.abs() - 20.0) / 55.0).clamp(0.0, 1.0) as f32)
            + 0.12 * (frame.micro_doppler_modulation / 140.0).clamp(0.0, 1.0)
            - 0.25 * frame.rfi_pressure)
            .clamp(0.0, 1.0);
        if self.window.len() == 16 {
            self.window.pop_front();
        }
        self.window.push_back(instantaneous);
        let mean = self.window.iter().sum::<f32>() / self.window.len().max(1) as f32;
        let confidence = (0.25 * instantaneous + 0.75 * mean).clamp(0.0, 1.0);
        detection_update(
            self.model_id(),
            frame,
            confidence,
            self.threshold,
            &mut self.consecutive,
            &mut self.first_trigger,
        )
    }
}

fn detection_update(
    model_id: &str,
    frame: &FrameFeature,
    confidence: f32,
    threshold: f32,
    consecutive: &mut usize,
    first_trigger: &mut Option<FirstTriggerEvent>,
) -> DetectionState {
    if confidence >= threshold {
        *consecutive += 1;
    } else {
        *consecutive = 0;
    }
    let event = if first_trigger.is_none() && *consecutive >= 2 {
        let event = FirstTriggerEvent {
            model_id: model_id.to_string(),
            frame_index: frame.frame_index,
            time_s: frame.time_s,
            confidence,
            threshold,
            consecutive_frames: *consecutive,
        };
        *first_trigger = Some(event.clone());
        Some(event)
    } else {
        None
    };

    let mut probabilities = BTreeMap::new();
    probabilities.insert("owa_delta_pusher_public_proxy".to_string(), confidence);
    probabilities.insert("other_or_hard_negative".to_string(), 1.0 - confidence);
    DetectionState {
        model_id: model_id.to_string(),
        frame_index: frame.frame_index,
        confidence,
        class_probabilities: probabilities,
        first_trigger_event: event,
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ModelPredictionRow {
    record_id: String,
    model_id: String,
    frame_index: usize,
    time_s: f64,
    confidence: f32,
    class_probabilities_json: String,
    triggered_on_this_frame: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct RecordTruthMetadata {
    record_id: String,
    neutral_object_id: String,
    target_family: String,
    is_shahed_public_proxy: bool,
    is_hard_negative: bool,
    hard_negative_family: String,
    source_dossier_ref: String,
    dimensions_m: DimensionsSample,
    rcs_dbsm_proxy: f64,
    cruise_speed_mps: f64,
    cpi_pulses: usize,
    frame_count: usize,
    time_window_s: f64,
    first_detectable_frame: Option<usize>,
    confidence_threshold: f32,
    guardrails: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct DimensionsSample {
    length: f64,
    wingspan: f64,
    height: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CampaignRecordSummary {
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
    max_confidence_by_model: BTreeMap<String, f32>,
    first_trigger_by_model: BTreeMap<String, Option<usize>>,
}

#[derive(Debug, Clone)]
struct CampaignRecordOutput {
    summary: CampaignRecordSummary,
    events: Vec<FirstTriggerEvent>,
    predictions: Vec<ModelPredictionRow>,
}

pub fn run_monte_carlo_campaign(config: CampaignConfig) -> Result<CampaignReport, DatasetError> {
    validate_campaign_config(&config)?;
    refuse_campaign_output_path(&config.output_dir)?;
    if config.output_dir.exists() {
        return Err(DatasetError::RefusingOutputPath(format!(
            "{} already exists",
            config.output_dir.display()
        )));
    }

    let overall_start = Instant::now();
    let runtime = RuntimePlan::from_signals(config.backend, BackendSignals::detect())?;
    let worker_count = effective_worker_count(&config, &runtime);
    let progress_enabled = config.progress && io::stderr().is_terminal();
    let frame_count = ((config.time_window_s * config.frame_rate_hz).round() as usize).max(1);
    fs::create_dir_all(&config.output_dir)?;

    let mut stage_timings = Vec::new();
    let plan_start = Instant::now();
    let record_plans = build_campaign_plan(&config)?;
    stage_timings.push(StageTiming {
        stage: "campaign_plan".to_string(),
        elapsed_ns: elapsed_ns(plan_start),
    });

    let generation_start = Instant::now();
    let mut outputs = run_campaign_workers(
        &config,
        &runtime,
        &record_plans,
        frame_count,
        worker_count,
        progress_enabled,
    )?;
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
    let all_events = outputs
        .iter()
        .flat_map(|output| output.events.iter().cloned())
        .collect::<Vec<_>>();
    let all_predictions = outputs
        .iter()
        .flat_map(|output| output.predictions.iter().cloned())
        .collect::<Vec<_>>();

    write_root_csvs(
        &config.output_dir,
        &summaries,
        &all_events,
        &all_predictions,
    )?;
    let class_balance = build_class_balance(&config, &summaries);
    write_json_pretty(
        &config.output_dir.join("class_balance.json"),
        &class_balance,
    )?;

    let model_reports =
        write_model_artifacts(&config.output_dir, &summaries, config.trigger_confidence)?;
    let runtime_report = build_runtime_report(
        &config,
        &runtime,
        worker_count,
        progress_enabled,
        &stage_timings,
        overall_start.elapsed(),
    );
    write_json_pretty(
        &config.output_dir.join("runtime_report.json"),
        &runtime_report,
    )?;

    let benchmark_report =
        build_campaign_benchmark_report(&config, &runtime_report, &class_balance, &model_reports);
    write_json_pretty(
        &config.output_dir.join("benchmark_report.json"),
        &benchmark_report,
    )?;

    let dataset_card = campaign_dataset_card(&config, &class_balance)?;
    write_json_pretty(&config.output_dir.join("dataset_card.json"), &dataset_card)?;

    let manifest = CampaignManifest {
        manifest_version: "1".to_string(),
        campaign_request_id: config.campaign.clone(),
        neutral_campaign_id: NEUTRAL_CAMPAIGN_ID.to_string(),
        generated_at: config.generated_at.clone(),
        root_seed: config.seed,
        records: summaries.clone(),
        frame_count,
        frame_rate_hz: config.frame_rate_hz,
        time_window_s: config.time_window_s,
        class_balance: class_balance.clone(),
        runtime_report_path: "runtime_report.json".to_string(),
        benchmark_report_path: "benchmark_report.json".to_string(),
        dataset_card_path: "dataset_card.json".to_string(),
        source_dossier_ref: "object-packs/public-proxy-v1/source_dossier.yaml".to_string(),
        guardrails: campaign_guardrails(),
    };
    write_json_pretty(&config.output_dir.join("campaign_manifest.json"), &manifest)?;
    stage_timings.push(StageTiming {
        stage: "postprocess".to_string(),
        elapsed_ns: elapsed_ns(post_start),
    });

    Ok(CampaignReport {
        output_dir: config.output_dir.clone(),
        campaign_request_id: config.campaign,
        neutral_campaign_id: NEUTRAL_CAMPAIGN_ID.to_string(),
        records: summaries.len(),
        shahed_positive_records: class_balance.shahed_positive_records,
        worker_count,
        runtime,
        progress_enabled,
        campaign_manifest_path: config.output_dir.join("campaign_manifest.json"),
        class_balance_path: config.output_dir.join("class_balance.json"),
        dataset_card_path: config.output_dir.join("dataset_card.json"),
        runtime_report_path: config.output_dir.join("runtime_report.json"),
        benchmark_report_path: config.output_dir.join("benchmark_report.json"),
    })
}

fn validate_campaign_config(config: &CampaignConfig) -> Result<(), DatasetError> {
    if config.campaign != DEFAULT_CAMPAIGN_REQUEST_ID && config.campaign != NEUTRAL_CAMPAIGN_ID {
        return Err(DatasetError::InvalidConfig(format!(
            "unknown campaign {}; expected {}",
            config.campaign, DEFAULT_CAMPAIGN_REQUEST_ID
        )));
    }
    if !(1..=50_000).contains(&config.records) {
        return Err(DatasetError::InvalidConfig(
            "records must be in the range 1..=50000".to_string(),
        ));
    }
    if config.shahed_min > config.records {
        return Err(DatasetError::InvalidConfig(
            "shahed-min cannot exceed records".to_string(),
        ));
    }
    if config.time_window_s <= 0.0 || config.frame_rate_hz <= 0.0 {
        return Err(DatasetError::InvalidConfig(
            "time-window-s and frame-rate-hz must be positive".to_string(),
        ));
    }
    if !(0.05..=0.99).contains(&config.trigger_confidence) {
        return Err(DatasetError::InvalidConfig(
            "trigger-confidence must be in the range 0.05..=0.99".to_string(),
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

fn refuse_campaign_output_path(path: &Path) -> Result<(), DatasetError> {
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

fn effective_worker_count(config: &CampaignConfig, runtime: &RuntimePlan) -> usize {
    let requested = config.workers.unwrap_or(40).min(40);
    requested
        .min(runtime.recommended_worker_budget.max(1))
        .min(config.records.max(1))
        .max(1)
}

fn build_campaign_plan(config: &CampaignConfig) -> Result<Vec<CampaignRecordPlan>, DatasetError> {
    let classes = campaign_classes();
    let positive = classes
        .iter()
        .find(|class| class.bucket == CampaignBucket::PositiveOwaDeltaPusher)
        .expect("positive class exists")
        .clone();
    let target_positive = config
        .shahed_target
        .max(config.shahed_min)
        .min(config.records);
    let remainder = config.records - target_positive;
    let bucket_targets = weighted_remainder_counts(remainder);
    let mut assignments = Vec::with_capacity(config.records);
    assignments.extend(std::iter::repeat(positive).take(target_positive));

    for (bucket, count) in bucket_targets {
        let bucket_classes = classes
            .iter()
            .filter(|class| class.bucket == bucket)
            .cloned()
            .collect::<Vec<_>>();
        if bucket_classes.is_empty() && count > 0 {
            return Err(DatasetError::InvalidConfig(format!(
                "no campaign classes for bucket {bucket:?}"
            )));
        }
        for index in 0..count {
            assignments.push(bucket_classes[index % bucket_classes.len()].clone());
        }
    }

    deterministic_shuffle(&mut assignments, config.seed);
    Ok(assignments
        .into_iter()
        .enumerate()
        .map(|(index, class)| {
            let record_id = format!("record_{:06}", index + 1);
            CampaignRecordPlan {
                index,
                record_id,
                seed: child_seed(config.seed, index as u64),
                class,
            }
        })
        .collect())
}

fn weighted_remainder_counts(remainder: usize) -> Vec<(CampaignBucket, usize)> {
    let weights = [
        (CampaignBucket::SmallUav, 30usize),
        (CampaignBucket::Biological, 20),
        (CampaignBucket::WindborneDebris, 15),
        (CampaignBucket::InfrastructureTerrain, 15),
        (CampaignBucket::GroundMoversMultipath, 10),
        (CampaignBucket::WeatherRfiSensorArtifacts, 10),
    ];
    let mut counts = Vec::with_capacity(weights.len());
    let mut assigned = 0usize;
    for (bucket, weight) in weights {
        let count = remainder * weight / 100;
        assigned += count;
        counts.push((bucket, count));
    }
    let mut cursor = 0usize;
    while assigned < remainder {
        counts[cursor].1 += 1;
        assigned += 1;
        cursor = (cursor + 1) % counts.len();
    }
    counts
}

fn campaign_classes() -> Vec<CampaignClass> {
    vec![
        campaign_class(
            OWA_DELTA_OBJECT_ID,
            "Delta Pusher Fixed-Wing OWA Public Proxy",
            "owa_delta_pusher_public_proxy",
            CampaignBucket::PositiveOwaDeltaPusher,
            "positive_public_proxy",
            true,
            false,
        ),
        campaign_class(
            "low-altitude-fixed-wing-takeoff-v1",
            "Low-Altitude Fixed-Wing UAV Proxy",
            "fixed_wing_uav",
            CampaignBucket::SmallUav,
            "small_fixed_wing_uav",
            false,
            true,
        ),
        campaign_class(
            "commercial-quadrotor-low-altitude-v1",
            "Commercial Quadrotor Low-Altitude Proxy",
            "quadrotor_uav",
            CampaignBucket::SmallUav,
            "quadrotor",
            false,
            true,
        ),
        campaign_class(
            "hexarotor-heavy-lift-low-altitude-v1",
            "Hexarotor Heavy-Lift Low-Altitude Proxy",
            "hexarotor_uav",
            CampaignBucket::SmallUav,
            "hexarotor",
            false,
            true,
        ),
        campaign_class(
            "rc-plane-hobby-glider-v1",
            "RC Plane and Hobby Glider Proxy",
            "rc_fixed_wing_or_hobby_glider",
            CampaignBucket::SmallUav,
            "rc_plane_hobby_glider",
            false,
            true,
        ),
        campaign_class(
            "bird-large-and-flock-v1",
            "Large Bird and Flock Proxy",
            "bird_or_flock",
            CampaignBucket::Biological,
            "bird_or_flock",
            false,
            true,
        ),
        campaign_class(
            "bat-and-insect-cloud-v1",
            "Bat and Insect Cloud Biological Proxy",
            "bat_or_insect_cloud",
            CampaignBucket::Biological,
            "bat_or_insect_cloud",
            false,
            true,
        ),
        campaign_class(
            "balloon-kite-debris-v1",
            "Balloon, Kite, and Windborne Debris Proxy",
            "windborne_slow_object",
            CampaignBucket::WindborneDebris,
            "balloon_kite_windborne_debris",
            false,
            true,
        ),
        campaign_class(
            "wind-turbine-industrial-glint-v1",
            "Wind Turbine and Industrial Glint Proxy",
            "static_or_rotating_infrastructure",
            CampaignBucket::InfrastructureTerrain,
            "infrastructure_glint",
            false,
            true,
        ),
        campaign_class(
            "commercial-aircraft-corridor-clutter-v1",
            "Commercial Aircraft Corridor Clutter Proxy",
            "commercial_aircraft_corridor",
            CampaignBucket::InfrastructureTerrain,
            "commercial_aircraft_corridor",
            false,
            true,
        ),
        campaign_class(
            "ground-vehicle-roadside-v1",
            "Ground Vehicle and Roadside Multipath Proxy",
            "ground_vehicle",
            CampaignBucket::GroundMoversMultipath,
            "ground_movers_multipath",
            false,
            true,
        ),
        campaign_class(
            "weather-terrain-only-scene-v1",
            "Weather and Terrain-Only Scene Proxy",
            "weather_terrain_only",
            CampaignBucket::WeatherRfiSensorArtifacts,
            "weather_rfi_sensor_artifacts",
            false,
            true,
        ),
    ]
}

fn campaign_class(
    id: &str,
    display_name: &str,
    target_family: &str,
    bucket: CampaignBucket,
    hard_negative_family: &str,
    is_shahed_public_proxy: bool,
    is_hard_negative: bool,
) -> CampaignClass {
    CampaignClass {
        id: id.to_string(),
        display_name: display_name.to_string(),
        target_family: target_family.to_string(),
        bucket,
        hard_negative_family: hard_negative_family.to_string(),
        is_shahed_public_proxy,
        is_hard_negative,
    }
}

fn run_campaign_workers(
    config: &CampaignConfig,
    runtime: &RuntimePlan,
    plans: &[CampaignRecordPlan],
    frame_count: usize,
    worker_count: usize,
    progress_enabled: bool,
) -> Result<Vec<CampaignRecordOutput>, DatasetError> {
    let chunk_size = (plans.len() + worker_count - 1) / worker_count;
    let progress_bar = if progress_enabled {
        let bar = ProgressBar::new(plans.len() as u64);
        let style = ProgressStyle::with_template(
            "{spinner:.green} {pos}/{len} records positives={msg} [{elapsed_precise}] {per_sec} ETA {eta_precise}",
        )
        .unwrap_or_else(|_| ProgressStyle::default_bar())
        .progress_chars("=> ");
        bar.set_style(style);
        Some(bar)
    } else {
        None
    };
    let positive_done = Arc::new(AtomicUsize::new(0));

    let result = thread::scope(|scope| -> Result<Vec<CampaignRecordOutput>, DatasetError> {
        let mut handles = Vec::new();
        for chunk in plans.chunks(chunk_size.max(1)) {
            let progress = progress_bar.clone();
            let positive_done = Arc::clone(&positive_done);
            handles.push(scope.spawn(move || {
                run_campaign_chunk(config, runtime, chunk, frame_count, progress, positive_done)
            }));
        }
        let mut outputs = Vec::with_capacity(plans.len());
        for handle in handles {
            let mut chunk = handle.join().map_err(|_| {
                DatasetError::InvalidConfig("campaign worker panicked".to_string())
            })??;
            outputs.append(&mut chunk);
        }
        Ok(outputs)
    });

    if let Some(bar) = progress_bar {
        bar.finish_and_clear();
    }
    result
}

fn run_campaign_chunk(
    config: &CampaignConfig,
    runtime: &RuntimePlan,
    plans: &[CampaignRecordPlan],
    frame_count: usize,
    progress_bar: Option<ProgressBar>,
    positive_done: Arc<AtomicUsize>,
) -> Result<Vec<CampaignRecordOutput>, DatasetError> {
    let mut outputs = Vec::with_capacity(plans.len());
    for plan in plans {
        let output = generate_campaign_record(config, runtime, plan, frame_count)?;
        if plan.class.is_shahed_public_proxy {
            positive_done.fetch_add(1, Ordering::SeqCst);
        }
        if let Some(bar) = progress_bar.as_ref() {
            let positives = positive_done.load(Ordering::SeqCst);
            bar.set_message(format!(
                "{positives} class={} backend={}",
                plan.class.target_family, runtime.selected_backend
            ));
            bar.inc(1);
        }
        outputs.push(output);
    }
    Ok(outputs)
}

fn generate_campaign_record(
    config: &CampaignConfig,
    runtime: &RuntimePlan,
    plan: &CampaignRecordPlan,
    frame_count: usize,
) -> Result<CampaignRecordOutput, DatasetError> {
    let record_dir = config.output_dir.join("records").join(&plan.record_id);
    let products_dir = record_dir.join("products");
    fs::create_dir_all(&products_dir)?;
    let mut rng = SplitMix64::new(plan.seed);
    let envelope = class_envelope(&plan.class, &mut rng);
    let cpi_pulses = rng.range_usize(32, 96);
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
    let profile = TakeoffProfile {
        initial_range_m: envelope.initial_range_m,
        runway_heading_deg: envelope.heading_deg,
        ground_speed_mps: envelope.speed_mps,
        acceleration_mps2: envelope.acceleration_mps2,
        climb_rate_mps: envelope.climb_rate_mps,
        max_altitude_m: envelope.max_altitude_m,
        radial_velocity_bias_mps: envelope.radial_velocity_bias_mps,
        pitch_jitter_deg: envelope.attitude_jitter_deg,
        yaw_jitter_deg: envelope.attitude_jitter_deg,
        propulsor_hz: envelope.propulsor_hz,
        micro_doppler_hz: envelope.micro_doppler_hz,
        rcs_scalar: 10f64.powf(envelope.rcs_dbsm / 20.0).max(0.01),
        blade_count: None,
        blade_length_m: None,
    };
    let mut noise = NoiseProfile::real_world_proxy_v1();
    noise.awgn_sigma = (0.035 + 0.055 * envelope.clutter_profile.false_alarm_pressure()) as f32;
    noise.rfi_probability = envelope.rfi_profile.burst_probability.min(0.12);
    noise.rfi_amplitude = envelope.rfi_profile.burst_amplitude;
    noise.clutter_sigma = (0.025 + 0.12 * envelope.clutter_profile.false_alarm_pressure()) as f32;
    noise.ground_glint_count =
        (2.0f32 + 12.0f32 * envelope.clutter_profile.false_alarm_pressure()).round() as usize;

    let mut episode = synthesize_takeoff_episode(
        sim_config,
        profile,
        noise,
        EpisodeSeed(plan.seed ^ 0x5eed_cafe_136),
    );
    apply_campaign_stressors(&mut episode, &envelope, plan.seed);
    write_episode_tensors(&products_dir, &episode)?;

    let (features, first_detectable_frame) =
        build_frame_features(config, plan, &envelope, frame_count, cpi_pulses);
    let model_run = run_streaming_models(&plan.record_id, &features, config.trigger_confidence)?;
    let first_model_trigger_frame = model_run.events.iter().map(|event| event.frame_index).min();
    let labels = build_frame_labels(
        config,
        plan,
        &features,
        first_detectable_frame,
        first_model_trigger_frame,
    );
    let truth = RecordTruthMetadata {
        record_id: plan.record_id.clone(),
        neutral_object_id: plan.class.id.clone(),
        target_family: plan.class.target_family.clone(),
        is_shahed_public_proxy: plan.class.is_shahed_public_proxy,
        is_hard_negative: plan.class.is_hard_negative,
        hard_negative_family: plan.class.hard_negative_family.clone(),
        source_dossier_ref: "object-packs/public-proxy-v1/source_dossier.yaml".to_string(),
        dimensions_m: envelope.dimensions_m.clone(),
        rcs_dbsm_proxy: envelope.rcs_dbsm,
        cruise_speed_mps: envelope.speed_mps,
        cpi_pulses,
        frame_count,
        time_window_s: config.time_window_s,
        first_detectable_frame,
        confidence_threshold: config.trigger_confidence,
        guardrails: campaign_guardrails(),
    };

    write_csv(&record_dir.join("streaming_features.csv"), &features)?;
    write_csv(&record_dir.join("frame_labels.csv"), &labels)?;
    write_json_pretty(&record_dir.join("truth_metadata.json"), &truth)?;
    write_json_pretty(&record_dir.join("detector_events.json"), &model_run.events)?;
    write_csv(
        &record_dir.join("model_predictions.csv"),
        &model_run.predictions,
    )?;
    write_json_pretty(
        &record_dir.join("runtime_stage.json"),
        &json!({
            "selected_backend": runtime.selected_backend.to_string(),
            "kernel_backend": "cpu-scaffold",
            "gpu_stage_fallback": gpu_stage_fallback_note(runtime),
        }),
    )?;

    Ok(CampaignRecordOutput {
        summary: CampaignRecordSummary {
            record_id: plan.record_id.clone(),
            record_index: plan.index,
            target_family: plan.class.target_family.clone(),
            class_id: plan.class.id.clone(),
            bucket: plan.class.bucket,
            is_shahed_public_proxy: plan.class.is_shahed_public_proxy,
            is_hard_negative: plan.class.is_hard_negative,
            hard_negative_family: plan.class.hard_negative_family.clone(),
            first_detectable_frame,
            first_model_trigger_frame,
            cpi_pulses,
            tensor_dir: relative_record_path(&plan.record_id, "products"),
            frame_labels_path: relative_record_path(&plan.record_id, "frame_labels.csv"),
            truth_metadata_path: relative_record_path(&plan.record_id, "truth_metadata.json"),
            model_predictions_path: relative_record_path(&plan.record_id, "model_predictions.csv"),
            detector_events_path: relative_record_path(&plan.record_id, "detector_events.json"),
            max_confidence_by_model: model_run.max_confidence_by_model,
            first_trigger_by_model: model_run.first_trigger_by_model,
        },
        events: model_run.events,
        predictions: model_run.predictions,
    })
}

fn relative_record_path(record_id: &str, suffix: &str) -> String {
    format!("records/{record_id}/{suffix}")
}

fn apply_campaign_stressors(episode: &mut SyntheticEpisode, envelope: &ClassEnvelope, seed: u64) {
    apply_clutter_to_profile(
        &mut episode.integrated_range_profile,
        envelope.clutter_profile,
        seed ^ 0x1111,
    );
    apply_rfi_to_profile(
        &mut episode.integrated_range_profile,
        envelope.rfi_profile,
        seed ^ 0x2222,
    );
    for (pulse_index, pulse) in episode.iq.iter_mut().enumerate() {
        apply_receiver_impairments(pulse, envelope.receiver_profile, seed ^ 0x3333, pulse_index);
    }
    for (row_index, row) in episode.range_doppler_proxy.iter_mut().enumerate() {
        apply_rfi_to_profile(row, envelope.rfi_profile, seed ^ 0x4444 ^ row_index as u64);
    }
}

#[derive(Debug, Clone)]
struct ClassEnvelope {
    dimensions_m: DimensionsSample,
    rcs_dbsm: f64,
    speed_mps: f64,
    initial_range_m: f64,
    heading_deg: f64,
    acceleration_mps2: f64,
    climb_rate_mps: f64,
    max_altitude_m: f64,
    radial_velocity_bias_mps: f64,
    attitude_jitter_deg: f64,
    propulsor_hz: f64,
    micro_doppler_hz: f64,
    base_snr_db: f32,
    clutter_profile: ClutterProfile,
    rfi_profile: RfiProfile,
    receiver_profile: ReceiverImpairmentProfile,
}

fn class_envelope(class: &CampaignClass, rng: &mut SplitMix64) -> ClassEnvelope {
    let mut clutter = ClutterProfile::moderate_mixed();
    let mut rfi = RfiProfile::contested_low_altitude();
    let mut receiver = ReceiverImpairmentProfile::public_proxy_default();
    let (dimensions, rcs, speed, snr, micro, prop, altitude, range, radial) = match class.bucket {
        CampaignBucket::PositiveOwaDeltaPusher => (
            DimensionsSample {
                length: rng.range_f64(3.3, 3.7),
                wingspan: rng.range_f64(2.3, 2.7),
                height: rng.range_f64(0.35, 0.75),
            },
            rng.range_f64(-16.0, -2.0),
            rng.range_f64(45.0, 60.0),
            rng.range_f32(7.0, 14.0),
            rng.range_f64(28.0, 95.0),
            rng.range_f64(75.0, 145.0),
            rng.range_f64(80.0, 450.0),
            rng.range_f64(4_800.0, 8_500.0),
            rng.range_f64(-55.0, -24.0),
        ),
        CampaignBucket::SmallUav => (
            DimensionsSample {
                length: rng.range_f64(0.35, 2.4),
                wingspan: rng.range_f64(0.35, 4.0),
                height: rng.range_f64(0.08, 0.7),
            },
            rng.range_f64(-32.0, -8.0),
            rng.range_f64(0.0, 32.0),
            rng.range_f32(0.0, 11.0),
            rng.range_f64(40.0, 240.0),
            rng.range_f64(0.0, 250.0),
            rng.range_f64(10.0, 260.0),
            rng.range_f64(700.0, 4_500.0),
            rng.range_f64(-22.0, 22.0),
        ),
        CampaignBucket::Biological => (
            DimensionsSample {
                length: rng.range_f64(0.03, 1.2),
                wingspan: rng.range_f64(0.04, 2.4),
                height: rng.range_f64(0.01, 0.45),
            },
            rng.range_f64(-48.0, -14.0),
            rng.range_f64(1.0, 24.0),
            rng.range_f32(-4.0, 9.0),
            rng.range_f64(3.0, 70.0),
            rng.range_f64(2.0, 45.0),
            rng.range_f64(5.0, 700.0),
            rng.range_f64(300.0, 3_500.0),
            rng.range_f64(-18.0, 18.0),
        ),
        CampaignBucket::WindborneDebris => (
            DimensionsSample {
                length: rng.range_f64(0.2, 5.0),
                wingspan: rng.range_f64(0.2, 8.0),
                height: rng.range_f64(0.2, 5.0),
            },
            rng.range_f64(-34.0, -6.0),
            rng.range_f64(0.0, 14.0),
            rng.range_f32(-5.0, 8.0),
            rng.range_f64(0.0, 8.0),
            rng.range_f64(0.0, 2.0),
            rng.range_f64(5.0, 1_200.0),
            rng.range_f64(300.0, 5_000.0),
            rng.range_f64(-8.0, 8.0),
        ),
        CampaignBucket::InfrastructureTerrain => {
            clutter.turbines = 0.6;
            clutter.buildings = 0.7;
            clutter.power_lines = 0.55;
            (
                DimensionsSample {
                    length: rng.range_f64(5.0, 80.0),
                    wingspan: rng.range_f64(2.0, 80.0),
                    height: rng.range_f64(5.0, 160.0),
                },
                rng.range_f64(0.0, 34.0),
                rng.range_f64(0.0, 260.0),
                rng.range_f32(-4.0, 14.0),
                rng.range_f64(0.0, 45.0),
                rng.range_f64(0.0, 120.0),
                rng.range_f64(20.0, 12_000.0),
                rng.range_f64(800.0, 18_000.0),
                rng.range_f64(-140.0, 140.0),
            )
        }
        CampaignBucket::GroundMoversMultipath => {
            clutter.roads_vehicles = 0.75;
            clutter.urban_multipath = 0.65;
            (
                DimensionsSample {
                    length: rng.range_f64(2.0, 14.0),
                    wingspan: rng.range_f64(1.5, 3.5),
                    height: rng.range_f64(1.0, 4.2),
                },
                rng.range_f64(-5.0, 18.0),
                rng.range_f64(0.0, 32.0),
                rng.range_f32(-2.0, 16.0),
                rng.range_f64(2.0, 40.0),
                rng.range_f64(4.0, 22.0),
                rng.range_f64(0.0, 3.0),
                rng.range_f64(200.0, 4_000.0),
                rng.range_f64(-25.0, 25.0),
            )
        }
        CampaignBucket::WeatherRfiSensorArtifacts => {
            clutter.rain = 0.75;
            clutter.dust_haze = 0.65;
            clutter.terrain_only_scene = 0.8;
            rfi.burst_probability = 0.07;
            rfi.narrowband_cw_power = 0.35;
            rfi.cochannel_emitters = 8;
            receiver.dropped_pulse_probability = 0.03;
            receiver.clipping_level = 1.2;
            (
                DimensionsSample {
                    length: 0.0,
                    wingspan: 0.0,
                    height: 0.0,
                },
                rng.range_f64(-45.0, -18.0),
                rng.range_f64(0.0, 12.0),
                rng.range_f32(-8.0, 8.0),
                rng.range_f64(0.0, 6.0),
                0.0,
                rng.range_f64(0.0, 2_500.0),
                rng.range_f64(100.0, 8_000.0),
                rng.range_f64(-6.0, 6.0),
            )
        }
    };

    ClassEnvelope {
        dimensions_m: dimensions,
        rcs_dbsm: rcs,
        speed_mps: speed,
        initial_range_m: range,
        heading_deg: rng.range_f64(-18.0, 18.0),
        acceleration_mps2: rng.range_f64(0.0, 1.2),
        climb_rate_mps: rng.range_f64(0.0, 4.0),
        max_altitude_m: altitude.max(1.0),
        radial_velocity_bias_mps: radial,
        attitude_jitter_deg: rng.range_f64(0.2, 4.0),
        propulsor_hz: prop,
        micro_doppler_hz: micro,
        base_snr_db: snr,
        clutter_profile: clutter,
        rfi_profile: rfi,
        receiver_profile: receiver,
    }
}

fn build_frame_features(
    config: &CampaignConfig,
    plan: &CampaignRecordPlan,
    envelope: &ClassEnvelope,
    frame_count: usize,
    cpi_pulses: usize,
) -> (Vec<FrameFeature>, Option<usize>) {
    let mut features = Vec::with_capacity(frame_count);
    let mut first_detectable = None;
    let mut persistence_frames = 0usize;
    let mut prev_snr = envelope.base_snr_db;

    for frame_index in 0..frame_count {
        let time_s = frame_index as f64 / config.frame_rate_hz;
        let progress = (time_s / config.time_window_s).clamp(0.0, 1.0);
        let mut frame_rng =
            SplitMix64::new(plan.seed ^ (frame_index as u64).wrapping_mul(0x1360_0d5));
        let clutter = sample_clutter_frame(envelope.clutter_profile, plan.seed, frame_index);
        let rfi = sample_rfi_frame(envelope.rfi_profile, plan.seed, frame_index, 128);
        let receiver =
            sample_receiver_impairments(envelope.receiver_profile, plan.seed, frame_index);
        let positive_rise = if plan.class.is_shahed_public_proxy {
            8.0 * (1.0 - (-time_s / 16.0).exp()) as f32
        } else {
            0.0
        };
        let family_noise = frame_rng.range_f32(-1.5, 1.5);
        let snr_db = (envelope.base_snr_db + positive_rise + family_noise
            - 4.0 * rfi.pressure
            - 2.0 * clutter.false_alarm_pressure)
            .clamp(-12.0, 30.0);
        if snr_db >= 7.0 && !receiver.dropped_pulse {
            persistence_frames += 1;
        } else {
            persistence_frames = persistence_frames.saturating_sub(1);
        }

        let first_gate =
            snr_db >= 7.0 && persistence_frames >= 2 && clutter.false_alarm_pressure < 0.85;
        if first_detectable.is_none() && first_gate {
            first_detectable = Some(frame_index);
        }

        let range_direction = if plan.class.is_shahed_public_proxy {
            -0.78
        } else {
            frame_rng.range_f64(-0.35, 0.35)
        };
        let range_m =
            (envelope.initial_range_m + range_direction * envelope.speed_mps * time_s).max(50.0);
        let radial_velocity = envelope.radial_velocity_bias_mps
            + frame_rng.range_f64(-2.0, 2.0)
            + if plan.class.bucket == CampaignBucket::InfrastructureTerrain {
                18.0 * (progress * std::f64::consts::PI).sin()
            } else {
                0.0
            };
        let altitude = if plan.class.is_shahed_public_proxy {
            (30.0 + 160.0 * progress + 20.0 * (progress * std::f64::consts::PI).sin())
                .min(envelope.max_altitude_m)
        } else {
            envelope.max_altitude_m * (0.15 + 0.8 * frame_rng.unit_f64())
        };
        let doppler_spread = (clutter.doppler_spread_hz
            + envelope.micro_doppler_hz as f32 * 0.12
            + frame_rng.range_f32(0.0, 8.0))
        .clamp(0.0, 320.0);
        let blob_area = (2.0
            + snr_db.max(0.0) * 0.35
            + clutter.false_alarm_pressure * 5.0
            + frame_rng.range_f32(0.0, 3.0))
        .clamp(0.0, 80.0);
        let micro_mod = (envelope.micro_doppler_hz as f32
            * (0.75 + 0.25 * (2.0 * std::f64::consts::PI * progress).sin() as f32)
            + frame_rng.range_f32(-4.0, 4.0))
        .max(0.0);

        features.push(FrameFeature {
            frame_index,
            time_s,
            cpi_pulses,
            range_m,
            range_rate_mps: radial_velocity,
            radial_velocity_mps: radial_velocity,
            altitude_m: altitude,
            snr_db,
            snr_trend_db: snr_db - prev_snr,
            doppler_spread_hz: doppler_spread,
            blob_area_bins: blob_area,
            micro_doppler_modulation: micro_mod,
            track_persistence_s: persistence_frames as f32 / config.frame_rate_hz as f32,
            clutter_pressure: clutter.false_alarm_pressure,
            rfi_pressure: rfi.pressure,
            receiver_dropout: receiver.dropped_pulse,
            phase_noise_rad: receiver.phase_offset_rad,
            amplitude_scintillation: envelope.receiver_profile.amplitude_scintillation_sigma,
        });
        prev_snr = snr_db;
    }

    (features, first_detectable)
}

fn build_frame_labels(
    config: &CampaignConfig,
    plan: &CampaignRecordPlan,
    features: &[FrameFeature],
    first_detectable_frame: Option<usize>,
    first_model_trigger_frame: Option<usize>,
) -> Vec<FrameLabel> {
    features
        .iter()
        .map(|feature| FrameLabel {
            frame_index: feature.frame_index,
            time_s: feature.time_s,
            target_family: plan.class.target_family.clone(),
            is_shahed_public_proxy: plan.class.is_shahed_public_proxy,
            is_hard_negative: plan.class.is_hard_negative,
            hard_negative_family: plan.class.hard_negative_family.clone(),
            scenario_phase: scenario_phase(plan.class.is_shahed_public_proxy, feature.time_s),
            first_detectable_frame,
            first_model_trigger_frame,
            confidence_threshold: config.trigger_confidence,
        })
        .collect()
}

fn scenario_phase(is_positive: bool, time_s: f64) -> String {
    if !is_positive {
        return "confuser_or_artifact_motion".to_string();
    }
    if time_s < 8.0 {
        "rail_or_booster_launch_proxy".to_string()
    } else if time_s < 18.0 {
        "transition_to_pusher_prop".to_string()
    } else if time_s < 72.0 {
        "low_altitude_cruise".to_string()
    } else {
        "shallow_climb_turn".to_string()
    }
}

#[derive(Debug, Clone)]
struct ModelRun {
    events: Vec<FirstTriggerEvent>,
    predictions: Vec<ModelPredictionRow>,
    max_confidence_by_model: BTreeMap<String, f32>,
    first_trigger_by_model: BTreeMap<String, Option<usize>>,
}

fn run_streaming_models(
    record_id: &str,
    features: &[FrameFeature],
    threshold: f32,
) -> Result<ModelRun, DatasetError> {
    let mut detectors: Vec<Box<dyn StreamingDetector>> = vec![
        Box::new(CfarTrackerBaseline::new(threshold)),
        Box::new(FeatureTreeClassifier::new(threshold)),
        Box::new(TemporalTinyModel::new(threshold)),
    ];
    let mut events = Vec::new();
    let mut predictions = Vec::with_capacity(features.len() * detectors.len());
    let mut max_confidence_by_model = BTreeMap::<String, f32>::new();
    let mut first_trigger_by_model = BTreeMap::<String, Option<usize>>::new();

    for detector in detectors.iter() {
        first_trigger_by_model.insert(detector.model_id().to_string(), None);
        max_confidence_by_model.insert(detector.model_id().to_string(), 0.0);
    }

    for frame in features {
        for detector in detectors.iter_mut() {
            let state = detector.update(frame);
            max_confidence_by_model
                .entry(state.model_id.clone())
                .and_modify(|value| *value = (*value).max(state.confidence))
                .or_insert(state.confidence);
            let triggered_on_this_frame = state.first_trigger_event.is_some();
            if let Some(event) = state.first_trigger_event.clone() {
                first_trigger_by_model.insert(state.model_id.clone(), Some(event.frame_index));
                events.push(event);
            }
            predictions.push(ModelPredictionRow {
                record_id: record_id.to_string(),
                model_id: state.model_id,
                frame_index: frame.frame_index,
                time_s: frame.time_s,
                confidence: state.confidence,
                class_probabilities_json: serde_json::to_string(&state.class_probabilities)?,
                triggered_on_this_frame,
            });
        }
    }

    Ok(ModelRun {
        events,
        predictions,
        max_confidence_by_model,
        first_trigger_by_model,
    })
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

fn write_root_csvs(
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
                .unwrap_or_else(|_| "{}".to_string()),
            first_trigger_by_model_json: serde_json::to_string(&value.first_trigger_by_model)
                .unwrap_or_else(|_| "{}".to_string()),
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

fn build_class_balance(
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

fn write_model_artifacts(
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
    progress_requested: bool,
    progress_enabled: bool,
    records: usize,
    stage_timings: Vec<StageTiming>,
    total_elapsed_ns: u64,
    throughput_records_per_sec: f64,
}

fn build_runtime_report(
    config: &CampaignConfig,
    runtime: &RuntimePlan,
    worker_count: usize,
    progress_enabled: bool,
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
        progress_requested: config.progress,
        progress_enabled,
        records: config.records,
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
            "GPU probe selected a GPU-capable runtime, but the campaign radar kernel stage is currently CPU-backed; generated tensors record this fallback.".to_string(),
        ),
        echoforge_radar::RuntimeBackend::Cpu => runtime.fallback_reason.clone(),
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CampaignBenchmarkReport {
    campaign_request_id: String,
    neutral_campaign_id: String,
    class_balance: ClassBalanceReport,
    runtime: RuntimeReport,
    model_reports: Vec<ModelEvaluationReport>,
    limitations: Vec<String>,
}

fn build_campaign_benchmark_report(
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

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CampaignManifest {
    manifest_version: String,
    campaign_request_id: String,
    neutral_campaign_id: String,
    generated_at: String,
    root_seed: u64,
    records: Vec<CampaignRecordSummary>,
    frame_count: usize,
    frame_rate_hz: f64,
    time_window_s: f64,
    class_balance: ClassBalanceReport,
    runtime_report_path: String,
    benchmark_report_path: String,
    dataset_card_path: String,
    source_dossier_ref: String,
    guardrails: Vec<String>,
}

fn campaign_dataset_card(
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
                "object-packs/public-proxy-v1/source_dossier.yaml".to_string(),
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

fn campaign_guardrails() -> Vec<String> {
    vec![
        "public proxy".to_string(),
        "no exact measured truth".to_string(),
        "uncertainty-bounded".to_string(),
        "No payload effects, terminal behavior, evasion tactics, or operational optimization are included.".to_string(),
        "Hard negatives are robustness and false-alarm stressors only.".to_string(),
    ]
}

#[cfg(test)]
#[derive(Debug, Clone, Deserialize)]
struct SourceDossier {
    public_proxy_id: String,
    aliases: Vec<SourceAlias>,
    proxy_envelope: SourceProxyEnvelope,
}

#[cfg(test)]
#[derive(Debug, Clone, Deserialize)]
struct SourceAlias {
    alias: String,
    policy: String,
}

#[cfg(test)]
#[derive(Debug, Clone, Deserialize)]
struct SourceProxyEnvelope {
    dimensions_m: SourceDimensions,
    rcs_dbsm_proxy: [f64; 2],
    cruise_speed_mps: [f64; 2],
}

#[cfg(test)]
#[derive(Debug, Clone, Deserialize)]
struct SourceDimensions {
    length: [f64; 2],
    wingspan: [f64; 2],
    height: [f64; 2],
}

#[cfg(test)]
fn parse_source_dossier(input: &str) -> Result<SourceDossier, DatasetError> {
    serde_yaml::from_str(input).map_err(DatasetError::Yaml)
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
    fn source_dossier_priors_parse_and_aliases_are_dossier_only() {
        let dossier = parse_source_dossier(include_str!(
            "../../../object-packs/public-proxy-v1/source_dossier.yaml"
        ))
        .expect("source dossier parses");
        assert_eq!(dossier.public_proxy_id, OWA_DELTA_OBJECT_ID);
        assert_eq!(dossier.proxy_envelope.dimensions_m.length, [3.3, 3.7]);
        assert_eq!(dossier.proxy_envelope.dimensions_m.wingspan, [2.3, 2.7]);
        assert_eq!(dossier.proxy_envelope.dimensions_m.height, [0.35, 0.75]);
        assert_eq!(dossier.proxy_envelope.rcs_dbsm_proxy, [-16.0, -2.0]);
        assert_eq!(dossier.proxy_envelope.cruise_speed_mps, [45.0, 60.0]);
        assert!(dossier
            .aliases
            .iter()
            .any(|alias| alias.alias == "Shahed-136"));
        assert!(dossier
            .aliases
            .iter()
            .all(|alias| alias.policy == "source_dossier_only"));
    }

    #[test]
    fn campaign_object_ids_are_neutral() {
        for class in campaign_classes() {
            let id = class.id.to_ascii_lowercase();
            assert!(!id.contains("shahed"), "{id}");
            assert!(!id.contains("iranian"), "{id}");
            assert!(!id.contains("geran"), "{id}");
        }
    }

    #[test]
    fn default_balance_has_exact_records_and_minimum_positive_margin() {
        let config = CampaignConfig::shahed_public_proxy_default();
        let plan = build_campaign_plan(&config).expect("plan builds");
        assert_eq!(plan.len(), 1_000);
        let positives = plan
            .iter()
            .filter(|record| record.class.is_shahed_public_proxy)
            .count();
        assert_eq!(positives, 80);
        assert!(positives >= 50);
    }

    #[test]
    fn worker_cap_never_exceeds_forty_or_runtime_budget() {
        let mut config = CampaignConfig::shahed_public_proxy_default();
        config.records = 100;
        config.workers = Some(128);
        let runtime =
            RuntimePlan::from_signals(BackendMode::Cpu, BackendSignals::new(128, false, false))
                .expect("runtime");
        let workers = effective_worker_count(&config, &runtime);
        assert!(workers <= 40);
        assert!(workers <= runtime.recommended_worker_budget);
    }

    #[test]
    fn streaming_trigger_requires_two_consecutive_frames() {
        let mut detector = FeatureTreeClassifier::new(0.8);
        let mut frame = FrameFeature {
            frame_index: 0,
            time_s: 0.0,
            cpi_pulses: 32,
            range_m: 4000.0,
            range_rate_mps: -52.0,
            radial_velocity_mps: -52.0,
            altitude_m: 120.0,
            snr_db: 24.0,
            snr_trend_db: 1.0,
            doppler_spread_hz: 25.0,
            blob_area_bins: 9.0,
            micro_doppler_modulation: 70.0,
            track_persistence_s: 4.0,
            clutter_pressure: 0.02,
            rfi_pressure: 0.01,
            receiver_dropout: false,
            phase_noise_rad: 0.0,
            amplitude_scintillation: 0.05,
        };
        let first = detector.update(&frame);
        assert!(first.first_trigger_event.is_none());
        frame.frame_index = 1;
        frame.time_s = 0.5;
        let second = detector.update(&frame);
        assert!(second.first_trigger_event.is_some());
    }

    #[test]
    fn negative_low_confidence_never_triggers() {
        let mut detector = TemporalTinyModel::new(0.8);
        for index in 0..20 {
            let state = detector.update(&FrameFeature {
                frame_index: index,
                time_s: index as f64 * 0.5,
                cpi_pulses: 32,
                range_m: 1200.0,
                range_rate_mps: 2.0,
                radial_velocity_mps: 2.0,
                altitude_m: 10.0,
                snr_db: -4.0,
                snr_trend_db: 0.0,
                doppler_spread_hz: 2.0,
                blob_area_bins: 1.0,
                micro_doppler_modulation: 2.0,
                track_persistence_s: 0.0,
                clutter_pressure: 0.2,
                rfi_pressure: 0.1,
                receiver_dropout: false,
                phase_noise_rad: 0.0,
                amplitude_scintillation: 0.05,
            });
            assert!(state.first_trigger_event.is_none());
        }
    }

    #[test]
    fn small_campaign_fixture_writes_manifest_and_disables_progress_in_tests() {
        let temp = tempfile::tempdir().expect("tempdir");
        let mut config = CampaignConfig::shahed_public_proxy_default();
        config.records = 12;
        config.shahed_min = 3;
        config.shahed_target = 4;
        config.workers = Some(4);
        config.progress = false;
        config.output_dir = temp.path().join("campaign");
        let report = run_monte_carlo_campaign(config).expect("campaign run");
        assert_eq!(report.records, 12);
        assert_eq!(report.shahed_positive_records, 4);
        assert!(report.worker_count <= 4);
        assert!(!report.progress_enabled);
        assert!(report.campaign_manifest_path.exists());
        assert!(report.class_balance_path.exists());
        assert!(report.dataset_card_path.exists());
        assert!(report.runtime_report_path.exists());
        assert!(report.benchmark_report_path.exists());
        assert!(temp
            .path()
            .join("campaign/records/record_000001/frame_labels.csv")
            .exists());
        assert!(temp
            .path()
            .join("campaign/models/cfar_tracker_baseline/model_metadata.json")
            .exists());
        assert!(temp
            .path()
            .join("campaign/qa/model_eval_temporal_tiny_model.json")
            .exists());
    }
}
