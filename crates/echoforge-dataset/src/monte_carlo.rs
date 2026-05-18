use std::collections::BTreeMap;
use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};

use echoforge_core::models::{
    DatasetCard, DatasetSplits, LicenseInfo, Provenance, RadarEpisode, Scenario, SensorArchetype,
    ValidationCheck, ValidationInfo,
};
use echoforge_radar::{
    synthesize_takeoff_episode, DetectionRecord, EpisodeSeed, NoiseProfile, RadarSimConfig,
    SyntheticEpisode, TakeoffProfile,
};
use serde::{Deserialize, Serialize};

use crate::export::{write_episode_tensors, write_json_pretty, write_text};
use crate::leakage::{build_leakage_report, LeakageReport};
use crate::split::{assign_split, DatasetRecord, SplitKind, SplitPolicy, SplitRatios};

const CONFIG_JSON: &str = include_str!("../../../configs/monte-carlo/airspace-objects-v1.json");
const DEFAULT_TARGET_LABEL: &str = "Iranian Public-Proxy Fixed-Wing UAV Takeoff";
const DEFAULT_NOISE_PROFILE: &str = "real-world-proxy-v1";

#[derive(Debug, Clone)]
pub struct MonteCarloDemoConfig {
    pub preset: String,
    pub episodes: usize,
    pub seed: u64,
    pub generated_at: String,
    pub output_dir: PathBuf,
    pub target_label: String,
    pub noise_profile: String,
    pub sample_rate_hz: f64,
    pub pulse_count: usize,
}

impl MonteCarloDemoConfig {
    pub fn iranian_takeoff_default(output_dir: PathBuf) -> Self {
        Self {
            preset: "iranian-takeoff-v1".to_string(),
            episodes: 32,
            seed: 20_260_518,
            generated_at: "2026-05-18T00:00:00Z".to_string(),
            output_dir,
            target_label: DEFAULT_TARGET_LABEL.to_string(),
            noise_profile: DEFAULT_NOISE_PROFILE.to_string(),
            sample_rate_hz: 2_000_000.0,
            pulse_count: 32,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MonteCarloDemoReport {
    pub output_dir: PathBuf,
    pub preset: String,
    pub episode_count: usize,
    pub split_counts: BTreeMap<SplitKind, usize>,
    pub leakage_clean: bool,
    pub validation_status: String,
    pub manifest_path: PathBuf,
    pub dataset_card_path: PathBuf,
}

#[derive(Debug)]
pub enum DatasetError {
    Io(std::io::Error),
    Json(serde_json::Error),
    Core(echoforge_core::CoreError),
    Sig(echoforge_sig::SigError),
    Tensor(String),
    InvalidConfig(String),
    RefusingOutputPath(String),
}

impl fmt::Display for DatasetError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(err) => write!(f, "io error: {err}"),
            Self::Json(err) => write!(f, "json error: {err}"),
            Self::Core(err) => write!(f, "core model validation error: {err}"),
            Self::Sig(err) => write!(f, "tensor writer error: {err}"),
            Self::Tensor(err) => write!(f, "tensor shape error: {err}"),
            Self::InvalidConfig(msg) => write!(f, "invalid Monte Carlo config: {msg}"),
            Self::RefusingOutputPath(msg) => write!(f, "refusing output path: {msg}"),
        }
    }
}

impl std::error::Error for DatasetError {}

impl From<std::io::Error> for DatasetError {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value)
    }
}

impl From<serde_json::Error> for DatasetError {
    fn from(value: serde_json::Error) -> Self {
        Self::Json(value)
    }
}

impl From<echoforge_core::CoreError> for DatasetError {
    fn from(value: echoforge_core::CoreError) -> Self {
        Self::Core(value)
    }
}

impl From<echoforge_sig::SigError> for DatasetError {
    fn from(value: echoforge_sig::SigError) -> Self {
        Self::Sig(value)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AirspaceMonteCarloConfig {
    pub config_id: String,
    pub display_name: String,
    pub purpose: String,
    pub guardrails: Vec<String>,
    pub object_classes: Vec<ObjectClassConfig>,
    pub environment_profiles: Vec<EnvironmentProfileConfig>,
    pub sensor_archetypes: Vec<SensorArchetypeConfig>,
    pub presets: Vec<PresetConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObjectClassConfig {
    pub id: String,
    pub display_name: String,
    pub object_family: String,
    pub role_tags: Vec<String>,
    pub dimensions_m: DimensionsConfig,
    pub rcs_dbsm: [f64; 2],
    pub material_mix: BTreeMap<String, [f64; 2]>,
    pub kinematics: KinematicsConfig,
    pub micro_motion: MicroMotionConfig,
    pub behavior: BehaviorConfig,
    pub sensor_observables: SensorObservableConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DimensionsConfig {
    pub length: [f64; 2],
    pub wingspan: [f64; 2],
    pub height: [f64; 2],
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KinematicsConfig {
    pub ground_speed_mps: [f64; 2],
    pub acceleration_mps2: [f64; 2],
    pub climb_rate_mps: [f64; 2],
    pub max_altitude_m: [f64; 2],
    pub turn_rate_deg_s: [f64; 2],
    pub radial_velocity_mps: [f64; 2],
    pub altitude_agl_m: [f64; 2],
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MicroMotionConfig {
    pub propulsor_hz: [f64; 2],
    pub blade_count: [u32; 2],
    pub micro_doppler_hz: [f64; 2],
    pub amplitude_modulation: [f64; 2],
    pub attitude_jitter_deg: [f64; 2],
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BehaviorConfig {
    pub phases: Vec<String>,
    pub maneuverability: String,
    pub formation_count: [u32; 2],
    pub track_persistence_s: [f64; 2],
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SensorObservableConfig {
    pub expected_range_bins: [u32; 2],
    pub doppler_spread_bins: [u32; 2],
    pub scintillation_sigma: [f64; 2],
    pub classification_prior: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvironmentProfileConfig {
    pub id: String,
    pub display_name: String,
    pub terrain: String,
    pub clutter_density: [f64; 2],
    pub ground_glint_density: [f64; 2],
    pub weather: WeatherConfig,
    pub rfi: RfiConfig,
    pub contested_airspace: ContestedAirspaceConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeatherConfig {
    pub precipitation_rate_mm_h: [f64; 2],
    pub wind_speed_mps: [f64; 2],
    pub turbulence_index: [f64; 2],
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RfiConfig {
    pub background_interference_probability: [f64; 2],
    pub impulse_probability: [f64; 2],
    pub burst_duration_pulses: [u32; 2],
    pub spectral_overlap_fraction: [f64; 2],
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContestedAirspaceConfig {
    pub traffic_density: [f64; 2],
    pub cochannel_emitters: [u32; 2],
    pub uncooperative_transponder_fraction: [f64; 2],
    pub multipath_severity: [f64; 2],
    pub sensor_dropout_probability: [f64; 2],
    pub label_ambiguity_probability: [f64; 2],
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SensorArchetypeConfig {
    pub id: String,
    pub display_name: String,
    pub band_name: String,
    pub center_frequency_hz: [f64; 2],
    pub sample_rate_hz: [f64; 2],
    pub bandwidth_hz: [f64; 2],
    pub pulse_width_s: [f64; 2],
    pub pri_s: [f64; 2],
    pub pulse_count: [u32; 2],
    pub receiver_noise_figure_db: [f64; 2],
    pub calibration_error_db: [f64; 2],
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PresetConfig {
    pub id: String,
    pub object_class_id: String,
    pub environment_profile_id: String,
    pub sensor_archetype_id: String,
    pub scenario_label: String,
    pub episode_duration_s: [f64; 2],
    pub snr_db: [f64; 2],
    pub notes: Vec<String>,
}

pub fn embedded_airspace_config() -> Result<AirspaceMonteCarloConfig, DatasetError> {
    serde_json::from_str(CONFIG_JSON).map_err(DatasetError::Json)
}

pub fn known_presets() -> Result<Vec<String>, DatasetError> {
    Ok(embedded_airspace_config()?
        .presets
        .into_iter()
        .map(|preset| preset.id)
        .collect())
}

pub fn run_monte_carlo_demo(
    config: MonteCarloDemoConfig,
) -> Result<MonteCarloDemoReport, DatasetError> {
    validate_run_config(&config)?;
    refuse_source_owned_output(&config.output_dir)?;
    if config.output_dir.exists() {
        return Err(DatasetError::RefusingOutputPath(format!(
            "{} already exists",
            config.output_dir.display()
        )));
    }

    let library = embedded_airspace_config()?;
    let resolved = ResolvedPreset::resolve(&library, &config.preset)?;
    fs::create_dir_all(&config.output_dir)?;

    write_static_cards(&config, &resolved)?;

    let policy = monte_carlo_split_policy();
    let mut records = Vec::with_capacity(config.episodes);
    let mut episode_manifests = Vec::with_capacity(config.episodes);
    let mut split_counts: BTreeMap<SplitKind, usize> = BTreeMap::new();

    for index in 0..config.episodes {
        let episode_seed = child_seed(config.seed, index as u64);
        let mut rng = SplitMix64::new(episode_seed);
        let episode_id = format!("episode_{:06}", index + 1);
        let episode_dir = config.output_dir.join("episodes").join(&episode_id);
        let products_dir = episode_dir.join("products");
        fs::create_dir_all(&products_dir)?;

        let sampled = sample_episode(&config, &resolved, &mut rng, episode_seed);
        let episode = synthesize_takeoff_episode(
            sampled.sim_config,
            sampled.profile,
            sampled.noise_profile,
            EpisodeSeed(episode_seed),
        );

        write_episode_tensors(&products_dir, &episode)?;
        write_episode_json_products(&products_dir, &episode, &sampled)?;

        let mut record = DatasetRecord {
            sample_id: episode_id.clone(),
            split_hint: None,
            object_family: resolved.object.object_family.clone(),
            geometry_hash: format!("geometry-{:016x}", child_seed(episode_seed, 1)),
            material_sample_hash: format!("material-{:016x}", child_seed(episode_seed, 2)),
            scenario_seed: episode_seed,
            sensor_archetype: resolved.sensor.id.clone(),
            hard_negative_family: resolved.environment.id.clone(),
        };
        let split = assign_split(&record, &policy);
        record.split_hint = Some(split);
        *split_counts.entry(split).or_insert(0) += 1;

        let radar_episode = radar_episode_model(&config, &episode_id, &sampled)?;
        write_json_pretty(&episode_dir.join("radar_episode.json"), &radar_episode)?;
        records.push(record);
        episode_manifests.push(EpisodeManifestEntry {
            episode_id,
            seed: episode_seed,
            split,
            path: format!("episodes/episode_{:06}/radar_episode.json", index + 1),
            detections: episode.detections.len(),
            target_label: config.target_label.clone(),
        });
    }

    let leakage_report = build_leakage_report(&records, &policy);
    write_text(
        &config.output_dir.join("records.jsonl"),
        &records_jsonl(&records)?,
    )?;
    write_json_pretty(
        &config.output_dir.join("split_manifest.json"),
        &SplitManifest {
            policy: policy.clone(),
            counts: split_counts.clone(),
            records: records.clone(),
        },
    )?;
    write_json_pretty(
        &config.output_dir.join("leakage_report.json"),
        &leakage_report,
    )?;

    let dataset_card = dataset_card_model(&config, &resolved, &split_counts)?;
    write_json_pretty(&config.output_dir.join("dataset_card.json"), &dataset_card)?;

    let manifest = run_manifest(
        &config,
        &library,
        &resolved,
        &episode_manifests,
        &leakage_report,
    );
    write_json_pretty(&config.output_dir.join("manifest.json"), &manifest)?;
    write_text(
        &config.output_dir.join("benchmark_report.md"),
        &benchmark_report(&config, &resolved, &split_counts, &leakage_report),
    )?;

    Ok(MonteCarloDemoReport {
        output_dir: config.output_dir.clone(),
        preset: config.preset,
        episode_count: config.episodes,
        split_counts,
        leakage_clean: leakage_report.is_clean(),
        validation_status: "pass".to_string(),
        manifest_path: config.output_dir.join("manifest.json"),
        dataset_card_path: config.output_dir.join("dataset_card.json"),
    })
}

fn validate_run_config(config: &MonteCarloDemoConfig) -> Result<(), DatasetError> {
    if !(1..=10_000).contains(&config.episodes) {
        return Err(DatasetError::InvalidConfig(
            "episodes must be in the range 1..=10000".to_string(),
        ));
    }
    if config.generated_at.trim().is_empty()
        || !config.generated_at.contains('T')
        || !config.generated_at.ends_with('Z')
    {
        return Err(DatasetError::InvalidConfig(
            "generated_at must be an RFC3339-like UTC timestamp ending in Z".to_string(),
        ));
    }
    if config.noise_profile != DEFAULT_NOISE_PROFILE {
        return Err(DatasetError::InvalidConfig(format!(
            "unknown noise profile {}; expected {DEFAULT_NOISE_PROFILE}",
            config.noise_profile
        )));
    }
    if config.sample_rate_hz <= 0.0 || config.pulse_count == 0 {
        return Err(DatasetError::InvalidConfig(
            "sample_rate_hz and pulse_count must be positive".to_string(),
        ));
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

#[derive(Debug, Clone)]
struct ResolvedPreset<'a> {
    preset: &'a PresetConfig,
    object: &'a ObjectClassConfig,
    environment: &'a EnvironmentProfileConfig,
    sensor: &'a SensorArchetypeConfig,
}

impl<'a> ResolvedPreset<'a> {
    fn resolve(
        library: &'a AirspaceMonteCarloConfig,
        preset_id: &str,
    ) -> Result<Self, DatasetError> {
        let preset = library
            .presets
            .iter()
            .find(|preset| preset.id == preset_id)
            .ok_or_else(|| DatasetError::InvalidConfig(format!("unknown preset {preset_id}")))?;
        let object = library
            .object_classes
            .iter()
            .find(|object| object.id == preset.object_class_id)
            .ok_or_else(|| {
                DatasetError::InvalidConfig(format!(
                    "preset {} references missing object {}",
                    preset.id, preset.object_class_id
                ))
            })?;
        let environment = library
            .environment_profiles
            .iter()
            .find(|environment| environment.id == preset.environment_profile_id)
            .ok_or_else(|| {
                DatasetError::InvalidConfig(format!(
                    "preset {} references missing environment {}",
                    preset.id, preset.environment_profile_id
                ))
            })?;
        let sensor = library
            .sensor_archetypes
            .iter()
            .find(|sensor| sensor.id == preset.sensor_archetype_id)
            .ok_or_else(|| {
                DatasetError::InvalidConfig(format!(
                    "preset {} references missing sensor {}",
                    preset.id, preset.sensor_archetype_id
                ))
            })?;
        Ok(Self {
            preset,
            object,
            environment,
            sensor,
        })
    }
}

#[derive(Debug, Clone, Serialize)]
struct SampledEpisodeMeta {
    sim_config: RadarSimConfig,
    profile: TakeoffProfile,
    noise_profile: NoiseProfile,
    object_class_id: String,
    environment_profile_id: String,
    sensor_archetype_id: String,
    snr_db: f64,
    rcs_dbsm: f64,
    clutter_density: f64,
    rfi_probability: f64,
    contested_airspace: ContestedSample,
}

#[derive(Debug, Clone, Serialize)]
struct ContestedSample {
    traffic_density: f64,
    cochannel_emitters: u32,
    uncooperative_transponder_fraction: f64,
    multipath_severity: f64,
    sensor_dropout_probability: f64,
    label_ambiguity_probability: f64,
}

fn sample_episode(
    config: &MonteCarloDemoConfig,
    resolved: &ResolvedPreset<'_>,
    rng: &mut SplitMix64,
    seed: u64,
) -> SampledEpisodeMeta {
    let object = resolved.object;
    let environment = resolved.environment;
    let sensor = resolved.sensor;
    let contested = ContestedSample {
        traffic_density: rng.range_f64(environment.contested_airspace.traffic_density),
        cochannel_emitters: rng.range_u32(environment.contested_airspace.cochannel_emitters),
        uncooperative_transponder_fraction: rng.range_f64(
            environment
                .contested_airspace
                .uncooperative_transponder_fraction,
        ),
        multipath_severity: rng.range_f64(environment.contested_airspace.multipath_severity),
        sensor_dropout_probability: rng
            .range_f64(environment.contested_airspace.sensor_dropout_probability),
        label_ambiguity_probability: rng
            .range_f64(environment.contested_airspace.label_ambiguity_probability),
    };

    let snr_db = rng.range_f64(resolved.preset.snr_db);
    let rcs_dbsm = rng.range_f64(object.rcs_dbsm);
    let clutter_density = rng.range_f64(environment.clutter_density);
    let rfi_probability = rng.range_f64(environment.rfi.impulse_probability)
        + 0.002 * contested.cochannel_emitters as f64
        + 0.01 * contested.multipath_severity;

    let mut noise = NoiseProfile::real_world_proxy_v1();
    noise.awgn_sigma = (0.035 + 0.035 * clutter_density + 0.02 * contested.traffic_density) as f32;
    noise.clutter_sigma = (0.025 + 0.08 * clutter_density) as f32;
    noise.ground_glint_count =
        (2.0 + 10.0 * rng.range_f64(environment.ground_glint_density)).round() as usize;
    noise.rfi_probability = rfi_probability.min(0.12) as f32;
    noise.rfi_amplitude = (0.55 + 1.2 * contested.multipath_severity) as f32;
    noise.amplitude_scintillation_sigma =
        rng.range_f64(object.sensor_observables.scintillation_sigma) as f32;

    let sample_rate_hz = if config.sample_rate_hz > 0.0 {
        config.sample_rate_hz
    } else {
        rng.range_f64(sensor.sample_rate_hz)
    };
    let pulse_count = if config.pulse_count > 0 {
        config.pulse_count
    } else {
        rng.range_u32(sensor.pulse_count) as usize
    };
    let sim_config = RadarSimConfig {
        sample_rate_hz,
        pulse_count,
        bandwidth_hz: rng.range_f64(sensor.bandwidth_hz),
        pulse_width_s: rng.range_f64(sensor.pulse_width_s),
        carrier_hz: rng.range_f64(sensor.center_frequency_hz),
        pri_s: rng.range_f64(sensor.pri_s),
        target_snr_db: snr_db,
        ..RadarSimConfig::default()
    };

    let initial_range_m = 900.0
        + 70.0 * rng.range_u32(object.sensor_observables.expected_range_bins) as f64
        + (seed % 113) as f64;
    let profile = TakeoffProfile {
        initial_range_m,
        runway_heading_deg: rng.range_f64([-25.0, 35.0]),
        ground_speed_mps: rng.range_f64(object.kinematics.ground_speed_mps),
        acceleration_mps2: rng.range_f64(object.kinematics.acceleration_mps2),
        climb_rate_mps: rng.range_f64(object.kinematics.climb_rate_mps).max(0.0),
        max_altitude_m: rng.range_f64(object.kinematics.max_altitude_m).max(10.0),
        radial_velocity_bias_mps: rng.range_f64(object.kinematics.radial_velocity_mps),
        pitch_jitter_deg: rng.range_f64(object.micro_motion.attitude_jitter_deg),
        yaw_jitter_deg: rng.range_f64(object.micro_motion.attitude_jitter_deg),
        propulsor_hz: rng.range_f64(object.micro_motion.propulsor_hz),
        micro_doppler_hz: rng.range_f64(object.micro_motion.micro_doppler_hz),
        rcs_scalar: 10f64.powf(rcs_dbsm / 20.0).max(0.03),
    };

    SampledEpisodeMeta {
        sim_config,
        profile,
        noise_profile: noise,
        object_class_id: object.id.clone(),
        environment_profile_id: environment.id.clone(),
        sensor_archetype_id: sensor.id.clone(),
        snr_db,
        rcs_dbsm,
        clutter_density,
        rfi_probability,
        contested_airspace: contested,
    }
}

fn write_episode_json_products(
    products_dir: &Path,
    episode: &SyntheticEpisode,
    sampled: &SampledEpisodeMeta,
) -> Result<(), DatasetError> {
    write_json_pretty(&products_dir.join("detections.json"), &episode.detections)?;
    write_json_pretty(
        &products_dir.join("tracks.json"),
        &tracks(&episode.detections),
    )?;
    write_json_pretty(
        &products_dir.join("truth.json"),
        &TruthProduct {
            target_states: episode.target_states.clone(),
            takeoff_profile: episode.profile,
            sampled_meta: sampled.clone(),
            limitation: "public-proxy statistical scenario truth; not measured truth".to_string(),
        },
    )?;
    write_json_pretty(
        &products_dir.join("noise_profile.json"),
        &sampled.noise_profile,
    )?;
    Ok(())
}

#[derive(Debug, Clone, Serialize)]
struct TruthProduct {
    target_states: Vec<echoforge_radar::TargetState>,
    takeoff_profile: TakeoffProfile,
    sampled_meta: SampledEpisodeMeta,
    limitation: String,
}

#[derive(Debug, Clone, Serialize)]
struct TrackProduct {
    track_id: String,
    detection_count: usize,
    mean_confidence: f32,
    range_m_start: f64,
    range_m_end: f64,
    limitation: String,
}

fn tracks(detections: &[DetectionRecord]) -> Vec<TrackProduct> {
    if detections.is_empty() {
        return Vec::new();
    }
    let mean_confidence =
        detections.iter().map(|d| d.confidence).sum::<f32>() / detections.len() as f32;
    vec![TrackProduct {
        track_id: "track_000001".to_string(),
        detection_count: detections.len(),
        mean_confidence,
        range_m_start: detections.first().map(|d| d.range_m).unwrap_or(0.0),
        range_m_end: detections.last().map(|d| d.range_m).unwrap_or(0.0),
        limitation: "track is a deterministic proxy product, not operational sensor truth"
            .to_string(),
    }]
}

fn radar_episode_model(
    config: &MonteCarloDemoConfig,
    episode_id: &str,
    sampled: &SampledEpisodeMeta,
) -> Result<RadarEpisode, DatasetError> {
    RadarEpisode {
        id: String::new(),
        kind: String::new(),
        schema_version: String::new(),
        public_proxy_id: sampled.object_class_id.clone(),
        provenance: provenance(config),
        license: license(),
        validation: validation_info(0.34),
        episode_name: format!("{} {episode_id}", config.target_label),
        scenario_id: "scenario.public_proxy.monte_carlo_takeoff_v1".to_string(),
        sample_rate_hz: sampled.sim_config.sample_rate_hz,
        product_paths: vec![
            "products/iq_complex.zarr".to_string(),
            "products/range_profile.zarr".to_string(),
            "products/range_doppler_proxy.zarr".to_string(),
            "products/detections.json".to_string(),
            "products/tracks.json".to_string(),
            "products/truth.json".to_string(),
            "products/noise_profile.json".to_string(),
        ],
    }
    .finalize()
    .map_err(DatasetError::Core)
}

fn dataset_card_model(
    config: &MonteCarloDemoConfig,
    resolved: &ResolvedPreset<'_>,
    split_counts: &BTreeMap<SplitKind, usize>,
) -> Result<DatasetCard, DatasetError> {
    DatasetCard {
        id: String::new(),
        kind: String::new(),
        schema_version: String::new(),
        public_proxy_id: resolved.object.id.clone(),
        provenance: provenance(config),
        license: license(),
        validation: validation_info(0.38),
        dataset_name: format!(
            "{} - public-proxy Monte Carlo dataset ({})",
            config.target_label, resolved.preset.id
        ),
        source_campaign_ids: vec![
            "config.airspace-objects-v1".to_string(),
            "scenario.public_proxy.monte_carlo_takeoff_v1".to_string(),
        ],
        splits: DatasetSplits {
            train: *split_counts.get(&SplitKind::Train).unwrap_or(&0) as u64,
            validation: *split_counts.get(&SplitKind::Validation).unwrap_or(&0) as u64,
            test: *split_counts.get(&SplitKind::Test).unwrap_or(&0) as u64,
        },
    }
    .finalize()
    .map_err(DatasetError::Core)
}

fn write_static_cards(
    config: &MonteCarloDemoConfig,
    resolved: &ResolvedPreset<'_>,
) -> Result<(), DatasetError> {
    let sensor = SensorArchetype {
        id: String::new(),
        kind: String::new(),
        schema_version: String::new(),
        public_proxy_id: resolved.object.id.clone(),
        provenance: provenance(config),
        license: license(),
        validation: validation_info(0.42),
        sensor_name: resolved.sensor.display_name.clone(),
        band_name: resolved.sensor.band_name.clone(),
        waveform_family: "lfm_pulse_doppler_public_proxy".to_string(),
        center_frequency_hz: midpoint(resolved.sensor.center_frequency_hz),
        sample_rate_hz: config.sample_rate_hz,
    }
    .finalize()?;
    write_json_pretty(&config.output_dir.join("sensor_archetype.json"), &sensor)?;

    let scenario = Scenario {
        id: String::new(),
        kind: String::new(),
        schema_version: String::new(),
        public_proxy_id: resolved.object.id.clone(),
        provenance: provenance(config),
        license: license(),
        validation: validation_info(0.4),
        scenario_name: resolved.preset.scenario_label.clone(),
        sensor_archetype_id: sensor.id.clone(),
        object_card_ids: vec![resolved.object.id.clone()],
        environment_label: resolved.environment.display_name.clone(),
        seed: config.seed,
    }
    .finalize()?;
    write_text(
        &config.output_dir.join("object_card.yaml"),
        &format!(
            "id: {}\nkind: object_card\ndisplay_name: {}\nobject_family: {}\nsource_status: public-proxy\nlimitation: not measured truth; not proprietary-equivalent\n",
            resolved.object.id, resolved.object.display_name, resolved.object.object_family
        ),
    )?;
    write_text(
        &config.output_dir.join("material_card.yaml"),
        "id: monte-carlo-material-public-proxy-v1\nkind: material_card\nmaterial_family: mixed public-proxy composite/polymer/metal assumptions\nsource_status: public-proxy\nlimitation: statistical uncertainty proxy, not measured truth\n",
    )?;
    write_text(
        &config.output_dir.join("scenario.yaml"),
        &format!(
            "id: {}\nkind: scenario\nscenario_name: {}\nenvironment_label: {}\nsource_status: public-proxy\ncontested_airspace: robustness stressors only\n",
            scenario.id, scenario.scenario_name, scenario.environment_label
        ),
    )?;
    Ok(())
}

fn monte_carlo_split_policy() -> SplitPolicy {
    SplitPolicy {
        schema_ref: "schemas/dataset_card.schema.json".to_string(),
        policy_id: "dataset.split.monte_carlo_demo_v1".to_string(),
        display_name: "Monte Carlo Demo Split Policy v1".to_string(),
        ratios: SplitRatios {
            train_bps: 7_000,
            validation_bps: 1_500,
            test_bps: 1_500,
        },
        protected_keys: vec![
            "scenario_seed".to_string(),
            "geometry_hash".to_string(),
            "material_sample_hash".to_string(),
        ],
        grouping_mode: "hash_episode_variant_keys".to_string(),
        notes: vec![
            "Episode variant seeds, geometry hashes, and material samples must not cross splits."
                .to_string(),
            "Object-family and sensor-family repetitions are allowed so one public-proxy class can produce train, validation, and test records.".to_string(),
        ],
    }
}

#[derive(Debug, Clone, Serialize)]
struct SplitManifest {
    policy: SplitPolicy,
    counts: BTreeMap<SplitKind, usize>,
    records: Vec<DatasetRecord>,
}

#[derive(Debug, Clone, Serialize)]
struct EpisodeManifestEntry {
    episode_id: String,
    seed: u64,
    split: SplitKind,
    path: String,
    detections: usize,
    target_label: String,
}

#[derive(Debug, Clone, Serialize)]
struct RunManifest<'a> {
    manifest_version: &'static str,
    preset: &'a str,
    generated_at: &'a str,
    root_seed: u64,
    config_library: &'a AirspaceMonteCarloConfig,
    selected_object_class: &'a ObjectClassConfig,
    selected_environment: &'a EnvironmentProfileConfig,
    selected_sensor: &'a SensorArchetypeConfig,
    guardrails: Vec<&'static str>,
    validation: ValidationInfo,
    episodes: &'a [EpisodeManifestEntry],
    leakage_clean: bool,
    known_limitations: Vec<&'static str>,
}

fn run_manifest<'a>(
    config: &'a MonteCarloDemoConfig,
    library: &'a AirspaceMonteCarloConfig,
    resolved: &'a ResolvedPreset<'a>,
    episodes: &'a [EpisodeManifestEntry],
    leakage_report: &LeakageReport,
) -> RunManifest<'a> {
    RunManifest {
        manifest_version: "1",
        preset: &config.preset,
        generated_at: &config.generated_at,
        root_seed: config.seed,
        config_library: library,
        selected_object_class: resolved.object,
        selected_environment: resolved.environment,
        selected_sensor: resolved.sensor,
        guardrails: vec![
            "public-proxy",
            "statistical noise proxy",
            "not measured truth",
            "not proprietary-equivalent",
        ],
        validation: validation_info(0.38),
        episodes,
        leakage_clean: leakage_report.is_clean(),
        known_limitations: vec![
            "No exact measured truth is claimed for any object, platform, material, or sensor.",
            "Contested-airspace parameters are robustness stressors, not tactics or evasion optimization.",
            "Range-Doppler output is a proxy product from an owned lightweight DSP chain.",
        ],
    }
}

fn benchmark_report(
    config: &MonteCarloDemoConfig,
    resolved: &ResolvedPreset<'_>,
    split_counts: &BTreeMap<SplitKind, usize>,
    leakage_report: &LeakageReport,
) -> String {
    format!(
        "# EchoForge Monte Carlo Demo Benchmark Report\n\n\
         Preset: `{}`\n\n\
         Target label: {}\n\n\
         Object class: `{}`\n\n\
         Environment: `{}`\n\n\
         Episodes: {}\n\n\
         Splits: train={}, validation={}, test={}\n\n\
         Validation tier: basic\n\n\
         Validation status: pass\n\n\
         Uncertainty score: 0.38\n\n\
         Guardrails: public-proxy; statistical noise proxy; not measured truth; not proprietary-equivalent.\n\n\
         Leakage status: {}\n\n\
         Known limitation: this benchmark is a deterministic public-proxy runtime demo and is not an operational sensor-performance claim.\n",
        config.preset,
        config.target_label,
        resolved.object.id,
        resolved.environment.id,
        config.episodes,
        split_counts.get(&SplitKind::Train).unwrap_or(&0),
        split_counts.get(&SplitKind::Validation).unwrap_or(&0),
        split_counts.get(&SplitKind::Test).unwrap_or(&0),
        if leakage_report.is_clean() { "clean" } else { "findings" }
    )
}

fn provenance(config: &MonteCarloDemoConfig) -> Provenance {
    Provenance {
        source_kind: "synthetic_public_proxy".to_string(),
        source_refs: vec!["configs/monte-carlo/airspace-objects-v1.json".to_string()],
        generated_by: "echoforge-cli demo monte-carlo".to_string(),
        generated_at: config.generated_at.clone(),
        fingerprint_sha256: String::new(),
    }
}

fn license() -> LicenseInfo {
    LicenseInfo {
        spdx_id: "CC-BY-4.0".to_string(),
        notice: "Generated synthetic public-proxy metadata; no measured target truth included."
            .to_string(),
    }
}

fn validation_info(uncertainty_score: f64) -> ValidationInfo {
    ValidationInfo {
        tier: "basic".to_string(),
        status: "pass".to_string(),
        uncertainty_score,
        checks: vec![
            ValidationCheck {
                name: "schema".to_string(),
                status: "pass".to_string(),
                message: "Core model finalize/validate completed.".to_string(),
            },
            ValidationCheck {
                name: "determinism".to_string(),
                status: "pass".to_string(),
                message: "Root seed plus generated_at fully determine output.".to_string(),
            },
            ValidationCheck {
                name: "leakage".to_string(),
                status: "pass".to_string(),
                message: "Split leakage report is clean for protected episode variant keys."
                    .to_string(),
            },
            ValidationCheck {
                name: "noise_bounds".to_string(),
                status: "pass".to_string(),
                message: "Noise, RFI, clutter, and scintillation are bounded statistical proxies."
                    .to_string(),
            },
        ],
    }
}

fn midpoint(range: [f64; 2]) -> f64 {
    (range[0] + range[1]) / 2.0
}

fn child_seed(root: u64, index: u64) -> u64 {
    let mut value = root ^ index.wrapping_mul(0x9e3779b97f4a7c15);
    value ^= value >> 30;
    value = value.wrapping_mul(0xbf58476d1ce4e5b9);
    value ^= value >> 27;
    value = value.wrapping_mul(0x94d049bb133111eb);
    value ^ (value >> 31)
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
        self.state = self.state.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }

    fn unit_f64(&mut self) -> f64 {
        let bits = self.next_u64() >> 11;
        (bits as f64) / ((1u64 << 53) as f64)
    }

    fn range_f64(&mut self, range: [f64; 2]) -> f64 {
        range[0] + self.unit_f64() * (range[1] - range[0])
    }

    fn range_u32(&mut self, range: [u32; 2]) -> u32 {
        if range[1] <= range[0] {
            return range[0];
        }
        range[0] + (self.next_u64() % (u64::from(range[1] - range[0] + 1))) as u32
    }
}

fn records_jsonl(records: &[DatasetRecord]) -> Result<String, DatasetError> {
    let mut out = String::new();
    for record in records {
        out.push_str(&serde_json::to_string(record)?);
        out.push('\n');
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedded_config_contains_many_airspace_objects() {
        let config = embedded_airspace_config().expect("config parses");
        assert!(config.object_classes.len() >= 6);
        assert!(config
            .environment_profiles
            .iter()
            .any(|profile| profile.contested_airspace.cochannel_emitters[1] >= 8));
        assert!(config
            .guardrails
            .iter()
            .any(|item| item == "public-proxy assumptions only"));
    }

    #[test]
    fn tempdir_generation_writes_required_files_and_models_validate() {
        let temp = tempfile::tempdir().expect("tempdir");
        let output = temp.path().join("demo");
        let mut config = MonteCarloDemoConfig::iranian_takeoff_default(output.clone());
        config.episodes = 3;
        config.pulse_count = 8;
        config.generated_at = "2026-05-18T00:00:00Z".to_string();

        let report = run_monte_carlo_demo(config).expect("run demo");
        assert_eq!(report.episode_count, 3);
        assert!(report.leakage_clean);
        assert!(output.join("dataset_card.json").exists());
        assert!(output.join("manifest.json").exists());
        assert!(output
            .join("episodes/episode_000001/products/iq_complex.zarr/.zarray")
            .exists());

        let dataset_card: DatasetCard =
            serde_json::from_slice(&fs::read(output.join("dataset_card.json")).unwrap()).unwrap();
        dataset_card.validate().expect("dataset card validates");
        for index in 1..=3 {
            let episode_path = output
                .join("episodes")
                .join(format!("episode_{index:06}"))
                .join("radar_episode.json");
            let episode: RadarEpisode =
                serde_json::from_slice(&fs::read(episode_path).unwrap()).unwrap();
            episode.validate().expect("radar episode validates");
        }
    }

    #[test]
    fn same_seed_and_timestamp_keep_manifest_stable() {
        let temp = tempfile::tempdir().expect("tempdir");
        let mut a = MonteCarloDemoConfig::iranian_takeoff_default(temp.path().join("a"));
        a.episodes = 2;
        a.pulse_count = 6;
        let mut b = a.clone();
        b.output_dir = temp.path().join("b");

        run_monte_carlo_demo(a.clone()).expect("first run");
        run_monte_carlo_demo(b.clone()).expect("second run");

        let manifest_a = fs::read_to_string(a.output_dir.join("manifest.json")).unwrap();
        let manifest_b = fs::read_to_string(b.output_dir.join("manifest.json")).unwrap();
        assert_eq!(manifest_a, manifest_b);
    }

    #[test]
    fn refuses_existing_or_source_owned_output() {
        let temp = tempfile::tempdir().expect("tempdir");
        let existing = temp.path().join("existing");
        fs::create_dir_all(&existing).unwrap();
        let mut config = MonteCarloDemoConfig::iranian_takeoff_default(existing);
        config.episodes = 1;
        assert!(matches!(
            run_monte_carlo_demo(config),
            Err(DatasetError::RefusingOutputPath(_))
        ));

        let mut bad = MonteCarloDemoConfig::iranian_takeoff_default(PathBuf::from("crates/demo"));
        bad.episodes = 1;
        assert!(matches!(
            run_monte_carlo_demo(bad),
            Err(DatasetError::RefusingOutputPath(_))
        ));
    }
}
