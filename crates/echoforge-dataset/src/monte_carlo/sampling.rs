use std::path::Path;

use echoforge_core::models::RadarEpisode;
use echoforge_radar::{
    synthesize_takeoff_episode, DetectionRecord, EpisodeSeed, NoiseProfile, RadarSimConfig,
    SyntheticEpisode, TakeoffProfile,
};
use serde::Serialize;

use crate::export::write_json_pretty;

use super::config::MonteCarloDemoConfig;
use super::error::DatasetError;
use super::helpers::{license, provenance, validation_info, SplitMix64};
use super::scene_config::{
    AirspaceMonteCarloConfig, EnvironmentProfileConfig, ObjectClassConfig, PresetConfig,
    SensorArchetypeConfig,
};

#[path = "sampling_writers.rs"]
mod sampling_writers;
pub(super) use sampling_writers::write_static_cards;

// ── resolved preset ───────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub(super) struct ResolvedPreset<'a> {
    pub preset: &'a PresetConfig,
    pub object: &'a ObjectClassConfig,
    pub environment: &'a EnvironmentProfileConfig,
    pub sensor: &'a SensorArchetypeConfig,
}

impl<'a> ResolvedPreset<'a> {
    pub(super) fn resolve(
        library: &'a AirspaceMonteCarloConfig,
        preset_id: &str,
    ) -> Result<Self, DatasetError> {
        let preset = match library.presets.iter().find(|p| p.id == preset_id) {
            Some(p) => p,
            None => return Err(DatasetError::InvalidConfig(format!("unknown preset {preset_id}"))),
        };
        let object = match library.object_classes.iter().find(|o| o.id == preset.object_class_id) {
            Some(o) => o,
            None => return Err(DatasetError::InvalidConfig(format!(
                "preset {} references missing object {}",
                preset.id, preset.object_class_id
            ))),
        };
        let environment = match library.environment_profiles.iter().find(|e| e.id == preset.environment_profile_id) {
            Some(e) => e,
            None => return Err(DatasetError::InvalidConfig(format!(
                "preset {} references missing environment {}",
                preset.id, preset.environment_profile_id
            ))),
        };
        let sensor = match library.sensor_archetypes.iter().find(|s| s.id == preset.sensor_archetype_id) {
            Some(s) => s,
            None => return Err(DatasetError::InvalidConfig(format!(
                "preset {} references missing sensor {}",
                preset.id, preset.sensor_archetype_id
            ))),
        };
        Ok(Self {
            preset,
            object,
            environment,
            sensor,
        })
    }
}

// ── sampled episode structs ───────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize)]
pub(super) struct SampledEpisodeMeta {
    pub sim_config: RadarSimConfig,
    pub profile: TakeoffProfile,
    pub noise_profile: NoiseProfile,
    pub object_class_id: String,
    pub environment_profile_id: String,
    pub sensor_archetype_id: String,
    pub snr_db: f64,
    pub rcs_dbsm: f64,
    pub clutter_density: f64,
    pub rfi_probability: f64,
    pub contested_airspace: ContestedSample,
}

#[derive(Debug, Clone, Serialize)]
pub(super) struct ContestedSample {
    pub traffic_density: f64,
    pub cochannel_emitters: u32,
    pub uncooperative_transponder_fraction: f64,
    pub multipath_severity: f64,
    pub sensor_dropout_probability: f64,
    pub label_ambiguity_probability: f64,
}

// ── episode sampling ──────────────────────────────────────────────────────────

pub(super) fn sample_episode(
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
        // Wave 2 Lane D: opt into multi-blade PropellerGenerator dispatch
        // by setting `Some(...)`; prior single-sinusoid retained when both
        // are `None`. Dataset-tier defaults to prior for byte-stability.
        blade_count: None,
        blade_length_m: None,
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

/// Synthesize one takeoff episode from sampled parameters.
/// `sim_config` is cloned before moving because `RadarSimConfig` is not `Copy`.
pub(super) fn synthesize_episode(sampled: &SampledEpisodeMeta, episode_seed: u64) -> SyntheticEpisode {
    synthesize_takeoff_episode(
        sampled.sim_config.clone(),
        sampled.profile,
        sampled.noise_profile,
        EpisodeSeed(episode_seed),
    )
}

// ── episode product writers ───────────────────────────────────────────────────

pub(super) fn write_episode_json_products(
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

// ── model constructors ────────────────────────────────────────────────────────

pub(super) fn radar_episode_model(
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

