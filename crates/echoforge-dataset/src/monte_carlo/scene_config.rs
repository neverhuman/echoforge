use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use super::error::DatasetError;

const CONFIG_JSON: &str = include_str!("../../../../configs/monte-carlo/airspace-objects-v1.json");

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
