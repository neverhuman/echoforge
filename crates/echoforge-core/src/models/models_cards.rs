//! Radar-domain card types — extracted from models/mod.rs for LOC compliance.

use serde::{Deserialize, Serialize};

use super::{LicenseInfo, NumericRange, Provenance, ValidationInfo};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RcsCampaign {
    #[serde(default)]
    pub id: String,
    #[serde(default)]
    pub kind: String,
    #[serde(default = "super::default_schema_version")]
    pub schema_version: String,
    pub public_proxy_id: String,
    pub provenance: Provenance,
    pub license: LicenseInfo,
    pub validation: ValidationInfo,
    pub campaign_name: String,
    pub object_card_id: String,
    pub solver_card_id: String,
    pub frequency_range_hz: NumericRange,
    pub azimuth_deg: NumericRange,
    pub tx_polarization: String,
    pub rx_polarization: String,
    pub run_count: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EchosigManifest {
    #[serde(default)]
    pub id: String,
    #[serde(default)]
    pub kind: String,
    #[serde(default = "super::default_schema_version")]
    pub schema_version: String,
    pub public_proxy_id: String,
    pub provenance: Provenance,
    pub license: LicenseInfo,
    pub validation: ValidationInfo,
    pub artifact_name: String,
    pub object_card_id: String,
    #[serde(default)]
    pub tensor_axes: Vec<String>,
    #[serde(default)]
    pub tensor_paths: Vec<String>,
    #[serde(default)]
    pub qa_paths: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SensorArchetype {
    #[serde(default)]
    pub id: String,
    #[serde(default)]
    pub kind: String,
    #[serde(default = "super::default_schema_version")]
    pub schema_version: String,
    pub public_proxy_id: String,
    pub provenance: Provenance,
    pub license: LicenseInfo,
    pub validation: ValidationInfo,
    pub sensor_name: String,
    pub band_name: String,
    pub waveform_family: String,
    pub center_frequency_hz: f64,
    pub sample_rate_hz: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Scenario {
    #[serde(default)]
    pub id: String,
    #[serde(default)]
    pub kind: String,
    #[serde(default = "super::default_schema_version")]
    pub schema_version: String,
    pub public_proxy_id: String,
    pub provenance: Provenance,
    pub license: LicenseInfo,
    pub validation: ValidationInfo,
    pub scenario_name: String,
    pub sensor_archetype_id: String,
    #[serde(default)]
    pub object_card_ids: Vec<String>,
    pub environment_label: String,
    pub seed: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RadarEpisode {
    #[serde(default)]
    pub id: String,
    #[serde(default)]
    pub kind: String,
    #[serde(default = "super::default_schema_version")]
    pub schema_version: String,
    pub public_proxy_id: String,
    pub provenance: Provenance,
    pub license: LicenseInfo,
    pub validation: ValidationInfo,
    pub episode_name: String,
    pub scenario_id: String,
    pub sample_rate_hz: f64,
    #[serde(default)]
    pub product_paths: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DetectorGraph {
    #[serde(default)]
    pub id: String,
    #[serde(default)]
    pub kind: String,
    #[serde(default = "super::default_schema_version")]
    pub schema_version: String,
    pub public_proxy_id: String,
    pub provenance: Provenance,
    pub license: LicenseInfo,
    pub validation: ValidationInfo,
    pub graph_name: String,
    #[serde(default)]
    pub nodes: Vec<String>,
    #[serde(default)]
    pub edges: Vec<String>,
}
