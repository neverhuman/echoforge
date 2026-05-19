use serde::{Deserialize, Serialize};

use crate::validation::SCHEMA_VERSION;

pub mod models_ext;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Provenance {
    #[serde(default = "default_source_kind")]
    pub source_kind: String,
    #[serde(default)]
    pub source_refs: Vec<String>,
    #[serde(default)]
    pub generated_by: String,
    #[serde(default)]
    pub generated_at: String,
    #[serde(default)]
    pub fingerprint_sha256: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LicenseInfo {
    #[serde(default)]
    pub spdx_id: String,
    #[serde(default)]
    pub notice: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ValidationCheck {
    #[serde(default)]
    pub name: String,
    #[serde(default = "default_check_status")]
    pub status: String,
    #[serde(default)]
    pub message: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ValidationInfo {
    #[serde(default = "default_validation_tier")]
    pub tier: String,
    #[serde(default = "default_check_status")]
    pub status: String,
    #[serde(default)]
    pub uncertainty_score: f64,
    #[serde(default)]
    pub checks: Vec<ValidationCheck>,
    // Method-ceiling fidelity tier F0..F5 (fidelity-class-field packet).
    // Independent from `tier` (which tracks evidence accumulation). Optional
    // and skipped when None so existing canonical payloads and golden
    // SHA-256 fixtures remain byte-stable; producers that have not yet
    // adopted the dual-axis vocabulary may continue to omit the field.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fidelity_class: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Vector3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct NumericRange {
    pub min: f64,
    pub max: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ComplexScalar {
    pub real: f64,
    pub imag: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ObjectCard {
    #[serde(default)]
    pub id: String,
    #[serde(default)]
    pub kind: String,
    #[serde(default = "default_schema_version")]
    pub schema_version: String,
    pub public_proxy_id: String,
    pub provenance: Provenance,
    pub license: LicenseInfo,
    pub validation: ValidationInfo,
    pub display_name: String,
    pub object_family: String,
    pub geometry_variant: String,
    pub material_variant: String,
    pub dimensions_m: Vector3,
    #[serde(default)]
    pub tags: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MaterialCard {
    #[serde(default)]
    pub id: String,
    #[serde(default)]
    pub kind: String,
    #[serde(default = "default_schema_version")]
    pub schema_version: String,
    pub public_proxy_id: String,
    pub provenance: Provenance,
    pub license: LicenseInfo,
    pub validation: ValidationInfo,
    pub material_name: String,
    pub material_family: String,
    pub frequency_range_hz: NumericRange,
    pub permittivity: ComplexScalar,
    pub conductivity_s_per_m: f64,
    #[serde(default)]
    pub roughness_m: f64,
    // --- v2 additive fields (FUCKIT.md.done section B; material-card-v2 packet).
    // All Option<...> and skipped when None so the canonical JSON
    // serialization, golden SHA-256 hashes, and v1 fixtures remain stable.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub frequency_validity_hz: Option<NumericRange>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub solver_compatibility: Option<std::collections::BTreeMap<String, String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub uncertainty_policy: Option<MaterialUncertaintyPolicy>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub confidence_grade: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thickness_m: Option<NumericRange>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub layer_stackup: Option<Vec<MaterialLayer>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub anisotropy_flag: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub loss_tangent_distribution: Option<LossTangentDistribution>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MaterialUncertaintyPolicy {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sample_count_default: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub propagate_to_rcs_uncertainty: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub downgrade_confidence_if_unvalidated: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub epsilon_relative_sigma: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub epsilon_imag_relative_sigma: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MaterialLayer {
    pub name: String,
    pub thickness_m: NumericRange,
    pub epsilon_real_range: NumericRange,
    pub epsilon_imag_range: NumericRange,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LossTangentDistribution {
    pub mean: f64,
    pub sigma: f64,
    pub samples: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MeshManifest {
    #[serde(default)]
    pub id: String,
    #[serde(default)]
    pub kind: String,
    #[serde(default = "default_schema_version")]
    pub schema_version: String,
    pub public_proxy_id: String,
    pub provenance: Provenance,
    pub license: LicenseInfo,
    pub validation: ValidationInfo,
    pub mesh_name: String,
    pub mesh_format: String,
    pub source_files: Vec<String>,
    pub units: String,
    pub triangle_count: u64,
    pub watertight: bool,
    pub mesh_sha256: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SolverCard {
    #[serde(default)]
    pub id: String,
    #[serde(default)]
    pub kind: String,
    #[serde(default = "default_schema_version")]
    pub schema_version: String,
    pub public_proxy_id: String,
    pub provenance: Provenance,
    pub license: LicenseInfo,
    pub validation: ValidationInfo,
    pub solver_name: String,
    pub solver_family: String,
    pub version: String,
    pub container_image: String,
    #[serde(default)]
    pub supported_polarizations: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RcsCampaign {
    #[serde(default)]
    pub id: String,
    #[serde(default)]
    pub kind: String,
    #[serde(default = "default_schema_version")]
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
    #[serde(default = "default_schema_version")]
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
    #[serde(default = "default_schema_version")]
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
    #[serde(default = "default_schema_version")]
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
    #[serde(default = "default_schema_version")]
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
    #[serde(default = "default_schema_version")]
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

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DatasetSplits {
    pub train: u64,
    pub validation: u64,
    pub test: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DatasetCard {
    #[serde(default)]
    pub id: String,
    #[serde(default)]
    pub kind: String,
    #[serde(default = "default_schema_version")]
    pub schema_version: String,
    pub public_proxy_id: String,
    pub provenance: Provenance,
    pub license: LicenseInfo,
    pub validation: ValidationInfo,
    pub dataset_name: String,
    #[serde(default)]
    pub source_campaign_ids: Vec<String>,
    pub splits: DatasetSplits,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ValidationReport {
    #[serde(default)]
    pub id: String,
    #[serde(default)]
    pub kind: String,
    #[serde(default = "default_schema_version")]
    pub schema_version: String,
    pub public_proxy_id: String,
    pub provenance: Provenance,
    pub license: LicenseInfo,
    pub validation: ValidationInfo,
    pub report_name: String,
    pub subject_kind: String,
    pub subject_id: String,
    #[serde(default)]
    pub checks: Vec<ValidationCheck>,
    pub overall_status: String,
}

fn default_schema_version() -> String {
    SCHEMA_VERSION.to_string()
}

fn default_source_kind() -> String {
    "synthetic".to_string()
}

fn default_check_status() -> String {
    "warn".to_string()
}

fn default_validation_tier() -> String {
    "unvalidated".to_string()
}
