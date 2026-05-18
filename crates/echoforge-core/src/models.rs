use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::error::CoreError;
use crate::id::deterministic_id;
use crate::validation::{
    ensure_non_empty, ensure_non_empty_vec, ensure_probability, ensure_slug, SCHEMA_VERSION,
};

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

macro_rules! impl_document {
    ($ty:ty, $kind:literal, [$($field:ident),+]) => {
        impl $ty {
            pub const KIND: &'static str = $kind;

            pub fn finalize(mut self) -> Result<Self, CoreError> {
                self.validate_fields()?;
                let payload = json!({$( stringify!($field): &self.$field ),+});
                let fingerprint = crate::id::fingerprint_sha256(&payload)?;
                if self.provenance.fingerprint_sha256.is_empty() {
                    self.provenance.fingerprint_sha256 = fingerprint.clone();
                } else if self.provenance.fingerprint_sha256 != fingerprint {
                    return Err(CoreError::Validation("provenance fingerprint does not match payload".to_string()));
                }
                let expected_id = deterministic_id(Self::KIND, &self.public_proxy_id, &payload)?;
                if self.id.is_empty() {
                    self.id = expected_id;
                } else if self.id != expected_id {
                    return Err(CoreError::Validation("deterministic id does not match payload".to_string()));
                }
                self.kind = Self::KIND.to_string();
                self.schema_version = SCHEMA_VERSION.to_string();
                Ok(self)
            }

            pub fn validate(&self) -> Result<(), CoreError> {
                self.clone().finalize().map(|_| ())
            }
        }
    };
}

impl ObjectCard {
    fn validate_fields(&self) -> Result<(), CoreError> {
        ensure_slug(&self.public_proxy_id, "public_proxy_id")?;
        ensure_non_empty(&self.display_name, "display_name")?;
        ensure_non_empty(&self.object_family, "object_family")?;
        ensure_non_empty(&self.geometry_variant, "geometry_variant")?;
        ensure_non_empty(&self.material_variant, "material_variant")?;
        ensure_non_empty_vec(&self.tags, "tags")?;
        Ok(())
    }
}
impl_document!(
    ObjectCard,
    "object_card",
    [
        display_name,
        object_family,
        geometry_variant,
        material_variant,
        dimensions_m,
        tags
    ]
);

impl MaterialCard {
    fn validate_fields(&self) -> Result<(), CoreError> {
        ensure_slug(&self.public_proxy_id, "public_proxy_id")?;
        ensure_non_empty(&self.material_name, "material_name")?;
        ensure_non_empty(&self.material_family, "material_family")?;
        if self.frequency_range_hz.max < self.frequency_range_hz.min {
            return Err(CoreError::Validation(
                "frequency_range_hz must have max >= min".to_string(),
            ));
        }
        Ok(())
    }
}
impl_document!(
    MaterialCard,
    "material_card",
    [
        material_name,
        material_family,
        frequency_range_hz,
        permittivity,
        conductivity_s_per_m,
        roughness_m
    ]
);

impl MeshManifest {
    fn validate_fields(&self) -> Result<(), CoreError> {
        ensure_slug(&self.public_proxy_id, "public_proxy_id")?;
        ensure_non_empty(&self.mesh_name, "mesh_name")?;
        ensure_non_empty(&self.mesh_format, "mesh_format")?;
        ensure_non_empty_vec(&self.source_files, "source_files")?;
        ensure_non_empty(&self.units, "units")?;
        Ok(())
    }
}
impl_document!(
    MeshManifest,
    "mesh_manifest",
    [
        mesh_name,
        mesh_format,
        source_files,
        units,
        triangle_count,
        watertight,
        mesh_sha256
    ]
);

impl SolverCard {
    fn validate_fields(&self) -> Result<(), CoreError> {
        ensure_slug(&self.public_proxy_id, "public_proxy_id")?;
        ensure_non_empty(&self.solver_name, "solver_name")?;
        ensure_non_empty(&self.solver_family, "solver_family")?;
        ensure_non_empty(&self.version, "version")?;
        ensure_non_empty_vec(&self.supported_polarizations, "supported_polarizations")?;
        Ok(())
    }
}
impl_document!(
    SolverCard,
    "solver_card",
    [
        solver_name,
        solver_family,
        version,
        container_image,
        supported_polarizations
    ]
);

impl RcsCampaign {
    fn validate_fields(&self) -> Result<(), CoreError> {
        ensure_slug(&self.public_proxy_id, "public_proxy_id")?;
        ensure_non_empty(&self.campaign_name, "campaign_name")?;
        ensure_non_empty(&self.object_card_id, "object_card_id")?;
        ensure_non_empty(&self.solver_card_id, "solver_card_id")?;
        ensure_non_empty(&self.tx_polarization, "tx_polarization")?;
        ensure_non_empty(&self.rx_polarization, "rx_polarization")?;
        if self.run_count == 0 {
            return Err(CoreError::Validation("run_count must be > 0".to_string()));
        }
        if self.frequency_range_hz.max < self.frequency_range_hz.min {
            return Err(CoreError::Validation(
                "frequency_range_hz must have max >= min".to_string(),
            ));
        }
        if self.azimuth_deg.max < self.azimuth_deg.min {
            return Err(CoreError::Validation(
                "azimuth_deg must have max >= min".to_string(),
            ));
        }
        Ok(())
    }
}
impl_document!(
    RcsCampaign,
    "rcs_campaign",
    [
        campaign_name,
        object_card_id,
        solver_card_id,
        frequency_range_hz,
        azimuth_deg,
        tx_polarization,
        rx_polarization,
        run_count
    ]
);

impl EchosigManifest {
    fn validate_fields(&self) -> Result<(), CoreError> {
        ensure_slug(&self.public_proxy_id, "public_proxy_id")?;
        ensure_non_empty(&self.artifact_name, "artifact_name")?;
        ensure_non_empty(&self.object_card_id, "object_card_id")?;
        ensure_non_empty_vec(&self.tensor_axes, "tensor_axes")?;
        ensure_non_empty_vec(&self.tensor_paths, "tensor_paths")?;
        Ok(())
    }
}
impl_document!(
    EchosigManifest,
    "echosig_manifest",
    [
        artifact_name,
        object_card_id,
        tensor_axes,
        tensor_paths,
        qa_paths
    ]
);

impl SensorArchetype {
    fn validate_fields(&self) -> Result<(), CoreError> {
        ensure_slug(&self.public_proxy_id, "public_proxy_id")?;
        ensure_non_empty(&self.sensor_name, "sensor_name")?;
        ensure_non_empty(&self.band_name, "band_name")?;
        ensure_non_empty(&self.waveform_family, "waveform_family")?;
        if self.center_frequency_hz <= 0.0 {
            return Err(CoreError::Validation(
                "center_frequency_hz must be positive".to_string(),
            ));
        }
        if self.sample_rate_hz <= 0.0 {
            return Err(CoreError::Validation(
                "sample_rate_hz must be positive".to_string(),
            ));
        }
        Ok(())
    }
}
impl_document!(
    SensorArchetype,
    "sensor_archetype",
    [
        sensor_name,
        band_name,
        waveform_family,
        center_frequency_hz,
        sample_rate_hz
    ]
);

impl Scenario {
    fn validate_fields(&self) -> Result<(), CoreError> {
        ensure_slug(&self.public_proxy_id, "public_proxy_id")?;
        ensure_non_empty(&self.scenario_name, "scenario_name")?;
        ensure_non_empty(&self.sensor_archetype_id, "sensor_archetype_id")?;
        ensure_non_empty_vec(&self.object_card_ids, "object_card_ids")?;
        ensure_non_empty(&self.environment_label, "environment_label")?;
        Ok(())
    }
}
impl_document!(
    Scenario,
    "scenario",
    [
        scenario_name,
        sensor_archetype_id,
        object_card_ids,
        environment_label,
        seed
    ]
);

impl RadarEpisode {
    fn validate_fields(&self) -> Result<(), CoreError> {
        ensure_slug(&self.public_proxy_id, "public_proxy_id")?;
        ensure_non_empty(&self.episode_name, "episode_name")?;
        ensure_non_empty(&self.scenario_id, "scenario_id")?;
        if self.sample_rate_hz <= 0.0 {
            return Err(CoreError::Validation(
                "sample_rate_hz must be positive".to_string(),
            ));
        }
        ensure_non_empty_vec(&self.product_paths, "product_paths")?;
        Ok(())
    }
}
impl_document!(
    RadarEpisode,
    "radar_episode",
    [episode_name, scenario_id, sample_rate_hz, product_paths]
);

impl DetectorGraph {
    fn validate_fields(&self) -> Result<(), CoreError> {
        ensure_slug(&self.public_proxy_id, "public_proxy_id")?;
        ensure_non_empty(&self.graph_name, "graph_name")?;
        ensure_non_empty_vec(&self.nodes, "nodes")?;
        Ok(())
    }
}
impl_document!(DetectorGraph, "detector_graph", [graph_name, nodes, edges]);

impl DatasetCard {
    fn validate_fields(&self) -> Result<(), CoreError> {
        ensure_slug(&self.public_proxy_id, "public_proxy_id")?;
        ensure_non_empty(&self.dataset_name, "dataset_name")?;
        ensure_non_empty_vec(&self.source_campaign_ids, "source_campaign_ids")?;
        ensure_non_empty_vec(&self.source_campaign_ids, "source_campaign_ids")?;
        Ok(())
    }
}
impl_document!(
    DatasetCard,
    "dataset_card",
    [dataset_name, source_campaign_ids, splits]
);

impl ValidationReport {
    fn validate_fields(&self) -> Result<(), CoreError> {
        ensure_slug(&self.public_proxy_id, "public_proxy_id")?;
        ensure_non_empty(&self.report_name, "report_name")?;
        ensure_non_empty(&self.subject_kind, "subject_kind")?;
        ensure_non_empty(&self.subject_id, "subject_id")?;
        ensure_non_empty_vec(&self.checks, "checks")?;
        ensure_non_empty(&self.overall_status, "overall_status")?;
        ensure_probability(
            self.validation.uncertainty_score,
            "validation.uncertainty_score",
        )?;
        Ok(())
    }
}
impl_document!(
    ValidationReport,
    "validation_report",
    [
        report_name,
        subject_kind,
        subject_id,
        checks,
        overall_status
    ]
);

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
