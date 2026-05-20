use serde_json::json;

use crate::error::CoreError;
use crate::id::deterministic_id;
use crate::validation::{
    ensure_non_empty, ensure_non_empty_vec, ensure_probability, ensure_slug, SCHEMA_VERSION,
};

use super::{
    DatasetCard, DetectorGraph, EchosigManifest, MaterialCard, MeshManifest, ObjectCard,
    RadarEpisode, RcsCampaign, Scenario, SensorArchetype, SolverCard, ValidationReport,
};

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
