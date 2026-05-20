use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::error::{EchoSigError, Result};

const MANIFEST_FILE: &str = "manifest.json";
const PROVENANCE_FILE: &str = "provenance.json";
const LICENSE_FILE: &str = "license.json";
const OBJECT_CARD_FILE: &str = "object_card.yaml";
const MATERIAL_CARD_FILE: &str = "material_card.yaml";
const SOLVER_CARD_FILE: &str = "solver_card.yaml";

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct AxisDescriptor {
    pub name: String,
    pub units: Option<String>,
    pub labels: Option<Vec<String>>,
    pub description: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ValidationTier {
    Pending,
    AnalyticV1,
    CrossSolverV1,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct EchoSigManifest {
    pub artifact_id: String,
    pub version: String,
    pub validation_tier: ValidationTier,
    pub axes: Vec<AxisDescriptor>,
    pub tensor_files: Vec<String>,
    pub dynamic_files: Vec<String>,
    pub qa_files: Vec<String>,
}

impl EchoSigManifest {
    pub fn pending(artifact_id: impl Into<String>) -> Self {
        Self {
            artifact_id: artifact_id.into(),
            version: "v0".to_string(),
            validation_tier: ValidationTier::Pending,
            axes: default_axes(),
            tensor_files: vec![
                "tensors/scattering_matrix_complex.zarr".to_string(),
                "tensors/sigma_m2.zarr".to_string(),
                "tensors/sigma_dbsm.zarr".to_string(),
                "tensors/phase_rad.zarr".to_string(),
                "tensors/uncertainty_sigma_db.zarr".to_string(),
                "tensors/confidence.zarr".to_string(),
                "tensors/validity_mask.zarr".to_string(),
            ],
            dynamic_files: vec![
                "dynamic/state_sequence.parquet".to_string(),
                "dynamic/rcs_time_series.zarr".to_string(),
                "dynamic/micro_doppler_spectrogram.zarr".to_string(),
            ],
            qa_files: vec![
                "qa/mesh_quality.json".to_string(),
                "qa/convergence_report.json".to_string(),
                "qa/canonical_validation.json".to_string(),
                "qa/cross_solver_delta.json".to_string(),
                "qa/known_limitations.md".to_string(),
            ],
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ProvenanceRecord {
    pub source: String,
    pub generated_by: String,
    pub created_at_utc: String,
    pub seed: u64,
    pub lineage: Vec<String>,
}

impl ProvenanceRecord {
    pub fn pending() -> Self {
        Self {
            source: "synthetic".to_string(),
            generated_by: "echoforge-sig".to_string(),
            created_at_utc: "1970-01-01T00:00:00Z".to_string(),
            seed: 0,
            lineage: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct LicenseRecord {
    pub expression: String,
    pub spdx_id: Option<String>,
    pub notes: Option<String>,
}

impl LicenseRecord {
    pub fn pending() -> Self {
        Self {
            expression: "Apache-2.0".to_string(),
            spdx_id: Some("Apache-2.0".to_string()),
            notes: Some("pending license record for synthetic bundle".to_string()),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct BundleCard {
    pub kind: String,
    pub raw_yaml: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct EchoSigArtifactBundle {
    pub manifest: EchoSigManifest,
    pub provenance: ProvenanceRecord,
    pub license: LicenseRecord,
    pub object_card: Option<BundleCard>,
    pub material_card: Option<BundleCard>,
    pub solver_card: Option<BundleCard>,
}

impl EchoSigArtifactBundle {
    pub fn pending(artifact_id: impl Into<String>) -> Self {
        Self {
            manifest: EchoSigManifest::pending(artifact_id),
            provenance: ProvenanceRecord::pending(),
            license: LicenseRecord::pending(),
            object_card: None,
            material_card: None,
            solver_card: None,
        }
    }

    pub fn bundle_dir(root: impl AsRef<Path>) -> PathBuf {
        root.as_ref().to_path_buf()
    }

    pub fn write_to_dir(&self, root: impl AsRef<Path>) -> Result<()> {
        let root = root.as_ref();
        fs::create_dir_all(root).map_err(|source| EchoSigError::Io {
            path: root.to_path_buf(),
            source,
        })?;

        write_json(root.join(MANIFEST_FILE), &self.manifest)?;
        write_json(root.join(PROVENANCE_FILE), &self.provenance)?;
        write_json(root.join(LICENSE_FILE), &self.license)?;
        write_card(root.join(OBJECT_CARD_FILE), &self.object_card)?;
        write_card(root.join(MATERIAL_CARD_FILE), &self.material_card)?;
        write_card(root.join(SOLVER_CARD_FILE), &self.solver_card)?;
        Ok(())
    }

    pub fn read_from_dir(root: impl AsRef<Path>) -> Result<Self> {
        let root = root.as_ref();
        let manifest: EchoSigManifest = read_json(root.join(MANIFEST_FILE))?;
        let provenance: ProvenanceRecord = read_json(root.join(PROVENANCE_FILE))?;
        let license: LicenseRecord = read_json(root.join(LICENSE_FILE))?;
        let object_card = read_card(root.join(OBJECT_CARD_FILE))?;
        let material_card = read_card(root.join(MATERIAL_CARD_FILE))?;
        let solver_card = read_card(root.join(SOLVER_CARD_FILE))?;

        Ok(Self {
            manifest,
            provenance,
            license,
            object_card,
            material_card,
            solver_card,
        })
    }
}

pub fn default_axes() -> Vec<AxisDescriptor> {
    vec![
        axis(
            "frequency_hz",
            Some("Hz"),
            "Center or sampled radar frequency",
        ),
        axis("azimuth_deg", Some("deg"), "Azimuth observation angle"),
        axis("elevation_deg", Some("deg"), "Elevation observation angle"),
        axis(
            "tx_polarization",
            None,
            "Transmit polarization label such as h, v, circular",
        ),
        axis(
            "rx_polarization",
            None,
            "Receive polarization label such as h, v, circular",
        ),
        axis(
            "geometry_variant",
            None,
            "Geometry variant or mesh configuration identifier",
        ),
        axis(
            "material_variant",
            None,
            "Material variant or sample identifier",
        ),
        axis("attitude_state", None, "Rigid-body attitude state label"),
        axis("motion_state", None, "Bulk motion or maneuver state label"),
        axis("propulsor_state", None, "Propulsor or rotor state label"),
        axis("solver_regime", None, "Solver regime or fidelity label"),
    ]
}

fn axis(name: &str, units: Option<&str>, description: &str) -> AxisDescriptor {
    AxisDescriptor {
        name: name.to_string(),
        units: units.map(str::to_string),
        labels: None,
        description: Some(description.to_string()),
    }
}

fn write_json<T: Serialize>(path: PathBuf, value: &T) -> Result<()> {
    let bytes = serde_json::to_vec_pretty(value).map_err(|source| EchoSigError::Json {
        path: path.clone(),
        source,
    })?;
    fs::write(&path, bytes).map_err(|source| EchoSigError::Io { path, source })
}

fn read_json<T: for<'de> Deserialize<'de>>(path: PathBuf) -> Result<T> {
    if !path.exists() {
        return Err(EchoSigError::MissingFile(path));
    }
    let bytes = fs::read(&path).map_err(|source| EchoSigError::Io {
        path: path.clone(),
        source,
    })?;
    serde_json::from_slice(&bytes).map_err(|source| EchoSigError::Json { path, source })
}

fn write_card(path: PathBuf, card: &Option<BundleCard>) -> Result<()> {
    if let Some(card) = card {
        fs::write(&path, card.raw_yaml.as_bytes())
            .map_err(|source| EchoSigError::Io { path, source })?;
    }
    Ok(())
}

fn read_card(path: PathBuf) -> Result<Option<BundleCard>> {
    if !path.exists() {
        return Ok(None);
    }
    let raw_yaml = fs::read_to_string(&path).map_err(|source| EchoSigError::Io {
        path: path.clone(),
        source,
    })?;
    let kind = match path.file_stem().and_then(|stem| stem.to_str()) {
        Some(s) => s.to_string(),
        None => String::new(),
    };
    Ok(Some(BundleCard { kind, raw_yaml }))
}
