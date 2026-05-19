//! World and object-pack scaffolding for EchoForge.
//!
//! This crate stays deliberately neutral: it models public-proxy object packs,
//! hard-negative packs, and scenario manifests while referencing the shared
//! schema filenames expected from the repo-level schema package.
//!
//! Object cards, material cards, and scenario manifests now follow the
//! canonical YAML envelope (id / kind / schema_version / public_proxy_id /
//! provenance / license / validation) defined in `schemas/`. Pack-level
//! manifests still use the lightweight JSON shape with `schema_ref`.

use serde::{de::DeserializeOwned, Deserialize, Serialize};

pub mod schema {
    pub const OBJECT_CARD: &str = "schemas/object_card.schema.json";
    pub const MATERIAL_CARD: &str = "schemas/material_card.schema.json";
    pub const OBJECT_PACK_MANIFEST: &str = "schemas/object_pack_manifest.schema.json";
    pub const HARD_NEGATIVE_PACK: &str = "schemas/hard_negative_pack.schema.json";
    pub const MESH_MANIFEST: &str = "schemas/mesh_manifest.schema.json";
    pub const SCENARIO: &str = "schemas/scenario.schema.json";
}

pub fn from_json_str<T: DeserializeOwned>(input: &str) -> serde_json::Result<T> {
    serde_json::from_str(input)
}

pub fn from_yaml_str<T: DeserializeOwned>(input: &str) -> serde_yaml::Result<T> {
    serde_yaml::from_str(input)
}

/// Shared canonical envelope fields used by every individual artifact
/// (object cards, material cards, scenarios). Pack manifests intentionally
/// stay on the lighter JSON shape and do not embed this envelope.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Provenance {
    pub source_kind: String,
    pub source_refs: Vec<String>,
    pub generated_by: String,
    pub generated_at: String,
    pub fingerprint_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct License {
    pub spdx_id: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ValidationCheck {
    pub name: String,
    pub status: String,
    pub message: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Validation {
    pub tier: String,
    pub status: String,
    pub uncertainty_score: f64,
    pub checks: Vec<ValidationCheck>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Dimensions {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FrequencyRange {
    pub min: f64,
    pub max: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Permittivity {
    pub real: f64,
    pub imag: f64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PackManifest {
    pub schema_ref: String,
    pub id: String,
    pub display_name: String,
    pub purpose: String,
    pub public_proxy: bool,
    pub object_card_refs: Vec<String>,
    pub material_card_refs: Vec<String>,
    pub validation_tier: String,
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ObjectCard {
    pub id: String,
    pub kind: String,
    pub schema_version: String,
    pub public_proxy_id: String,
    pub provenance: Provenance,
    pub license: License,
    pub validation: Validation,
    pub display_name: String,
    pub object_family: String,
    pub geometry_variant: String,
    pub material_variant: String,
    pub dimensions_m: Dimensions,
    pub tags: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MaterialCard {
    pub id: String,
    pub kind: String,
    pub schema_version: String,
    pub public_proxy_id: String,
    pub provenance: Provenance,
    pub license: License,
    pub validation: Validation,
    pub material_name: String,
    pub material_family: String,
    pub frequency_range_hz: FrequencyRange,
    pub permittivity: Permittivity,
    pub conductivity_s_per_m: f64,
    pub roughness_m: f64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HardNegativePack {
    pub schema_ref: String,
    pub id: String,
    pub display_name: String,
    pub source_status: String,
    pub confuser_traits: Vec<String>,
    pub notes: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScenarioManifest {
    pub id: String,
    pub kind: String,
    pub schema_version: String,
    pub public_proxy_id: String,
    pub provenance: Provenance,
    pub license: License,
    pub validation: Validation,
    pub scenario_name: String,
    pub sensor_archetype_id: String,
    pub object_card_ids: Vec<String>,
    pub environment_label: String,
    pub seed: u64,
}

pub fn public_proxy_object_pack() -> PackManifest {
    PackManifest {
        schema_ref: schema::OBJECT_PACK_MANIFEST.to_string(),
        id: "pack.public_proxy_v1".to_string(),
        display_name: "Public Proxy Object Pack v1".to_string(),
        purpose: "Neutral public-proxy object assumptions for radar-first dataset generation."
            .to_string(),
        public_proxy: true,
        object_card_refs: vec![
            "object_card.yaml".to_string(),
            "material_card.yaml".to_string(),
        ],
        material_card_refs: vec!["material_card.yaml".to_string()],
        validation_tier: "draft".to_string(),
        notes: vec![
            "No exact real-world truth claim is implied.".to_string(),
            "Used to seed schema-driven scenario and dataset scaffolds.".to_string(),
        ],
    }
}

pub fn object_pack_manifest_schema_ref() -> &'static str {
    schema::OBJECT_PACK_MANIFEST
}

pub fn hard_negative_pack_schema_ref() -> &'static str {
    schema::HARD_NEGATIVE_PACK
}

pub fn mesh_manifest_schema_ref() -> &'static str {
    schema::MESH_MANIFEST
}

pub fn public_proxy_hard_negative_pack() -> HardNegativePack {
    HardNegativePack {
        schema_ref: schema::HARD_NEGATIVE_PACK.to_string(),
        id: "hn.public_proxy.robustness_v1".to_string(),
        display_name: "Public Proxy Hard-Negative Pack".to_string(),
        source_status: "public_proxy".to_string(),
        confuser_traits: vec![
            "birds_and_flocks".to_string(),
            "commercial_drones".to_string(),
            "balloons_and_kites".to_string(),
            "terrain_and_vegetation_glints".to_string(),
            "vehicles_and_cranes".to_string(),
            "weather_and_rfi".to_string(),
        ],
        notes: "Hard-negative pending entries for robustness and false-alarm stress testing."
            .to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_object_pack_mentions_shared_schema() {
        let pack = public_proxy_object_pack();
        assert_eq!(pack.schema_ref, schema::OBJECT_PACK_MANIFEST);
        assert!(pack.public_proxy);
    }

    #[test]
    fn fixture_object_card_parses() {
        let card: ObjectCard = from_yaml_str(include_str!(
            "../../../object-packs/public-proxy-v1/object_card.yaml"
        ))
        .expect("object card fixture should parse");
        assert_eq!(card.kind, "object_card");
        assert_eq!(card.public_proxy_id, "public-proxy-uav-1");
        assert_eq!(card.object_family, "fixed_wing_uav");
    }

    #[test]
    fn fixture_material_card_parses() {
        let card: MaterialCard = from_yaml_str(include_str!(
            "../../../object-packs/public-proxy-v1/material_card.yaml"
        ))
        .expect("material card fixture should parse");
        assert_eq!(card.kind, "material_card");
        assert_eq!(card.material_family, "composite");
    }

    #[test]
    fn fixture_scenario_parses() {
        let scenario: ScenarioManifest = from_yaml_str(include_str!(
            "../../../scenarios/public-proxy-clutter-v1/scenario.yaml"
        ))
        .expect("scenario fixture should parse");
        assert_eq!(scenario.kind, "scenario");
        assert_eq!(scenario.public_proxy_id, "public-proxy-clutter-v1");
        assert_eq!(scenario.environment_label, "moderate-clutter-light-weather");
    }

    #[test]
    fn fixture_object_pack_manifest_parses() {
        let pack: PackManifest = from_json_str(include_str!(
            "../../../object-packs/public-proxy-v1/pack.manifest.json"
        ))
        .expect("object pack manifest fixture should parse");
        assert_eq!(pack.schema_ref, schema::OBJECT_PACK_MANIFEST);
        assert!(pack.public_proxy);
    }

    #[test]
    fn fixture_hard_negative_pack_manifest_parses() {
        let pack: HardNegativePack = from_json_str(include_str!(
            "../../../object-packs/hard-negatives/pack.manifest.json"
        ))
        .expect("hard-negative pack manifest fixture should parse");
        assert_eq!(pack.schema_ref, schema::HARD_NEGATIVE_PACK);
        assert!(pack
            .confuser_traits
            .iter()
            .any(|item| item == "weather_and_rfi"));
    }

    #[test]
    fn hard_negative_pack_is_neutral() {
        let pack = public_proxy_hard_negative_pack();
        assert!(pack
            .confuser_traits
            .iter()
            .any(|item| item == "weather_and_rfi"));
        assert_eq!(pack.source_status, "public_proxy");
    }
}
