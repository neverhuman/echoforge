//! World and object-pack scaffolding for EchoForge.
//!
//! This crate stays deliberately neutral: it models public-proxy object packs,
//! hard-negative packs, and scenario manifests while referencing the shared
//! schema filenames expected from the repo-level schema package.

use serde::{de::DeserializeOwned, Deserialize, Serialize};
use std::collections::BTreeMap;

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

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ObjectCard {
    pub schema_ref: String,
    pub id: String,
    pub display_name: String,
    pub family: String,
    pub public_proxy: bool,
    pub source_status: String,
    pub geometry_variants: Vec<String>,
    pub material_variants: Vec<String>,
    pub motion_variants: Vec<String>,
    pub notes: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MaterialCard {
    pub schema_ref: String,
    pub id: String,
    pub display_name: String,
    pub base_material: String,
    pub frequency_band_hz: Vec<u64>,
    pub parameter_model: ParameterModel,
    pub uncertainty_model: UncertaintyModel,
    pub notes: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ParameterModel {
    pub model_type: String,
    pub parameters: BTreeMap<String, String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct UncertaintyModel {
    pub model_type: String,
    pub bands: Vec<UncertaintyBand>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct UncertaintyBand {
    pub name: String,
    pub sigma_db: String,
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

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ScenarioManifest {
    pub schema_ref: String,
    pub id: String,
    pub display_name: String,
    pub object_pack_refs: Vec<String>,
    pub hard_negative_pack_refs: Vec<String>,
    pub sensor_archetype_ref: String,
    pub split_policy_ref: String,
    pub validation_tier: String,
    pub environment: EnvironmentRecipe,
    pub notes: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EnvironmentRecipe {
    pub clutter: String,
    pub weather: String,
    pub multipath: String,
    pub rfi: String,
    pub difficulty_tier: String,
}

pub fn public_proxy_object_pack() -> PackManifest {
    PackManifest {
        schema_ref: schema::OBJECT_PACK_MANIFEST.to_string(),
        id: "pack.public_proxy_v1".to_string(),
        display_name: "Public Proxy Object Pack v1".to_string(),
        purpose: "Neutral public-proxy object assumptions for radar-first dataset generation.".to_string(),
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

pub fn public_proxy_object_card() -> ObjectCard {
    ObjectCard {
        schema_ref: schema::OBJECT_CARD.to_string(),
        id: "obj.public_proxy.fixed_wing_v1".to_string(),
        display_name: "Public Proxy Fixed-Wing Object".to_string(),
        family: "fixed_wing_uav".to_string(),
        public_proxy: true,
        source_status: "public_proxy".to_string(),
        geometry_variants: vec![
            "baseline_airframe".to_string(),
            "landing_configuration".to_string(),
            "takeoff_configuration".to_string(),
        ],
        material_variants: vec![
            "composite_skin_proxy".to_string(),
            "metal_fastener_proxy".to_string(),
        ],
        motion_variants: vec![
            "steady_flight".to_string(),
            "climb".to_string(),
            "turn".to_string(),
        ],
        notes: "Neutral placeholder card for a public-proxy airborne object.".to_string(),
    }
}

pub fn public_proxy_material_card() -> MaterialCard {
    let mut parameters = BTreeMap::new();
    parameters.insert("epsilon_r_mean".to_string(), "3.1".to_string());
    parameters.insert("epsilon_r_stddev".to_string(), "0.4".to_string());
    parameters.insert("loss_tangent_mean".to_string(), "0.018".to_string());
    parameters.insert("loss_tangent_stddev".to_string(), "0.006".to_string());

    MaterialCard {
        schema_ref: schema::MATERIAL_CARD.to_string(),
        id: "mat.public_proxy.skin_composite_v1".to_string(),
        display_name: "Public Proxy Composite Skin".to_string(),
        base_material: "composite_skin".to_string(),
        frequency_band_hz: vec![8_000_000_000, 12_000_000_000],
        parameter_model: ParameterModel {
            model_type: "bounded_distribution".to_string(),
            parameters,
        },
        uncertainty_model: UncertaintyModel {
            model_type: "lognormal".to_string(),
            bands: vec![
                UncertaintyBand {
                    name: "x_band".to_string(),
                    sigma_db: "1.5".to_string(),
                },
                UncertaintyBand {
                    name: "ku_band".to_string(),
                    sigma_db: "2.0".to_string(),
                },
            ],
        },
        notes: "Placeholder material assumptions for public-proxy radar research.".to_string(),
    }
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
        notes: "Hard-negative placeholders for robustness and false-alarm stress testing.".to_string(),
    }
}

pub fn public_proxy_scenario_manifest() -> ScenarioManifest {
    ScenarioManifest {
        schema_ref: schema::SCENARIO.to_string(),
        id: "scenario.public_proxy.clutter_v1".to_string(),
        display_name: "Public Proxy Clutter Scenario v1".to_string(),
        object_pack_refs: vec!["object-packs/public-proxy-v1/pack.manifest.json".to_string()],
        hard_negative_pack_refs: vec!["object-packs/hard-negatives/pack.manifest.json".to_string()],
        sensor_archetype_ref: "sensors/x_band_generic_airborne_scan_v1.yaml".to_string(),
        split_policy_ref: "tests/datasets/split_policy.json".to_string(),
        validation_tier: "draft".to_string(),
        environment: EnvironmentRecipe {
            clutter: "moderate".to_string(),
            weather: "light".to_string(),
            multipath: "enabled".to_string(),
            rfi: "monitored".to_string(),
            difficulty_tier: "baseline".to_string(),
        },
        notes: "Neutral public-proxy scenario scaffold; no exact truth claim.".to_string(),
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
        assert_eq!(card.schema_ref, schema::OBJECT_CARD);
        assert!(card.public_proxy);
    }

    #[test]
    fn fixture_scenario_parses() {
        let scenario: ScenarioManifest = from_yaml_str(include_str!(
            "../../../scenarios/public-proxy-clutter-v1/scenario.yaml"
        ))
        .expect("scenario fixture should parse");
        assert_eq!(scenario.schema_ref, schema::SCENARIO);
        assert_eq!(scenario.environment.difficulty_tier, "baseline");
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
        assert!(pack.confuser_traits.iter().any(|item| item == "weather_and_rfi"));
    }

    #[test]
    fn hard_negative_pack_is_neutral() {
        let pack = public_proxy_hard_negative_pack();
        assert!(pack.confuser_traits.iter().any(|item| item == "weather_and_rfi"));
        assert_eq!(pack.source_status, "public_proxy");
    }
}
