//! Pack-manifest deserialisation.
//!
//! Supports both v1 (prior `object_pack_manifest.schema.json` /
//! `hard_negative_pack.schema.json`) and v2 (`pack_v2.schema.json`) shapes.
//! The discriminator is the top-level `schema_version` field plus the
//! presence of a typed `card_refs` map (v2) versus per-kind ref arrays (v1).

use std::collections::BTreeMap;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::error::PackError;

/// Detected pack manifest schema version. The loader routes v1 manifests
/// through the prior reader (which collects per-kind ref arrays from
/// fixed keys) and v2 manifests through the typed `card_refs` map.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ManifestVersion {
    V1Prior,
    V2Unified,
}

/// In-memory pack manifest. Holds the pack-level fields and a normalised
/// `card_refs` map (kind -> Vec<relative path>) suitable for downstream
/// discovery iteration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PackManifest {
    pub id: String,
    pub display_name: String,
    pub purpose: String,
    pub validation_tier: String,
    pub source_status: String,
    pub manifest_version: ManifestVersion,
    /// Map from `kind` discriminator (e.g. `"object_card"`,
    /// `"radar_platform_card"`) to relative card-file paths inside the
    /// pack directory.
    pub card_refs: BTreeMap<String, Vec<String>>,
    /// Optional pack-level metadata (license, provenance, notes) carried
    /// through verbatim for inspection / `ef pack show`.
    #[serde(default)]
    pub extras: serde_json::Map<String, Value>,
}

impl PackManifest {
    /// Parse a manifest from the raw JSON value at `path`, auto-detecting
    /// v1 vs v2.
    pub fn from_value(path: &PathBuf, value: &Value) -> Result<Self, PackError> {
        let obj = match value.as_object() {
            Some(o) => o,
            None => return Err(PackError::MissingField {
                path: path.clone(),
                field: "<root object>".into(),
            }),
        };

        let id = require_string(obj, "id", path)?;
        let display_name = match obj.get("display_name").and_then(Value::as_str) {
            Some(s) => s.to_string(),
            None => id.clone(),
        };
        let purpose = match obj.get("purpose").and_then(Value::as_str) {
            Some(s) => s.to_string(),
            None => String::from("(no purpose declared)"),
        };

        let manifest_version = if obj.get("schema_version").and_then(Value::as_str)
            == Some("2.0.0")
            || obj.contains_key("card_refs")
        {
            ManifestVersion::V2Unified
        } else {
            ManifestVersion::V1Prior
        };

        let validation_tier = match obj.get("validation_tier").and_then(Value::as_str) {
            Some(s) => s.to_string(),
            None => String::from("unvalidated"),
        };
        let source_status = match obj.get("source_status").and_then(Value::as_str) {
            Some(s) => s.to_string(),
            None => if obj.get("public_proxy") == Some(&Value::Bool(true)) {
                String::from("public_proxy")
            } else {
                String::from("unknown")
            },
        };

        let card_refs = match manifest_version {
            ManifestVersion::V2Unified => parse_v2_card_refs(obj),
            ManifestVersion::V1Prior => parse_v1_card_refs(obj),
        };

        let mut extras = serde_json::Map::new();
        for (k, v) in obj {
            if matches!(
                k.as_str(),
                "id"
                    | "display_name"
                    | "purpose"
                    | "schema_version"
                    | "validation_tier"
                    | "source_status"
                    | "card_refs"
                    | "object_card_refs"
                    | "material_card_refs"
                    | "object_card_slugs"
                    | "material_card_slugs"
                    | "card_slugs"
                    | "kind"
            ) {
                continue;
            }
            extras.insert(k.clone(), v.clone());
        }

        Ok(Self {
            id,
            display_name,
            purpose,
            validation_tier,
            source_status,
            manifest_version,
            card_refs,
            extras,
        })
    }

    /// Convenience helper: total card count across all kinds.
    pub fn total_card_count(&self) -> usize {
        self.card_refs.values().map(|v| v.len()).sum()
    }
}

fn require_string(
    obj: &serde_json::Map<String, Value>,
    field: &str,
    path: &PathBuf,
) -> Result<String, PackError> {
    match obj.get(field).and_then(Value::as_str) {
        Some(s) => Ok(s.to_string()),
        None => Err(PackError::MissingField {
            path: path.clone(),
            field: field.into(),
        }),
    }
}

fn parse_v2_card_refs(obj: &serde_json::Map<String, Value>) -> BTreeMap<String, Vec<String>> {
    let mut out = BTreeMap::new();
    if let Some(refs) = obj.get("card_refs").and_then(Value::as_object) {
        for (kind, value) in refs {
            if let Some(arr) = value.as_array() {
                let paths: Vec<String> = arr
                    .iter()
                    .filter_map(|v| v.as_str().map(str::to_string))
                    .collect();
                if !paths.is_empty() {
                    out.insert(kind.clone(), paths);
                }
            }
        }
    }
    out
}

fn parse_v1_card_refs(obj: &serde_json::Map<String, Value>) -> BTreeMap<String, Vec<String>> {
    let mut out = BTreeMap::new();
    let map = [
        ("object_card_refs", "object_card"),
        ("material_card_refs", "material_card"),
        ("sensor_archetype_refs", "sensor_archetype"),
        ("scenario_refs", "scenario"),
        ("mesh_manifest_refs", "mesh_manifest"),
        ("echosig_manifest_refs", "echosig_manifest"),
    ];
    for (prior_key, kind) in map {
        if let Some(arr) = obj.get(prior_key).and_then(Value::as_array) {
            let paths: Vec<String> = arr
                .iter()
                .filter_map(|v| v.as_str().map(str::to_string))
                .collect();
            if !paths.is_empty() {
                out.insert(kind.to_string(), paths);
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn v1_manifest_classified() {
        let v = json!({
            "id": "pack.public-proxy-v1",
            "object_card_refs": ["object_card.yaml"],
            "material_card_refs": ["material_card.yaml"],
        });
        let m = PackManifest::from_value(&PathBuf::from("manifest.json"), &v).unwrap();
        assert_eq!(m.manifest_version, ManifestVersion::V1Prior);
        assert_eq!(m.card_refs.get("object_card").map(Vec::len), Some(1));
        assert_eq!(m.card_refs.get("material_card").map(Vec::len), Some(1));
        assert_eq!(m.total_card_count(), 2);
    }

    #[test]
    fn v2_manifest_classified() {
        let v = json!({
            "id": "pack.radar-platforms-v1",
            "schema_version": "2.0.0",
            "card_refs": {
                "radar_platform_card": ["saab.yaml", "rtx.yaml"],
                "object_card": ["a.yaml"]
            }
        });
        let m = PackManifest::from_value(&PathBuf::from("manifest.json"), &v).unwrap();
        assert_eq!(m.manifest_version, ManifestVersion::V2Unified);
        assert_eq!(m.total_card_count(), 3);
        assert_eq!(
            m.card_refs.get("radar_platform_card").map(Vec::len),
            Some(2)
        );
    }

    #[test]
    fn missing_id_errors() {
        let v = json!({"display_name": "no-id-pack"});
        let err = PackManifest::from_value(&PathBuf::from("m.json"), &v).unwrap_err();
        assert!(matches!(err, PackError::MissingField { field, .. } if field == "id"));
    }

    #[test]
    fn extras_preserve_unknown_fields() {
        let v = json!({
            "id": "pack.x",
            "notes": ["hello"],
            "confuser_traits": ["a", "b"],
        });
        let m = PackManifest::from_value(&PathBuf::from("m.json"), &v).unwrap();
        assert!(m.extras.contains_key("notes"));
        assert!(m.extras.contains_key("confuser_traits"));
    }
}
