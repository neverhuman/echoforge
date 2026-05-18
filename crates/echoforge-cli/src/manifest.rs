use crate::core::{Health, ManifestSummary, StatusCheck, StatusSummary};
use serde_json::{Map, Value};
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone)]
pub struct EchoSigManifest {
    pub path: PathBuf,
    pub raw: Value,
}

impl EchoSigManifest {
    pub fn load(path: impl AsRef<Path>) -> Self {
        let path = path.as_ref().to_path_buf();
        let raw = fs::read_to_string(&path)
            .ok()
            .and_then(|text| serde_json::from_str::<Value>(&text).ok())
            .unwrap_or(Value::Null);

        Self { path, raw }
    }

    pub fn from_value(path: impl AsRef<Path>, raw: Value) -> Self {
        Self {
            path: path.as_ref().to_path_buf(),
            raw,
        }
    }

    fn object(&self) -> Option<&Map<String, Value>> {
        self.raw.as_object()
    }

    fn string_field(&self, keys: &[&str]) -> Option<String> {
        let object = self.object()?;
        for key in keys {
            if let Some(value) = object.get(*key).and_then(Value::as_str) {
                return Some(value.to_string());
            }
        }
        None
    }

    pub fn artifact(&self) -> Option<String> {
        self.string_field(&["artifact", "artifact_id", "name"])
    }

    pub fn validation_tier(&self) -> Option<String> {
        self.string_field(&["validation_tier", "validationTier"])
    }

    pub fn inspect(&self) -> ManifestSummary {
        let mut status = StatusSummary::new("EchoSig manifest");
        let is_object = self.raw.is_object();

        if !is_object {
            status.push(StatusCheck::new(
                "structure",
                Health::Fail,
                "manifest is not a JSON object",
            ));
            return ManifestSummary {
                path: self.path.clone(),
                artifact: None,
                validation_tier: None,
                status,
            };
        }

        let artifact = self.artifact();
        status.push(StatusCheck::new(
            "artifact",
            if artifact.is_some() {
                Health::Ok
            } else {
                Health::Warn
            },
            artifact
                .as_deref()
                .unwrap_or("missing artifact/artifact_id/name field"),
        ));

        let validation_tier = self.validation_tier();
        status.push(StatusCheck::new(
            "validation_tier",
            if validation_tier.is_some() {
                Health::Ok
            } else {
                Health::Warn
            },
            validation_tier
                .as_deref()
                .unwrap_or("missing validation_tier field"),
        ));

        status.push(StatusCheck::new(
            "provenance",
            if self.object().and_then(|o| o.get("provenance")).is_some()
                || self.object().and_then(|o| o.get("provenance_path")).is_some()
            {
                Health::Ok
            } else {
                Health::Warn
            },
            "provenance reference not found",
        ));

        status.push(StatusCheck::new(
            "license",
            if self.object().and_then(|o| o.get("license")).is_some()
                || self.object().and_then(|o| o.get("license_path")).is_some()
            {
                Health::Ok
            } else {
                Health::Warn
            },
            "license reference not found",
        ));

        ManifestSummary {
            path: self.path.clone(),
            artifact,
            validation_tier,
            status,
        }
    }
}

