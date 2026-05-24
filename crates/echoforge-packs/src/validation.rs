//! JSON-schema validation against the EchoForge schema set.
//!
//! Loads `schemas/*.schema.json` from the repo root, compiles each schema
//! once via [`jsonschema::Validator`], and caches the compiled validators
//! per `kind` discriminator. Card payloads are routed to the matching
//! schema by the `kind` field; missing-kind payloads are rejected.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use jsonschema::{JSONSchema, SchemaResolver, SchemaResolverError};
use serde_json::Value;
use url::Url;

use crate::error::PackError;

/// Custom schema resolver: maps `https://echoforge.local/schemas/<name>.schema.json`
/// to local files under `<repo>/schemas/`. The EchoForge schemas use
/// `$ref` against this host because the schemas were authored before a
/// loader existed; this resolver lets the loader honour those refs
/// without HTTP fetching.
struct LocalSchemaResolver {
    schemas_root: PathBuf,
}

impl SchemaResolver for LocalSchemaResolver {
    fn resolve(
        &self,
        _root_schema: &Value,
        url: &Url,
        _original_reference: &str,
    ) -> Result<Arc<Value>, SchemaResolverError> {
        if url.host_str() == Some("echoforge.local") {
            let path = url.path();
            if let Some(filename) = path.rsplit('/').next() {
                let local_path = self.schemas_root.join(filename);
                if local_path.exists() {
                    let text = std::fs::read_to_string(&local_path).map_err(|e| {
                        anyhow::anyhow!(
                            "local schema read failed at {}: {e}",
                            local_path.display()
                        )
                    })?;
                    let val: Value = serde_json::from_str(&text).map_err(|e| {
                        anyhow::anyhow!(
                            "local schema parse failed at {}: {e}",
                            local_path.display()
                        )
                    })?;
                    return Ok(Arc::new(val));
                }
            }
        }
        Err(anyhow::anyhow!("unresolvable external schema URL: {url}"))
    }
}

/// Compiled-validator cache keyed by card-`kind` discriminator (the
/// `const` value declared at the top of every EchoForge schema, e.g.
/// `"object_card"`, `"radar_platform_card"`, `"source_platform_card"`).
pub struct SchemaCatalog {
    schemas_root: PathBuf,
    validators: HashMap<String, Arc<JSONSchema>>,
}

impl SchemaCatalog {
    /// Load `<repo_root>/schemas/` and return a [`SchemaCatalog`] that
    /// lazy-compiles validators on first use per `kind`.
    pub fn from_repo(repo_root: &Path) -> Result<Self, PackError> {
        let schemas_root = repo_root.join("schemas");
        if !schemas_root.exists() {
            return Err(PackError::SchemaDirNotFound {
                path: schemas_root,
            });
        }
        Ok(Self {
            schemas_root,
            validators: HashMap::new(),
        })
    }

    /// Validate `payload` against the schema corresponding to its `kind`
    /// field. Returns `Ok(())` on validation pass; `Err(PackError::SchemaValidation)`
    /// when one or more JSON-schema rules fail.
    pub fn validate(&mut self, payload_path: &Path, payload: &Value) -> Result<(), PackError> {
        let kind = match payload.get("kind").and_then(Value::as_str) {
            Some(k) => k.to_string(),
            None => return Err(PackError::MissingKind { path: payload_path.to_path_buf() }),
        };

        let validator = self.validator_for(&kind, payload_path)?;
        // `JSONSchema::validate` returns errors with lifetimes tied to the
        // validator; collect into owned strings within this scope before
        // returning so the iterator does not outlive the borrow.
        let messages: Vec<String> = match validator.validate(payload) {
            Ok(()) => return Ok(()),
            Err(errors) => errors
                .take(10)
                .map(|e| format!("[{}] {}", e.instance_path, e))
                .collect(),
        };
        Err(PackError::SchemaValidation {
            path: payload_path.to_path_buf(),
            schema: format!("{}.schema.json", kind),
            messages: messages.join("; "),
        })
    }

    /// Look up (and lazy-compile) a validator for the given `kind`.
    fn validator_for(
        &mut self,
        kind: &str,
        payload_path: &Path,
    ) -> Result<Arc<JSONSchema>, PackError> {
        if let Some(v) = self.validators.get(kind) {
            return Ok(Arc::clone(v));
        }
        let schema_path = self.schemas_root.join(format!("{kind}.schema.json"));
        if !schema_path.exists() {
            return Err(PackError::UnknownKind {
                path: payload_path.to_path_buf(),
                kind: kind.to_string(),
            });
        }
        let text = std::fs::read_to_string(&schema_path).map_err(|e| PackError::Io {
            path: schema_path.clone(),
            source: e,
        })?;
        let schema_value: Value = serde_json::from_str(&text).map_err(|e| PackError::Json {
            path: schema_path.clone(),
            source: e,
        })?;
        let resolver = LocalSchemaResolver {
            schemas_root: self.schemas_root.clone(),
        };
        let compiled = JSONSchema::options()
            .with_resolver(resolver)
            .compile(&schema_value)
            .map_err(|err| PackError::SchemaValidation {
                path: schema_path.clone(),
                schema: format!("{kind}.schema.json"),
                messages: format!("schema compile failed: {err}"),
            })?;
        let arc = Arc::new(compiled);
        self.validators.insert(kind.to_string(), Arc::clone(&arc));
        Ok(arc)
    }

    /// Number of compiled validators currently cached.
    pub fn cached_validator_count(&self) -> usize {
        self.validators.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn repo_root() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .to_path_buf()
    }

    #[test]
    fn catalog_loads_from_repo() {
        let cat = SchemaCatalog::from_repo(&repo_root()).expect("schemas/ exists in repo");
        assert_eq!(cat.cached_validator_count(), 0);
    }

    #[test]
    fn validate_missing_kind_errors() {
        let mut cat = SchemaCatalog::from_repo(&repo_root()).expect("schemas/ exists");
        let payload = json!({"foo": 1});
        let err = cat
            .validate(&PathBuf::from("p.json"), &payload)
            .unwrap_err();
        assert!(matches!(err, PackError::MissingKind { .. }));
    }

    #[test]
    fn validate_unknown_kind_errors() {
        let mut cat = SchemaCatalog::from_repo(&repo_root()).expect("schemas/ exists");
        let payload = json!({"kind": "totally_not_a_real_schema"});
        let err = cat
            .validate(&PathBuf::from("p.json"), &payload)
            .unwrap_err();
        assert!(matches!(err, PackError::UnknownKind { .. }));
    }
}
