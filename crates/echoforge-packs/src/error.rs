use std::path::PathBuf;

use thiserror::Error;

/// Errors produced by the pack loader.
#[derive(Debug, Error)]
pub enum PackError {
    #[error("IO error reading {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("JSON parse error in {path}: {source}")]
    Json {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },

    #[error("YAML parse error in {path}: {source}")]
    Yaml {
        path: PathBuf,
        #[source]
        source: serde_yaml::Error,
    },

    #[error("Pack manifest at {path} is missing required field: {field}")]
    MissingField { path: PathBuf, field: String },

    #[error("Pack manifest at {path} has unrecognised schema_version: {actual}")]
    UnknownManifestVersion { path: PathBuf, actual: String },

    #[error("Card at {path} fails schema {schema}: {messages}")]
    SchemaValidation {
        path: PathBuf,
        schema: String,
        messages: String,
    },

    #[error("Card at {path} has no `kind` discriminator; cannot route to a schema")]
    MissingKind { path: PathBuf },

    #[error("Card at {path} declares unknown kind: {kind}")]
    UnknownKind { path: PathBuf, kind: String },

    #[error("Duplicate pack slug `{slug}` (already registered from {existing})")]
    DuplicatePack { slug: String, existing: PathBuf },

    #[error("Duplicate card slug `{slug}` in pack `{pack}` (already at {existing})")]
    DuplicateCard {
        pack: String,
        slug: String,
        existing: PathBuf,
    },

    #[error("Schema directory not found at {path}")]
    SchemaDirNotFound { path: PathBuf },

    #[error("Schema file not found: {schema}")]
    SchemaNotFound { schema: String },
}
