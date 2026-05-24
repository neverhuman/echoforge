use std::io;
use std::path::PathBuf;

use thiserror::Error;

/// Result alias for the prior `EchoSigError` API.
pub type Result<T> = std::result::Result<T, EchoSigError>;

/// Result alias for the new bundle I/O API.
pub type SigResult<T> = std::result::Result<T, SigError>;

/// Prior error preserved for the original `artifact::EchoSigArtifactBundle`
/// pathway. New code should use [`SigError`].
#[derive(Debug, Error)]
pub enum EchoSigError {
    #[error("io error at {path:?}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("json error at {path:?}: {source}")]
    Json {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },
    #[error("yaml error at {path:?}: {source}")]
    Yaml {
        path: PathBuf,
        #[source]
        source: serde_yaml::Error,
    },
    #[error("missing required file: {0:?}")]
    MissingFile(PathBuf),
}

/// Error type for the new EchoSig bundle writer/reader.
#[derive(Debug, Error)]
pub enum SigError {
    #[error("io error: {0}")]
    Io(#[from] io::Error),

    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("core error: {0}")]
    Core(#[from] echoforge_core::CoreError),

    #[error("schema mismatch: {0}")]
    SchemaMismatch(String),

    #[error("missing tensor: {0}")]
    MissingTensor(String),

    #[error("axis mismatch: expected {expected} dims, got {actual}")]
    AxisMismatch { expected: usize, actual: usize },

    #[error("undeclared bundle path: {0}")]
    UndeclaredPath(String),

    #[error("invalid tensor data: {0}")]
    InvalidTensor(String),
}
