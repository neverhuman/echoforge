use std::path::PathBuf;

use thiserror::Error;

pub type Result<T> = std::result::Result<T, EchoSigError>;

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

