use std::fmt;

use echoforge_radar::BackendSelectionError;

#[derive(Debug)]
pub enum DatasetError {
    Io(std::io::Error),
    Csv(csv::Error),
    Json(serde_json::Error),
    Yaml(serde_yaml::Error),
    Core(echoforge_core::CoreError),
    Sig(echoforge_sig::SigError),
    Runtime(BackendSelectionError),
    Tensor(String),
    InvalidConfig(String),
    RefusingOutputPath(String),
}

impl fmt::Display for DatasetError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(err) => write!(f, "io error: {err}"),
            Self::Csv(err) => write!(f, "csv error: {err}"),
            Self::Json(err) => write!(f, "json error: {err}"),
            Self::Yaml(err) => write!(f, "yaml error: {err}"),
            Self::Core(err) => write!(f, "core model validation error: {err}"),
            Self::Sig(err) => write!(f, "tensor writer error: {err}"),
            Self::Runtime(err) => write!(f, "runtime selection error: {err}"),
            Self::Tensor(err) => write!(f, "tensor shape error: {err}"),
            Self::InvalidConfig(msg) => write!(f, "invalid Monte Carlo config: {msg}"),
            Self::RefusingOutputPath(msg) => write!(f, "refusing output path: {msg}"),
        }
    }
}

impl std::error::Error for DatasetError {}

impl From<std::io::Error> for DatasetError {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value)
    }
}

impl From<csv::Error> for DatasetError {
    fn from(value: csv::Error) -> Self {
        Self::Csv(value)
    }
}

impl From<serde_json::Error> for DatasetError {
    fn from(value: serde_json::Error) -> Self {
        Self::Json(value)
    }
}

impl From<serde_yaml::Error> for DatasetError {
    fn from(value: serde_yaml::Error) -> Self {
        Self::Yaml(value)
    }
}

impl From<echoforge_core::CoreError> for DatasetError {
    fn from(value: echoforge_core::CoreError) -> Self {
        Self::Core(value)
    }
}

impl From<echoforge_sig::SigError> for DatasetError {
    fn from(value: echoforge_sig::SigError) -> Self {
        Self::Sig(value)
    }
}

impl From<BackendSelectionError> for DatasetError {
    fn from(value: BackendSelectionError) -> Self {
        Self::Runtime(value)
    }
}
