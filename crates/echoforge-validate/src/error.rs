use thiserror::Error;

#[derive(Debug, Error)]
pub enum ValidateError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("schema error: {0}")]
    Schema(String),
    #[error("bad args: {0}")]
    BadArgs(String),
    #[error("validation failed: {0}")]
    Failed(String),
}
