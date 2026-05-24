use thiserror::Error;

#[derive(Debug, Error)]
pub enum CoreError {
    #[error("validation error: {0}")]
    Validation(String),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
}
