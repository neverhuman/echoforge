use thiserror::Error;

/// Errors that can arise while running the hard-negative mining loop.
///
/// All variants carry enough context to identify the file or class id that
/// triggered the failure so that loop-driver receipts can quote the source
/// of the problem without re-deriving it from a backtrace.
#[derive(Debug, Error)]
pub enum MiningError {
    #[error("io error while reading {path}: {source}")]
    Io {
        path: String,
        #[source]
        source: std::io::Error,
    },

    #[error("failed to parse JSON from {path}: {source}")]
    Json {
        path: String,
        #[source]
        source: serde_json::Error,
    },

    #[error("unknown base class id: {class_id}")]
    UnknownBaseClass { class_id: String },

    #[error("invalid fp_threshold {threshold}: must be a finite probability in (0.0, 1.0]")]
    InvalidThreshold { threshold: f64 },

    #[error("campaign root does not exist: {path}")]
    MissingCampaignRoot { path: String },

    #[error("no model_eval_*.json files found under {path}")]
    NoModelEvalReports { path: String },

    #[error("failed to write output {path}: {source}")]
    Output {
        path: String,
        #[source]
        source: std::io::Error,
    },
}
