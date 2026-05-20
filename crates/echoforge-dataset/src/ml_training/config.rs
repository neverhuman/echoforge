//! Configuration structs and validation for the ML training dataset generator.

use std::path::PathBuf;

use echoforge_radar::BackendMode;
use crate::monte_carlo::DatasetError;

pub const DEFAULT_ML_TRAINING_DATASET_ID: &str = "shahed136-public-proxy-ml-training-v1";
pub const DEFAULT_ML_TRAINING_OUTPUT: &str =
    "outputs/training-data/shahed136-public-proxy-ml-training-v1";
pub(super) const NEUTRAL_OBJECT_ID: &str = "owa-delta-pusher-fixed-wing-public-proxy-v1";

#[derive(Debug, Clone)]
pub struct MlTrainingDataConfig {
    pub dataset: String,
    pub records: usize,
    pub positive_fraction: f64,
    pub time_window_s: f64,
    pub frame_rate_hz: f64,
    pub backend: BackendMode,
    pub workers: Option<usize>,
    pub seed: u64,
    pub generated_at: String,
    pub output_dir: PathBuf,
}

impl MlTrainingDataConfig {
    pub fn shahed_public_proxy_default() -> Self {
        Self {
            dataset: DEFAULT_ML_TRAINING_DATASET_ID.to_string(),
            records: 10_000,
            positive_fraction: 0.20,
            time_window_s: 90.0,
            frame_rate_hz: 2.0,
            backend: BackendMode::Auto,
            workers: Some(40),
            seed: 20_260_518_136_001,
            generated_at: "2026-05-18T00:00:00Z".to_string(),
            output_dir: PathBuf::from(DEFAULT_ML_TRAINING_OUTPUT),
        }
    }
}

pub(super) fn validate_config(config: &MlTrainingDataConfig) -> Result<(), DatasetError> {
    if config.dataset != DEFAULT_ML_TRAINING_DATASET_ID {
        return Err(DatasetError::InvalidConfig(format!(
            "unknown ML training dataset {}; expected {}",
            config.dataset, DEFAULT_ML_TRAINING_DATASET_ID
        )));
    }
    let checks: &[(bool, &str)] = &[
        (
            (1..=50_000).contains(&config.records),
            "record count must be in range 1..=50_000",
        ),
        (
            (0.01..=0.99).contains(&config.positive_fraction),
            "positive-fraction must be in range 0.01..=0.99",
        ),
        (
            config.time_window_s > 0.0 && config.frame_rate_hz > 0.0,
            "time-window-s and frame-rate-hz must each be positive",
        ),
        (
            !config.generated_at.trim().is_empty()
                && config.generated_at.contains('T')
                && config.generated_at.ends_with('Z'),
            "generated-at must be an RFC3339-like UTC timestamp ending in Z",
        ),
        (
            config.workers.map_or(true, |w| w > 0),
            "workers must be at least 1",
        ),
    ];
    for (ok, msg) in checks {
        if !ok {
            return Err(DatasetError::InvalidConfig(msg.to_string()));
        }
    }
    Ok(())
}

