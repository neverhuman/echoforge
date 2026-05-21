//! Configuration structs and validation for the ML training dataset generator.

use std::path::PathBuf;

use crate::monte_carlo::DatasetError;
use echoforge_radar::BackendMode;

pub const DEFAULT_ML_TRAINING_DATASET_ID: &str = "shahed136-public-proxy-ml-training-v1";
pub const DEFAULT_ML_TRAINING_OUTPUT: &str =
    "outputs/training-data/shahed136-public-proxy-ml-training-v1";
pub const BEST_FINAL_SCENARIO_ID: &str = "best-final-scenario-v1";
pub const BEST_FINAL_OUTPUT: &str = "outputs/training-data/best-final-scenario-v1";
pub(super) const NEUTRAL_OBJECT_ID: &str = "owa-delta-pusher-fixed-wing-public-proxy-v1";

pub const BEST_FINAL_SENSOR_IDS: [&str; 3] = [
    "saab-giraffe-1x",
    "rtx-kurfs",
    "hensoldt-spexer-2000-3d-mkiii",
];

pub const BEST_FINAL_POSITIVE_CLASS_IDS: [&str; 3] =
    ["shahed-136-geran-2", "shahed-131-geran-1", "mohajer-6"];

#[derive(Debug, Clone)]
pub struct MlTrainingDataConfig {
    pub dataset: String,
    pub records: usize,
    pub positive_fraction: f64,
    pub sensor_ids: Vec<String>,
    pub positive_class_ids: Vec<String>,
    pub positives_per_class_per_sensor: Option<usize>,
    pub negatives_per_sensor: Option<usize>,
    pub phase_targets: Vec<String>,
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
            sensor_ids: vec!["generic-x-band-public-proxy".to_string()],
            positive_class_ids: vec![NEUTRAL_OBJECT_ID.to_string()],
            positives_per_class_per_sensor: None,
            negatives_per_sensor: None,
            phase_targets: vec![
                "early_takeoff".to_string(),
                "mid_ramp".to_string(),
                "cruise_altitude".to_string(),
            ],
            time_window_s: 90.0,
            frame_rate_hz: 2.0,
            backend: BackendMode::Auto,
            workers: Some(40),
            seed: 20_260_518_136_001,
            generated_at: "2026-05-18T00:00:00Z".to_string(),
            output_dir: PathBuf::from(DEFAULT_ML_TRAINING_OUTPUT),
        }
    }

    pub fn best_final_default() -> Self {
        Self {
            dataset: BEST_FINAL_SCENARIO_ID.to_string(),
            records: 3_900,
            positive_fraction: 900.0 / 3_900.0,
            sensor_ids: BEST_FINAL_SENSOR_IDS
                .iter()
                .map(|sensor| (*sensor).to_string())
                .collect(),
            positive_class_ids: BEST_FINAL_POSITIVE_CLASS_IDS
                .iter()
                .map(|class_id| (*class_id).to_string())
                .collect(),
            positives_per_class_per_sensor: Some(100),
            negatives_per_sensor: Some(1_000),
            phase_targets: vec![
                "early_takeoff".to_string(),
                "mid_ramp".to_string(),
                "cruise_altitude".to_string(),
            ],
            time_window_s: 90.0,
            frame_rate_hz: 2.0,
            backend: BackendMode::Auto,
            workers: Some(40),
            seed: 20_260_520_390_001,
            generated_at: "2026-05-20T00:00:00Z".to_string(),
            output_dir: PathBuf::from(BEST_FINAL_OUTPUT),
        }
    }

    pub fn best_final_smoke() -> Self {
        let mut config = Self::best_final_default();
        config.records = 30;
        config.positive_fraction = 18.0 / 30.0;
        config.positives_per_class_per_sensor = Some(2);
        config.negatives_per_sensor = Some(4);
        config.time_window_s = 6.0;
        config.frame_rate_hz = 2.0;
        config.backend = BackendMode::Cpu;
        config.workers = Some(4);
        config.output_dir = PathBuf::from("outputs/training-data/best-final-scenario-v1-smoke");
        config
    }
}

pub(super) fn validate_config(config: &MlTrainingDataConfig) -> Result<(), DatasetError> {
    if config.dataset != DEFAULT_ML_TRAINING_DATASET_ID && config.dataset != BEST_FINAL_SCENARIO_ID
    {
        return Err(DatasetError::InvalidConfig(format!(
            "unknown ML training dataset {}; expected {} or {}",
            config.dataset, DEFAULT_ML_TRAINING_DATASET_ID, BEST_FINAL_SCENARIO_ID
        )));
    }
    if config.sensor_ids.is_empty() {
        return Err(DatasetError::InvalidConfig(
            "at least one sensor id is required".to_string(),
        ));
    }
    if config.positive_class_ids.is_empty() {
        return Err(DatasetError::InvalidConfig(
            "at least one positive class id is required".to_string(),
        ));
    }
    if config.phase_targets.is_empty() {
        return Err(DatasetError::InvalidConfig(
            "at least one phase target is required".to_string(),
        ));
    }
    if config.positives_per_class_per_sensor.is_some() != config.negatives_per_sensor.is_some() {
        return Err(DatasetError::InvalidConfig(
            "positives-per-class-per-sensor and negatives-per-sensor must be set together"
                .to_string(),
        ));
    }
    if let (Some(pos), Some(neg)) = (
        config.positives_per_class_per_sensor,
        config.negatives_per_sensor,
    ) {
        let expected = config.sensor_ids.len() * (config.positive_class_ids.len() * pos + neg);
        if expected != config.records {
            return Err(DatasetError::InvalidConfig(format!(
                "exact sensor/class count resolves to {expected} records, not {}",
                config.records
            )));
        }
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
            config.workers.is_none_or(|w| w > 0),
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
