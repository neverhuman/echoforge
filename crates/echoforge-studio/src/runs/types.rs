use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum RunMode {
    Live,
    Replay,
    MonteCarlo,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RunConfig {
    pub scenario_id: String,
    pub scenario_label: String,
    pub mode: RunMode,
    pub seed: u64,
    pub scenario_hash: String,
    pub object_source_card: String,
    pub material_assumption_card: String,
    pub solver_chain_version: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RunArtifact {
    pub id: String,
    pub kind: String,
    pub label: String,
    pub ready: bool,
    pub download_path: String,
    pub requires_validation: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RunValidationSummary {
    pub tier: String,
    pub grade: String,
    pub export_gate_passed: bool,
    pub source_confidence: String,
    pub uncertainty_statement: String,
    pub known_limitations: Vec<String>,
    pub leakage_guard_status: String,
    pub reproducibility_metadata: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RunSummary {
    pub run_id: String,
    pub created_utc: String,
    pub status: String,
    pub config: RunConfig,
    pub validation: RunValidationSummary,
    pub artifacts: Vec<RunArtifact>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RunQueueSummary {
    pub total: usize,
    pub active: usize,
    pub archived: usize,
    pub export_ready: usize,
    pub queued: usize,
    pub newest_created_utc: Option<String>,
    pub validation_tiers: Vec<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct RunArchiveRequest {
    #[serde(default)]
    pub reason: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct RunDuplicateRequest {
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default)]
    pub mode: Option<RunMode>,
    #[serde(default)]
    pub note: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct MonteCarloRunRequest {
    pub scenario_id: String,
    pub source_pack: String,
    pub object_pack: String,
    #[serde(default)]
    pub hard_negatives: Vec<String>,
    pub weather_profile: String,
    pub detector_pipeline: String,
    pub seed: u64,
    pub run_count: u32,
    pub workers: u16,
    pub max_concurrent: u16,
    pub smoke: bool,
    pub validation_target: String,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum ReplayMode {
    ExactSeed,
    ModifiedParameters,
    MonteCarloExpansion,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct RunReplayRequest {
    #[serde(default = "default_replay_mode")]
    pub mode: ReplayMode,
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default)]
    pub monte_carlo_count: Option<u32>,
}

fn default_replay_mode() -> ReplayMode {
    ReplayMode::ExactSeed
}

#[derive(Debug, Deserialize)]
pub struct DownloadQuery {
    #[serde(default = "default_download_kind")]
    pub kind: String,
}

fn default_download_kind() -> String {
    "bundle".to_string()
}
