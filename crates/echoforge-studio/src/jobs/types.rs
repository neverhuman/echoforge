use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct JobComposeRequest {
    #[serde(default = "super::defaults::default_job_type")]
    pub job_type: String,
    #[serde(default = "super::defaults::default_selection")]
    pub selection: String,
    #[serde(default = "super::defaults::default_pipeline_id")]
    pub pipeline_id: String,
    #[serde(default = "super::defaults::default_suite_id")]
    pub suite_id: String,
    #[serde(default = "super::defaults::default_data_root")]
    pub data_root: String,
    #[serde(default = "super::defaults::default_out_root")]
    pub out_root: String,
    #[serde(default = "super::defaults::default_workers_per_pipeline")]
    pub workers_per_pipeline: usize,
    #[serde(default = "super::defaults::default_max_concurrent")]
    pub max_concurrent: usize,
    #[serde(default = "super::defaults::default_seed")]
    pub seed: u64,
    #[serde(default)]
    pub smoke: bool,
    #[serde(default = "super::defaults::default_validation_tier")]
    pub validation_tier: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct JobArtifact {
    pub id: String,
    pub kind: String,
    pub path: String,
    pub ready: bool,
    #[serde(default)]
    pub incomplete: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MetricSlice {
    pub label: String,
    pub auc: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CalibrationBin {
    pub bin: u32,
    pub lower: f64,
    pub upper: f64,
    pub count: f64,
    pub mean_score: f64,
    pub positive_rate: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ValidationGate {
    pub gate: String,
    pub status: String,
    pub detail: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PipelineSummary {
    pub pipeline_id: String,
    pub status: String,
    pub output_dir: String,
    pub roc_auc: f64,
    pub pr_auc: f64,
    pub missing_input_kind: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct JobResults {
    pub primary_pipeline_id: String,
    pub suite_id: Option<String>,
    pub auc_table_path: String,
    pub roc_points: Vec<[f64; 2]>,
    pub pr_auc: f64,
    pub phase_auc: Vec<MetricSlice>,
    pub sensor_holdout_auc: Vec<MetricSlice>,
    pub class_holdout_auc: Vec<MetricSlice>,
    pub hard_negative_breakdown: Vec<MetricSlice>,
    pub leakage_gates: Vec<ValidationGate>,
    pub calibration_bins: Vec<CalibrationBin>,
    pub export_readiness: String,
    pub pipeline_summaries: Vec<PipelineSummary>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct JobSummary {
    pub job_id: String,
    pub created_utc: String,
    pub status: String,
    pub progress_percent: u8,
    pub message: String,
    pub request: JobComposeRequest,
    pub record_count: usize,
    pub output_dir: String,
    pub artifacts: Vec<JobArtifact>,
    pub results: JobResults,
}

#[derive(Debug, Clone)]
pub(super) struct JobExecutionPayload {
    pub message: String,
    pub record_count: usize,
    pub output_dir: String,
    pub artifacts: Vec<JobArtifact>,
    pub results: JobResults,
}
