use serde::Serialize;

use super::types::JobComposeRequest;

pub(super) const DEFAULT_PIPELINE_ID: &str = "physics_cfar_track_fusion_v1";
pub(super) const DEFAULT_SUITE_ID: &str = "evidence-ladder-v1";
const DEFAULT_DATA_ROOT: &str = "outputs/training-data/best-final-scenario-v1";
const DEFAULT_OUT_ROOT: &str = "outputs/ml-pipelines";
const DEFAULT_VALIDATION_TIER: &str = "evidence_ladder_v1";

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct JobOption {
    pub id: &'static str,
    pub label: &'static str,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct JobDefaultsResponse {
    pub request: JobComposeRequest,
    pub pipelines: Vec<JobOption>,
    pub suites: Vec<JobOption>,
    pub validation_tiers: Vec<String>,
}

impl Default for JobComposeRequest {
    fn default() -> Self {
        Self {
            job_type: default_job_type(),
            selection: default_selection(),
            pipeline_id: default_pipeline_id(),
            suite_id: default_suite_id(),
            data_root: default_data_root(),
            out_root: default_out_root(),
            workers_per_pipeline: default_workers_per_pipeline(),
            max_concurrent: default_max_concurrent(),
            seed: default_seed(),
            smoke: true,
            validation_tier: default_validation_tier(),
        }
    }
}

pub fn job_defaults_response() -> JobDefaultsResponse {
    JobDefaultsResponse {
        request: JobComposeRequest::default(),
        pipelines: vec![
            JobOption {
                id: "physics_cfar_track_fusion_v1",
                label: "Physics CFAR Track Fusion",
            },
            JobOption {
                id: "tensor_microdoppler_fusion_v1",
                label: "Tensor Micro-Doppler Fusion",
            },
            JobOption {
                id: "raw_iq_ssl_research_v1",
                label: "Raw IQ SSL Research",
            },
        ],
        suites: vec![JobOption {
            id: DEFAULT_SUITE_ID,
            label: "Evidence Ladder Suite",
        }],
        validation_tiers: vec![DEFAULT_VALIDATION_TIER.to_string()],
    }
}

pub(super) fn default_job_type() -> String {
    "ml_processing".to_string()
}

pub(super) fn default_selection() -> String {
    "pipeline".to_string()
}

pub(super) fn default_pipeline_id() -> String {
    DEFAULT_PIPELINE_ID.to_string()
}

pub(super) fn default_suite_id() -> String {
    DEFAULT_SUITE_ID.to_string()
}

pub(super) fn default_data_root() -> String {
    DEFAULT_DATA_ROOT.to_string()
}

pub(super) fn default_out_root() -> String {
    DEFAULT_OUT_ROOT.to_string()
}

pub(super) fn default_workers_per_pipeline() -> usize {
    20
}

pub(super) fn default_max_concurrent() -> usize {
    3
}

pub(super) fn default_seed() -> u64 {
    20_260_520_390_001
}

pub(super) fn default_validation_tier() -> String {
    DEFAULT_VALIDATION_TIER.to_string()
}
