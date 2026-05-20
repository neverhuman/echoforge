//! Dataset and benchmark scaffold for EchoForge.
//!
//! This crate owns split policy, leakage checks, and benchmark pending entries.

pub mod benchmark;
pub mod campaign;
pub mod export;
pub(crate) mod guard;
pub mod leakage;
pub mod ml_pipelines;
pub mod ml_training;
pub mod monte_carlo;
pub(crate) mod rng;
pub mod scenarios;
pub mod split;

pub use benchmark::{benchmark_report_template, BenchmarkReportSkeleton};
pub use campaign::{
    run_monte_carlo_campaign, CampaignConfig, CampaignReport, DEFAULT_CAMPAIGN_OUTPUT,
    DEFAULT_CAMPAIGN_REQUEST_ID, NEUTRAL_CAMPAIGN_ID, OWA_DELTA_OBJECT_ID,
};
pub use leakage::{build_leakage_report, LeakageFinding, LeakageReport};
pub use ml_pipelines::{
    default_ml_pipeline_request, discover_repo_root, inspect_pipeline, list_pipelines,
    run_pipeline, run_suite, MlPipelineError, PipelineArtifact, PipelineGate, PipelineRunRequest,
    PipelineRunResult, PipelineSpec, PipelineSuiteResult, DEFAULT_DATA_ROOT, DEFAULT_OUT_ROOT,
    DEFAULT_VALIDATION_TIER, MAX_PIPELINE_WORKERS, MAX_SUITE_CONCURRENCY,
};
pub use ml_training::{
    run_ml_training_data, MlTrainingDataConfig, MlTrainingDataReport, PerTierMetrics,
    BEST_FINAL_OUTPUT, BEST_FINAL_POSITIVE_CLASS_IDS, BEST_FINAL_SCENARIO_ID,
    BEST_FINAL_SENSOR_IDS, DEFAULT_ML_TRAINING_DATASET_ID, DEFAULT_ML_TRAINING_OUTPUT,
};
pub use monte_carlo::{
    embedded_airspace_config, known_presets, run_monte_carlo_demo, AirspaceMonteCarloConfig,
    DatasetError, MonteCarloBenchmarkReport, MonteCarloDemoConfig, MonteCarloDemoReport,
    MonteCarloRuntimePolicy, StageTiming, DEFAULT_PRESET, DEFAULT_TARGET_LABEL,
};
pub use scenarios::{
    EnvironmentState, RadarSite, ScenarioLoadError, SensorConfig, SurveillanceScenario,
    TargetLaunchSite,
};
pub use split::{
    assign_split, default_public_proxy_split_policy, DatasetRecord, SplitKind, SplitPolicy,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_policy_fixture_parses() {
        let policy: SplitPolicy =
            serde_json::from_str(include_str!("../../../tests/datasets/split_policy.json"))
                .expect("split policy fixture should parse");
        assert_eq!(policy.policy_id, "dataset.split.public_proxy_v1");
        assert_eq!(policy.protected_keys.len(), 6);
    }

    #[test]
    fn leakage_fixture_detects_overlap() {
        let records: Vec<DatasetRecord> =
            serde_json::from_str(include_str!("../../../tests/datasets/leakage_input.json"))
                .expect("leakage fixture should parse");
        let report = build_leakage_report(&records, &default_public_proxy_split_policy());
        assert!(!report.is_clean());
        assert!(!report.findings.is_empty());
    }

    #[test]
    fn benchmark_template_mentions_schema_refs() {
        let template = benchmark_report_template();
        assert!(template.contains("schemas/benchmark_report.schema.json"));
        assert!(template.contains("schemas/dataset_card.schema.json"));
    }
}
