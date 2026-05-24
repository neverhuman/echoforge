use std::io::{self, IsTerminal};
use std::time::Instant;

use crate::export::write_json_pretty;
use crate::monte_carlo::DatasetError;

mod detectors;
mod plan;
mod reporting;
mod reporting_models;
mod rng;
mod simulation;
mod types;
mod workers;

// Re-export types needed by types.rs (referenced in CampaignConfig::shahed_public_proxy_default)
pub use detectors::StreamingDetector;
pub use types::{
    CalibrationBin, CampaignBucket, CampaignClass, CampaignConfig, CampaignRecordPlan,
    CampaignReport, ClassBalanceReport, CurvePoint, DetectionState, FirstTriggerEvent,
    FrameFeature, FrameLabel, ModelEvaluationReport,
};

use crate::guard::{begin_generation_run, guard_output_dir};

use plan::{build_campaign_plan, validate_campaign_config};
use reporting::{
    build_campaign_benchmark_report, build_class_balance, build_runtime_report,
    campaign_dataset_card, campaign_guardrails, write_model_artifacts, write_root_csvs,
};
use types::CampaignManifest;
use workers::run_campaign_workers;

pub const DEFAULT_CAMPAIGN_REQUEST_ID: &str = "shahed136-public-proxy-early-detection-v1";
pub const NEUTRAL_CAMPAIGN_ID: &str = "owa-delta-pusher-public-proxy-early-detection-v1";
pub const OWA_DELTA_OBJECT_ID: &str = "owa-delta-pusher-fixed-wing-public-proxy-v1";
pub const DEFAULT_CAMPAIGN_OUTPUT: &str =
    "outputs/campaigns/shahed136-public-proxy-early-detection-v1";
pub(super) const SOURCE_DOSSIER_REF: &str = "object-packs/public-proxy-v1/source_dossier.yaml";

pub fn run_monte_carlo_campaign(config: CampaignConfig) -> Result<CampaignReport, DatasetError> {
    validate_campaign_config(&config)?;
    guard_output_dir(&config.output_dir)?;
    let mut ctx = begin_generation_run(
        &config.output_dir,
        config.backend,
        config.workers,
        config.records,
        config.time_window_s,
        config.frame_rate_hz,
    )?;
    let progress_enabled = config.progress && io::stderr().is_terminal();
    let plan_start = Instant::now();
    let record_plans = build_campaign_plan(&config)?;
    ctx.push_timing("campaign_plan", plan_start);

    let generation_start = Instant::now();
    let mut outputs = run_campaign_workers(
        &config,
        &ctx.runtime,
        &record_plans,
        ctx.frame_count,
        ctx.worker_count,
        progress_enabled,
    )?;
    outputs.sort_by_key(|output| output.summary.record_index);
    ctx.push_timing("record_generation", generation_start);
    let total_outputs = outputs.len();

    let post_start = Instant::now();
    let summaries = outputs
        .iter()
        .map(|output| output.summary.clone())
        .collect::<Vec<_>>();
    let all_events = outputs
        .iter()
        .flat_map(|output| output.events.iter().cloned())
        .collect::<Vec<_>>();
    let all_predictions = outputs
        .iter()
        .flat_map(|output| output.predictions.iter().cloned())
        .collect::<Vec<_>>();

    write_root_csvs(
        &config.output_dir,
        &summaries,
        &all_events,
        &all_predictions,
    )?;
    let class_balance = build_class_balance(&config, &summaries);
    write_json_pretty(
        &config.output_dir.join("class_balance.json"),
        &class_balance,
    )?;

    let model_reports =
        write_model_artifacts(&config.output_dir, &summaries, config.trigger_confidence)?;
    let runtime_report = build_runtime_report(
        &config,
        &ctx.runtime,
        ctx.worker_count,
        progress_enabled,
        &ctx.stage_timings,
        ctx.overall_start.elapsed(),
    );
    write_json_pretty(
        &config.output_dir.join("runtime_report.json"),
        &runtime_report,
    )?;

    let benchmark_report =
        build_campaign_benchmark_report(&config, &runtime_report, &class_balance, &model_reports);
    write_json_pretty(
        &config.output_dir.join("benchmark_report.json"),
        &benchmark_report,
    )?;

    let dataset_card = campaign_dataset_card(&config, &class_balance)?;
    write_json_pretty(&config.output_dir.join("dataset_card.json"), &dataset_card)?;

    let manifest = CampaignManifest {
        manifest_version: "1".to_string(),
        campaign_request_id: config.campaign.clone(),
        neutral_campaign_id: NEUTRAL_CAMPAIGN_ID.to_string(),
        generated_at: config.generated_at.clone(),
        root_seed: config.seed,
        records: summaries.clone(),
        frame_count: ctx.frame_count,
        frame_rate_hz: config.frame_rate_hz,
        time_window_s: config.time_window_s,
        class_balance: class_balance.clone(),
        runtime_report_path: "runtime_report.json".to_string(),
        benchmark_report_path: "benchmark_report.json".to_string(),
        dataset_card_path: "dataset_card.json".to_string(),
        source_dossier_ref: SOURCE_DOSSIER_REF.to_string(),
        guardrails: campaign_guardrails(),
    };
    write_json_pretty(&config.output_dir.join("campaign_manifest.json"), &manifest)?;
    ctx.push_timing("postprocess", post_start);

    Ok(CampaignReport {
        output_dir: config.output_dir.clone(),
        campaign_request_id: config.campaign,
        neutral_campaign_id: NEUTRAL_CAMPAIGN_ID.to_string(),
        records: total_outputs,
        shahed_positive_records: class_balance.shahed_positive_records,
        worker_count: ctx.worker_count,
        runtime: ctx.runtime,
        progress_enabled,
        campaign_manifest_path: config.output_dir.join("campaign_manifest.json"),
        class_balance_path: config.output_dir.join("class_balance.json"),
        dataset_card_path: config.output_dir.join("dataset_card.json"),
        runtime_report_path: config.output_dir.join("runtime_report.json"),
        benchmark_report_path: config.output_dir.join("benchmark_report.json"),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::campaign::plan::{build_campaign_plan, campaign_classes};
    use crate::guard::compute_worker_count;
    use echoforge_radar::{BackendMode, BackendSignals, RuntimePlan};

    #[cfg(test)]
    #[derive(Debug, Clone, serde::Deserialize)]
    struct SourceDossier {
        public_proxy_id: String,
        aliases: Vec<SourceAlias>,
        proxy_envelope: SourceProxyEnvelope,
    }

    #[cfg(test)]
    #[derive(Debug, Clone, serde::Deserialize)]
    struct SourceAlias {
        alias: String,
        policy: String,
    }

    #[cfg(test)]
    #[derive(Debug, Clone, serde::Deserialize)]
    struct SourceProxyEnvelope {
        dimensions_m: SourceDimensions,
        rcs_dbsm_proxy: [f64; 2],
        cruise_speed_mps: [f64; 2],
    }

    #[cfg(test)]
    #[derive(Debug, Clone, serde::Deserialize)]
    struct SourceDimensions {
        length: [f64; 2],
        wingspan: [f64; 2],
        height: [f64; 2],
    }

    fn parse_source_dossier(input: &str) -> Result<SourceDossier, DatasetError> {
        serde_yaml::from_str(input).map_err(DatasetError::Yaml)
    }

    #[test]
    fn source_dossier_priors_parse_and_aliases_are_dossier_only() {
        let dossier = parse_source_dossier(include_str!(
            "../../../../object-packs/public-proxy-v1/source_dossier.yaml"
        ))
        .expect("source dossier parses");
        assert_eq!(dossier.public_proxy_id, OWA_DELTA_OBJECT_ID);
        assert_eq!(dossier.proxy_envelope.dimensions_m.length, [3.3, 3.7]);
        assert_eq!(dossier.proxy_envelope.dimensions_m.wingspan, [2.3, 2.7]);
        assert_eq!(dossier.proxy_envelope.dimensions_m.height, [0.35, 0.75]);
        assert_eq!(dossier.proxy_envelope.rcs_dbsm_proxy, [-16.0, -2.0]);
        assert_eq!(dossier.proxy_envelope.cruise_speed_mps, [45.0, 60.0]);
        assert!(dossier
            .aliases
            .iter()
            .any(|alias| alias.alias == "Shahed-136"));
        assert!(dossier
            .aliases
            .iter()
            .all(|alias| alias.policy == "source_dossier_only"));
    }

    #[test]
    fn campaign_object_ids_are_neutral() {
        for class in campaign_classes() {
            let id = class.id.to_ascii_lowercase();
            assert!(!id.contains("shahed"), "{id}");
            assert!(!id.contains("iranian"), "{id}");
            assert!(!id.contains("geran"), "{id}");
        }
    }

    #[test]
    fn default_balance_has_exact_records_and_minimum_positive_margin() {
        let config = CampaignConfig::shahed_public_proxy_default();
        let plan = build_campaign_plan(&config).expect("plan builds");
        assert_eq!(plan.len(), 1_000);
        let positives = plan
            .iter()
            .filter(|record| record.class.is_shahed_public_proxy)
            .count();
        assert_eq!(positives, 80);
        assert!(positives >= 50);
    }

    #[test]
    fn worker_cap_never_exceeds_forty_or_runtime_budget() {
        let mut config = CampaignConfig::shahed_public_proxy_default();
        config.records = 100;
        config.workers = Some(128);
        let runtime =
            RuntimePlan::from_signals(BackendMode::Cpu, BackendSignals::new(128, false, false))
                .expect("runtime");
        let workers = compute_worker_count(config.workers, config.records, &runtime);
        assert!(workers <= 40);
        assert!(workers <= runtime.recommended_worker_budget);
    }

    #[test]
    fn small_campaign_fixture_writes_manifest_and_disables_progress_in_tests() {
        let tmp_dir = tempfile::tempdir().expect("tempdir");
        let mut config = CampaignConfig::shahed_public_proxy_default();
        config.records = 12;
        config.shahed_min = 3;
        config.shahed_target = 4;
        config.workers = Some(4);
        config.progress = false;
        config.output_dir = tmp_dir.path().join("campaign");
        let report = run_monte_carlo_campaign(config).expect("campaign run");
        assert_eq!(report.records, 12);
        assert_eq!(report.shahed_positive_records, 4);
        assert!(report.worker_count <= 4);
        assert!(!report.progress_enabled);
        assert!(report.campaign_manifest_path.exists());
        assert!(report.class_balance_path.exists());
        assert!(report.dataset_card_path.exists());
        assert!(report.runtime_report_path.exists());
        assert!(report.benchmark_report_path.exists());
        assert!(tmp_dir
            .path()
            .join("campaign/records/record_000001/frame_labels.csv")
            .exists());
        assert!(tmp_dir
            .path()
            .join("campaign/models/cfar_tracker_baseline/model_metadata.json")
            .exists());
        assert!(tmp_dir
            .path()
            .join("campaign/qa/model_eval_temporal_tiny_model.json")
            .exists());
    }
}
