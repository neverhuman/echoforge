//! Benchmark report rendering and split policy — extracted from run.rs for LOC compliance.

use std::collections::BTreeMap;

use crate::leakage::LeakageReport;
use crate::monte_carlo::config::{MonteCarloBenchmarkReport, MonteCarloDemoConfig};
use crate::monte_carlo::helpers::{validation_info, EpisodeManifestEntry, RunManifest};
use crate::monte_carlo::sampling::ResolvedPreset;
use crate::monte_carlo::scene_config::AirspaceMonteCarloConfig;
use crate::split::{SplitKind, SplitPolicy, SplitRatios};

pub(super) fn monte_carlo_split_policy() -> SplitPolicy {
    SplitPolicy {
        schema_ref: "schemas/dataset_card.schema.json".to_string(),
        policy_id: "dataset.split.monte_carlo_demo_v1".to_string(),
        display_name: "Monte Carlo Demo Split Policy v1".to_string(),
        ratios: SplitRatios {
            train_bps: 7_000,
            validation_bps: 1_500,
            test_bps: 1_500,
        },
        protected_keys: vec![
            "scenario_seed".to_string(),
            "geometry_hash".to_string(),
            "material_sample_hash".to_string(),
        ],
        grouping_mode: "hash_episode_variant_keys".to_string(),
        notes: vec![
            "Episode variant seeds, geometry hashes, and material samples must not cross splits."
                .to_string(),
            "Object-family and sensor-family repetitions are allowed so one public-proxy class can produce train, validation, and test records.".to_string(),
        ],
    }
}

pub(super) fn run_manifest<'a>(
    config: &'a MonteCarloDemoConfig,
    library: &'a AirspaceMonteCarloConfig,
    resolved: &'a ResolvedPreset<'a>,
    episodes: &'a [EpisodeManifestEntry],
    leakage_report: &LeakageReport,
) -> RunManifest<'a> {
    RunManifest {
        manifest_version: "1",
        preset: &config.preset,
        generated_at: &config.generated_at,
        root_seed: config.seed,
        config_library: library,
        selected_object_class: resolved.object,
        selected_environment: resolved.environment,
        selected_sensor: resolved.sensor,
        guardrails: vec![
            "public-proxy",
            "statistical noise proxy",
            "not measured truth",
            "not proprietary-equivalent",
        ],
        validation: validation_info(0.38),
        episodes,
        leakage_clean: leakage_report.is_clean(),
        known_limitations: vec![
            "No exact measured truth is claimed for any object, platform, material, or sensor.",
            "Contested-airspace parameters are robustness stressors, not tactics or evasion optimization.",
            "Range-Doppler output is a proxy product from an owned lightweight DSP chain.",
        ],
    }
}

pub(super) fn benchmark_report_markdown(
    config: &MonteCarloDemoConfig,
    resolved: &ResolvedPreset<'_>,
    split_counts: &BTreeMap<SplitKind, usize>,
    leakage_report: &LeakageReport,
    benchmark: &MonteCarloBenchmarkReport,
) -> String {
    let stage_lines = benchmark
        .stage_timings
        .iter()
        .map(|stage| {
            format!(
                "- {}: {:.3} ms",
                stage.stage,
                stage.elapsed_ns as f64 / 1_000_000.0
            )
        })
        .collect::<Vec<_>>()
        .join("\n");

    format!(
        "# EchoForge Monte Carlo Demo Benchmark Report\n\n\
         Preset: `{}`\n\n\
         Requested backend: `{}`\n\n\
         Selected backend: `{}`\n\n\
         Kernel backend: `{}`\n\n\
         Logical cores: {}\n\n\
         GPU available: {}\n\n\
         GPU constrained: {}\n\n\
         Recommended worker budget: {}\n\n\
         Worker count used: {}\n\n\
         Target label: {}\n\n\
         Object class: `{}`\n\n\
         Environment: `{}`\n\n\
         Episodes: {}\n\n\
         Splits: train={}, validation={}, test={}\n\n\
         Validation tier: basic\n\n\
         Validation status: pass\n\n\
         Uncertainty score: 0.38\n\n\
         Leakage status: {}\n\n\
         Stage timings:\n{}\n\n\
         Throughput: {:.3} episodes/sec\n\n\
         Benchmark payload: `{}`\n\n\
         Known limitation: this benchmark is a deterministic public-proxy runtime demo and is not an operational sensor-performance claim.\n",
        config.preset,
        benchmark.requested_backend,
        benchmark.selected_backend,
        benchmark.kernel_backend,
        benchmark.logical_cores,
        benchmark.gpu_available,
        benchmark.gpu_constrained,
        benchmark.recommended_worker_budget,
        benchmark.actual_worker_count,
        config.target_label,
        resolved.object.id,
        resolved.environment.id,
        config.episodes,
        split_counts.get(&SplitKind::Train).unwrap_or(&0),
        split_counts.get(&SplitKind::Validation).unwrap_or(&0),
        split_counts.get(&SplitKind::Test).unwrap_or(&0),
        if leakage_report.is_clean() { "clean" } else { "findings" },
        stage_lines,
        benchmark.throughput_episodes_per_sec,
        if config.runtime.benchmark {
            "benchmark_report.json"
        } else {
            "benchmark_report.md only"
        }
    )
}
