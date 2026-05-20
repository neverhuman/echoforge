//! ML training dataset generator (V3 unified-path migration).
//!
//! ## Wave 5 Lane K_rust — unified `synthesize_scene` path
//!
//! Before Wave 5, this module bifurcated its physics flow:
//!
//! 1. Positives went through [`echoforge_radar::synthesize_takeoff_episode`]
//!    (full radar chain: pulse-compression, slow-time DFT, K/Weibull
//!    clutter, OS-CFAR, MTI/MTD) for tensor products.
//! 2. Confusers went through the SAME wrapper for tensor products, but
//!    `build_frame_products` rebuilt streaming features from envelope
//!    statistics WITHOUT consulting the synthesised episode. The
//!    generator identity (envelope branch) literally labelled the class
//!    — a credibility-sweep red flag.
//!
//! This module now routes ALL records (positives + confusers) through
//! the unified [`echoforge_radar::synthesize_scene`] entry point that
//! Lane I (Wave 4) introduced. Each record builds a
//! [`echoforge_radar::SceneDescriptor`] whose single
//! [`echoforge_radar::TargetEntity`] carries an explicit
//! [`echoforge_radar::TargetClass`] that names the truth class, then
//! calls `synthesize_scene` to produce the [`echoforge_radar::SyntheticEpisode`].
//! Streaming features (`build_frame_products`) are then derived from
//! that single episode, no longer recomputed from envelope statistics
//! alone.
//!
//! Sub-modules
//! -----------
//! - `config`   — [`MlTrainingDataConfig`], constants, validation helpers
//! - `types`    — internal structs, [`MlTrainingDataReport`], [`PerTierMetrics`],
//!               [`SplitMix64`]
//! - `envelope` — per-class envelope sampling
//! - `scene`    — `build_scene_descriptor`, `confuser_class_for_family`
//! - `pipeline` — worker orchestration, `build_frame_products`, artifact writers
//! - `report`   — quality/runtime/manifest/card generation, per-tier aggregation
//! - `util`     — CSV/tensor I/O, DSP helpers, seed utilities, class tables

mod config;
mod envelope;
mod pipeline;
mod pipeline_writers;
mod report;
mod report_card;
mod scene;
mod types;
mod types_ext;
mod util;

// ---------------------------------------------------------------------------
// Public re-exports (all symbols that were previously at the flat module path
// remain accessible at `echoforge_dataset::ml_training::*`).
// ---------------------------------------------------------------------------

pub use config::{
    MlTrainingDataConfig, BEST_FINAL_OUTPUT, BEST_FINAL_POSITIVE_CLASS_IDS, BEST_FINAL_SCENARIO_ID,
    BEST_FINAL_SENSOR_IDS, DEFAULT_ML_TRAINING_DATASET_ID, DEFAULT_ML_TRAINING_OUTPUT,
};
pub use types::{MlTrainingDataReport, PerTierMetrics};

// ---------------------------------------------------------------------------
// Top-level entry point
// ---------------------------------------------------------------------------

use std::time::Instant;

use serde_json::json;

use crate::export::write_json_pretty;
use crate::monte_carlo::DatasetError;

use crate::guard::{begin_generation_run, guard_output_dir};
use config::validate_config;
use pipeline::run_record_workers;
use report::{
    aggregate_per_tier_metrics, dataset_card, dataset_manifest, external_calibration_sources,
    feature_schema, label_schema, local_external_data_config, normalization_stats, quality_report,
    runtime_report, split_counts,
};
use types::MlTrainingDataReport as Report;
use util::{build_record_plan, write_csv};

pub fn run_ml_training_data(
    config: MlTrainingDataConfig,
) -> Result<MlTrainingDataReport, DatasetError> {
    validate_config(&config)?;
    guard_output_dir(&config.output_dir)?;
    let mut ctx = begin_generation_run(
        &config.output_dir,
        config.backend,
        config.workers,
        config.records,
        config.time_window_s,
        config.frame_rate_hz,
    )?;
    let plan_start = Instant::now();
    let plans = build_record_plan(&config, ctx.frame_count)?;
    ctx.push_timing("record_plan", plan_start);

    let generation_start = Instant::now();
    let mut outputs = run_record_workers(
        &config,
        &ctx.runtime,
        &plans,
        ctx.frame_count,
        ctx.worker_count,
    )?;
    outputs.sort_by_key(|output| output.summary.record_index);
    ctx.push_timing("record_generation", generation_start);

    let post_start = Instant::now();
    let summaries: Vec<_> = outputs.iter().map(|o| o.summary.clone()).collect();
    let feature_rows: Vec<_> = outputs.iter().map(|o| o.features.clone()).collect();
    let split_rows: Vec<_> = outputs.iter().map(|o| o.split.clone()).collect();

    let out = &config.output_dir;
    write_csv(&out.join("records.csv"), &summaries)?;
    write_csv(&out.join("features.csv"), &feature_rows)?;
    write_csv(&out.join("split_manifest.csv"), &split_rows)?;
    write_json_pretty(&out.join("feature_schema.json"), &feature_schema())?;
    write_json_pretty(&out.join("label_schema.json"), &label_schema())?;

    let normalization = normalization_stats(&config.dataset, &feature_rows);
    write_json_pretty(&out.join("normalization_stats.json"), &normalization)?;

    ctx.push_timing("postprocess", post_start);
    let rt_report = runtime_report(
        &config,
        &ctx.runtime,
        ctx.worker_count,
        ctx.frame_count,
        &ctx.stage_timings,
        ctx.overall_start.elapsed(),
    );
    write_json_pretty(&out.join("runtime_report.json"), &rt_report)?;

    let q_report = quality_report(&config, &summaries, &feature_rows);
    write_json_pretty(&out.join("quality_report.json"), &q_report)?;

    let external_sources = external_calibration_sources();
    write_json_pretty(
        &out.join("external_calibration_sources.json"),
        &external_sources,
    )?;
    let local_ext = local_external_data_config(&external_sources);
    write_json_pretty(
        &out.join("local_external_data_config.example.json"),
        &local_ext,
    )?;

    let d_card = dataset_card(&config, &q_report)?;
    write_json_pretty(&out.join("dataset_card.json"), &d_card)?;

    // V3 unified-path artifact: per-tier Pd/Pfa proxy.
    let tier_observations: Vec<_> = outputs
        .iter()
        .map(|o| o.per_tier_observation.clone())
        .collect();
    let per_tier_rows = aggregate_per_tier_metrics(&tier_observations);
    let per_tier_path = out.join("per_tier_pd_pfa.json");
    write_json_pretty(
        &per_tier_path,
        &json!({
            "schema_id": "echoforge.ml_training.per_tier_pd_pfa.v1",
            "source": "V3 unified-path PhaseTieredDetector evaluation per CPI per record",
            "notes": vec![
                "pd_proxy = n_detections / n_episodes; positive rows are Pd surrogate, confuser rows are Pfa surrogate",
                "n_horizon_blocked counts CPIs where the detector reported below-horizon geometry",
                "Tier::None counts CPIs where the arbiter never latched a phase tier",
            ],
            "rows": per_tier_rows,
        }),
    )?;

    let manifest = dataset_manifest(
        &config,
        summaries.clone(),
        ctx.frame_count,
        external_sources,
    );
    write_json_pretty(&out.join("dataset_manifest.json"), &manifest)?;
    let s_counts = split_counts(&summaries);
    Ok(Report {
        output_dir: config.output_dir.clone(),
        dataset_id: config.dataset,
        records: summaries.len(),
        positive_records: q_report.positive_records,
        hard_negative_families: q_report.hard_negative_family_coverage,
        frame_count: ctx.frame_count,
        split_counts: s_counts,
        worker_count: ctx.worker_count,
        runtime: ctx.runtime,
        dataset_manifest_path: config.output_dir.join("dataset_manifest.json"),
        dataset_card_path: config.output_dir.join("dataset_card.json"),
        split_manifest_path: config.output_dir.join("split_manifest.csv"),
        records_path: config.output_dir.join("records.csv"),
        features_path: config.output_dir.join("features.csv"),
        label_schema_path: config.output_dir.join("label_schema.json"),
        feature_schema_path: config.output_dir.join("feature_schema.json"),
        normalization_stats_path: config.output_dir.join("normalization_stats.json"),
        quality_report_path: config.output_dir.join("quality_report.json"),
        runtime_report_path: config.output_dir.join("runtime_report.json"),
        per_tier_pd_pfa_path: per_tier_path,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[path = "mod_tests.rs"]
mod tests;
