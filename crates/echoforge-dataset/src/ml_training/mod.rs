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
mod report;
mod scene;
mod types;
mod util;

// ---------------------------------------------------------------------------
// Public re-exports (all symbols that were previously at the flat module path
// remain accessible at `echoforge_dataset::ml_training::*`).
// ---------------------------------------------------------------------------

pub use config::{
    MlTrainingDataConfig, DEFAULT_ML_TRAINING_DATASET_ID, DEFAULT_ML_TRAINING_OUTPUT,
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
    feature_schema, label_schema, local_external_data_config, normalization_stats,
    quality_report, runtime_report, split_counts,
};
use types::MlTrainingDataReport as Report;
use util::{build_record_plan, write_csv};

pub fn run_ml_training_data(
    config: MlTrainingDataConfig,
) -> Result<MlTrainingDataReport, DatasetError> {
    validate_config(&config)?;
    guard_output_dir(&config.output_dir)?;
    let mut ctx = begin_generation_run(
        &config.output_dir, config.backend, config.workers, config.records,
        config.time_window_s, config.frame_rate_hz,
    )?;
    let plan_start = Instant::now();
    let plans = build_record_plan(&config, ctx.frame_count)?;
    ctx.push_timing("record_plan", plan_start);

    let generation_start = Instant::now();
    let mut outputs =
        run_record_workers(&config, &ctx.runtime, &plans, ctx.frame_count, ctx.worker_count)?;
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
        &config, &ctx.runtime, ctx.worker_count, ctx.frame_count,
        &ctx.stage_timings, ctx.overall_start.elapsed(),
    );
    write_json_pretty(&out.join("runtime_report.json"), &rt_report)?;

    let q_report = quality_report(&config, &summaries, &feature_rows);
    write_json_pretty(&out.join("quality_report.json"), &q_report)?;

    let external_sources = external_calibration_sources();
    write_json_pretty(&out.join("external_calibration_sources.json"), &external_sources)?;
    let local_ext = local_external_data_config(&external_sources);
    write_json_pretty(&out.join("local_external_data_config.example.json"), &local_ext)?;

    let d_card = dataset_card(&config, &q_report)?;
    write_json_pretty(&out.join("dataset_card.json"), &d_card)?;

    // V3 unified-path artifact: per-tier Pd/Pfa proxy.
    let tier_observations: Vec<_> = outputs
        .iter()
        .map(|o| o.per_tier_observation.clone())
        .collect();
    let per_tier_rows = aggregate_per_tier_metrics(&tier_observations);
    let per_tier_path = out.join("per_tier_pd_pfa.json");
    write_json_pretty(&per_tier_path, &json!({
        "schema_id": "echoforge.ml_training.per_tier_pd_pfa.v1",
        "source": "V3 unified-path PhaseTieredDetector evaluation per CPI per record",
        "notes": vec![
            "pd_proxy = n_detections / n_episodes; positive rows are Pd surrogate, confuser rows are Pfa surrogate",
            "n_horizon_blocked counts CPIs where the detector reported below-horizon geometry",
            "Tier::None counts CPIs where the arbiter never latched a phase tier",
        ],
        "rows": per_tier_rows,
    }))?;

    let manifest = dataset_manifest(&config, summaries.clone(), ctx.frame_count, external_sources);
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
mod tests {
    use std::collections::BTreeMap;

    use echoforge_radar::{BackendMode, BackendSignals, RuntimePlan, Tier};

    use crate::split::SplitKind;

    use super::config::MlTrainingDataConfig;
    use super::envelope::sample_envelope;
    use super::pipeline::{build_frame_products, evaluate_phase_tiered};
    use super::scene::{adapt_envelope_to_takeoff_profile, build_noise_profile, build_scene_descriptor};
    use super::types::SplitMix64;
    use super::util::{build_record_plan, ml_classes};
    use super::run_ml_training_data;

    use echoforge_radar::{
        synthesize_scene, EpisodeSeed, RadarSimConfig, SyntheticEpisode,
    };

    fn synthesize_episode_for_test(
        envelope: &super::types::MlEnvelope,
        class: &super::types::MlClass,
        scenario_seed: u64,
    ) -> SyntheticEpisode {
        let mut rng = SplitMix64::new(scenario_seed);
        // Replay the envelope sampler so the takeoff-profile RNG draws
        // match generate_record exactly.
        let _ = sample_envelope(class, &mut rng);
        let _ = rng.range_usize(24, 48);
        let noise = build_noise_profile(envelope);
        let sim_config = RadarSimConfig {
            sample_rate_hz: 1_000_000.0,
            pulse_width_s: 64e-6,
            bandwidth_hz: 800_000.0,
            carrier_hz: 9_600_000_000.0,
            pulse_count: 32,
            pri_s: 900e-6,
            target_snr_db: envelope.base_snr_db as f64,
            ..RadarSimConfig::default()
        };
        let profile = adapt_envelope_to_takeoff_profile(envelope, &mut rng);
        let scene = build_scene_descriptor(class, profile, &sim_config, &noise);
        synthesize_scene(scene, sim_config, noise, EpisodeSeed(scenario_seed ^ 0x0dd5_136))
    }

    #[test]
    fn split_assignment_keeps_scenario_object_seed_in_one_split() {
        let mut config = MlTrainingDataConfig::shahed_public_proxy_default();
        config.records = 64;
        let plans = build_record_plan(&config, 180).expect("plan");
        let mut seen = BTreeMap::<String, SplitKind>::new();
        for plan in plans {
            let key = format!("{:016x}:{:016x}", plan.scenario_seed, plan.object_seed);
            if let Some(previous) = seen.insert(key, plan.split) {
                assert_eq!(previous, plan.split);
            }
        }
        let splits = seen.values().fold(BTreeMap::new(), |mut acc, split| {
            *acc.entry(*split).or_insert(0usize) += 1;
            acc
        });
        assert_eq!(*splits.get(&SplitKind::Train).unwrap_or(&0), 45);
        assert_eq!(*splits.get(&SplitKind::Validation).unwrap_or(&0), 10);
        assert_eq!(*splits.get(&SplitKind::Test).unwrap_or(&0), 9);
    }

    #[test]
    fn frame_feature_generation_is_deterministic_and_finite() {
        let mut config = MlTrainingDataConfig::shahed_public_proxy_default();
        config.records = 12;
        config.time_window_s = 6.0;
        config.frame_rate_hz = 2.0;
        let plans = build_record_plan(&config, 12).expect("plan");
        let plan = plans
            .iter()
            .find(|plan| plan.class.is_public_proxy_positive)
            .expect("positive plan");
        let mut rng_a = SplitMix64::new(plan.scenario_seed);
        let mut rng_b = SplitMix64::new(plan.scenario_seed);
        let envelope_a = sample_envelope(&plan.class, &mut rng_a);
        let envelope_b = sample_envelope(&plan.class, &mut rng_b);
        assert_eq!(envelope_a.dimensions_m, envelope_b.dimensions_m);
        let episode_a = synthesize_episode_for_test(&envelope_a, &plan.class, plan.scenario_seed);
        let episode_b = synthesize_episode_for_test(&envelope_b, &plan.class, plan.scenario_seed);
        let (features_a, _, _, _) =
            build_frame_products(&config, plan, &envelope_a, &episode_a, 12, 32);
        let (features_b, _, _, _) =
            build_frame_products(&config, plan, &envelope_b, &episode_b, 12, 32);
        assert_eq!(
            serde_json::to_string(&features_a).expect("features serialize"),
            serde_json::to_string(&features_b).expect("features serialize")
        );
        assert!(features_a.iter().all(|feature| {
            [
                feature.snr_db,
                feature.cfar_statistic,
                feature.tbd_track_score,
                feature.local_noise_floor_db,
                feature.doppler_scr,
                feature.micro_doppler_energy,
                feature.normalized_snr,
            ]
            .iter()
            .all(|value| value.is_finite())
        }));
    }

    #[test]
    fn small_training_fixture_writes_required_artifacts() {
        let tmp_dir = tempfile::tempdir().expect("tempdir");
        let mut config = MlTrainingDataConfig::shahed_public_proxy_default();
        config.records = 12;
        config.time_window_s = 6.0;
        config.frame_rate_hz = 2.0;
        config.backend = BackendMode::Cpu;
        config.workers = Some(4);
        config.output_dir = tmp_dir.path().join("ml-training");

        let report = run_ml_training_data(config).expect("training data run");
        assert_eq!(report.records, 12);
        assert_eq!(report.positive_records, 2);
        assert_eq!(report.hard_negative_families, 10);
        assert_eq!(report.frame_count, 12);
        assert!(report.dataset_manifest_path.exists());
        assert!(report.dataset_card_path.exists());
        assert!(report.split_manifest_path.exists());
        assert!(report.records_path.exists());
        assert!(report.features_path.exists());
        assert!(report.label_schema_path.exists());
        assert!(report.feature_schema_path.exists());
        assert!(report.normalization_stats_path.exists());
        assert!(report.quality_report_path.exists());
        assert!(report.runtime_report_path.exists());

        let record = tmp_dir.path().join("ml-training/records/record_000001");
        assert!(record.join("products/iq_complex.zarr/.zarray").exists());
        assert!(record.join("products/range_profile.zarr/.zarray").exists());
        assert!(record
            .join("products/range_doppler_proxy.zarr/.zarray")
            .exists());
        assert!(record
            .join("micro_doppler/stft_spectrogram.zarr/.zarray")
            .exists());
        assert!(record
            .join("multi_view/range_doppler_time.zarr/.zarray")
            .exists());
        assert!(record
            .join("learned_windows/window_16.zarr/.zarray")
            .exists());
        assert!(record.join("frame_labels.csv").exists());
        assert!(record.join("truth_metadata.json").exists());
        assert!(record.join("detector_events.json").exists());
    }

    #[test]
    fn forced_gpu_uses_existing_readiness_error() {
        let mut signals = BackendSignals::new(128, true, true);
        signals.gpu_usable = false;
        signals.gpu_unusable_reason = Some(
            "GPU detected but free memory is 606 MiB; need at least 2048 MiB for this runtime"
                .to_string(),
        );
        let err = RuntimePlan::from_signals(BackendMode::Gpu, signals)
            .expect_err("forced GPU should fail");
        assert!(err.to_string().contains("free memory is 606 MiB"));
    }

    #[test]
    fn phase_tiered_synthetic_positive_reaches_cruise() {
        let class_pos = ml_classes()
            .into_iter()
            .find(|class| class.is_public_proxy_positive)
            .unwrap();
        let class_neg = ml_classes()
            .into_iter()
            .find(|class| class.hard_negative_family == "ground_vehicle")
            .unwrap();
        let scenario_seed: u64 = 42;
        let mut rng_pos = SplitMix64::new(scenario_seed);
        let env_pos = sample_envelope(&class_pos, &mut rng_pos);
        let episode_pos = synthesize_episode_for_test(&env_pos, &class_pos, scenario_seed);
        let obs_pos = evaluate_phase_tiered(&episode_pos, true);

        let mut rng_neg = SplitMix64::new(scenario_seed + 1);
        let env_neg = sample_envelope(&class_neg, &mut rng_neg);
        let episode_neg =
            synthesize_episode_for_test(&env_neg, &class_neg, scenario_seed + 1);
        let obs_neg = evaluate_phase_tiered(&episode_neg, false);

        let pos_cruise_count =
            obs_pos.tier_counts.get(&Tier::Cruise).copied().unwrap_or(0);
        let neg_cruise_count =
            obs_neg.tier_counts.get(&Tier::Cruise).copied().unwrap_or(0);
        eprintln!(
            "pos tier_counts={:?} neg tier_counts={:?}",
            obs_pos.tier_counts, obs_neg.tier_counts
        );
        assert!(
            pos_cruise_count > 0,
            "positive synthetic stream must reach cruise tier"
        );
        assert_eq!(
            neg_cruise_count, 0,
            "confuser synthetic stream must NOT reach cruise tier"
        );
    }

    #[test]
    fn hard_negative_roster_has_acceptance_coverage() {
        let families = ml_classes()
            .into_iter()
            .filter(|class| class.is_hard_negative)
            .map(|class| class.hard_negative_family)
            .collect::<std::collections::BTreeSet<_>>();
        assert!(families.len() >= 10);
        for family in [
            "single_bird",
            "bird_flock",
            "bat_insect_cloud",
            "balloon_weather",
            "kite",
            "windborne_debris",
            "ground_vehicle",
            "power_line_pylon",
            "wind_turbine",
            "rain_cell",
            "dust_haze",
            "rfi_burst",
            "terrain_only",
        ] {
            assert!(families.contains(family), "missing {family}");
        }
    }
}
