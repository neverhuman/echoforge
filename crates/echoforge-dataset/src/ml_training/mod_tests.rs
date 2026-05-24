use std::collections::BTreeMap;

use echoforge_radar::{BackendMode, BackendSignals, RuntimePlan, Tier};

use crate::split::SplitKind;

use super::config::MlTrainingDataConfig;
use super::envelope::sample_envelope;
use super::pipeline::{build_frame_products, evaluate_phase_tiered};
use super::run_ml_training_data;
use super::scene::{build_noise_profile, build_scene_descriptor};
use super::types::SplitMix64;
use super::util::{build_record_plan, ml_classes};

use echoforge_radar::{synthesize_scene, EpisodeSeed, RadarSimConfig, SyntheticEpisode};

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
    let scene = build_scene_descriptor(class, envelope, &mut rng, &sim_config, &noise);
    synthesize_scene(
        scene,
        sim_config,
        noise,
        EpisodeSeed(scenario_seed ^ 0x00dd_5136),
    )
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
fn best_final_full_config_resolves_requested_sensor_class_counts() {
    let config = MlTrainingDataConfig::best_final_default();
    let plans = build_record_plan(&config, 180).expect("plan");
    assert_eq!(plans.len(), 3_900);

    let sensor_counts = plans.iter().fold(BTreeMap::new(), |mut acc, plan| {
        *acc.entry(plan.sensor_id.clone()).or_insert(0usize) += 1;
        acc
    });
    assert_eq!(sensor_counts.len(), 3);
    assert!(sensor_counts.values().all(|count| *count == 1_300));

    let positive_class_counts = plans
        .iter()
        .filter(|plan| plan.class.is_public_proxy_positive)
        .fold(BTreeMap::new(), |mut acc, plan| {
            *acc.entry(plan.class.class_id.clone()).or_insert(0usize) += 1;
            acc
        });
    assert_eq!(
        positive_class_counts.get("shahed-136-geran-2").copied(),
        Some(300)
    );
    assert_eq!(
        positive_class_counts.get("shahed-131-geran-1").copied(),
        Some(300)
    );
    assert_eq!(positive_class_counts.get("mohajer-6").copied(), Some(300));

    let negatives = plans
        .iter()
        .filter(|plan| plan.class.is_hard_negative)
        .count();
    assert_eq!(negatives, 3_000);
    for phase in ["early_takeoff", "mid_ramp", "cruise_altitude"] {
        assert!(
            plans.iter().any(|plan| plan.phase_target == phase),
            "missing phase target {phase}"
        );
    }
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
    let (features_a, _, _, _) = build_frame_products(&config, plan, &episode_a, 12, 32);
    let (features_b, _, _, _) = build_frame_products(&config, plan, &episode_b, 12, 32);
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
    let err =
        RuntimePlan::from_signals(BackendMode::Gpu, signals).expect_err("forced GPU should fail");
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
    let episode_neg = synthesize_episode_for_test(&env_neg, &class_neg, scenario_seed + 1);
    let obs_neg = evaluate_phase_tiered(&episode_neg, false);

    let pos_cruise_count = obs_pos.tier_counts.get(&Tier::Cruise).copied().unwrap_or(0);
    let neg_cruise_count = obs_neg.tier_counts.get(&Tier::Cruise).copied().unwrap_or(0);
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
