use super::tests_helpers::high_snr_episode;
use super::*;

#[test]
fn deterministic_episode_repeats_for_seed() {
    let config = RadarSimConfig {
        pulse_count: 8,
        ..RadarSimConfig::default()
    };
    let a = synthesize_takeoff_episode(
        config.clone(),
        TakeoffProfile::default(),
        NoiseProfile::real_world_proxy_v1(),
        EpisodeSeed(42),
    );
    let b = synthesize_takeoff_episode(
        config,
        TakeoffProfile::default(),
        NoiseProfile::real_world_proxy_v1(),
        EpisodeSeed(42),
    );
    assert_eq!(a.integrated_range_profile, b.integrated_range_profile);
    assert_eq!(a.detections, b.detections);
}

#[test]
fn high_snr_takeoff_has_detection() {
    let episode = high_snr_episode(12, 7);
    assert!(!episode.detections.is_empty());
}

#[test]
fn low_snr_products_are_finite() {
    let config = RadarSimConfig {
        pulse_count: 6,
        target_snr_db: 2.0,
        ..RadarSimConfig::default()
    };
    let mut noise = NoiseProfile::real_world_proxy_v1();
    noise.awgn_sigma = 0.2;
    noise.rfi_probability = 0.05;
    let episode =
        synthesize_takeoff_episode(config, TakeoffProfile::default(), noise, EpisodeSeed(99));
    assert!(episode
        .integrated_range_profile
        .iter()
        .all(|value| value.is_finite()));
    assert!(episode
        .range_doppler_proxy
        .iter()
        .flatten()
        .all(|value| value.is_finite()));
}

#[test]
fn rfi_changes_products_but_stays_deterministic() {
    let config = RadarSimConfig {
        pulse_count: 6,
        ..RadarSimConfig::default()
    };
    let mut clean_noise = NoiseProfile::real_world_proxy_v1();
    clean_noise.rfi_probability = 0.0;
    clean_noise.clutter_sigma = 0.0;
    clean_noise.ground_glint_count = 0;

    let mut dirty_noise = clean_noise;
    dirty_noise.rfi_probability = 0.15;
    dirty_noise.clutter_sigma = 0.08;
    dirty_noise.ground_glint_count = 3;

    let clean = synthesize_takeoff_episode(
        config.clone(),
        TakeoffProfile::default(),
        clean_noise,
        EpisodeSeed(12),
    );
    let dirty_a = synthesize_takeoff_episode(
        config.clone(),
        TakeoffProfile::default(),
        dirty_noise,
        EpisodeSeed(12),
    );
    let dirty_b = synthesize_takeoff_episode(
        config,
        TakeoffProfile::default(),
        dirty_noise,
        EpisodeSeed(12),
    );

    assert_ne!(
        clean.integrated_range_profile,
        dirty_a.integrated_range_profile
    );
    assert_eq!(
        dirty_a.integrated_range_profile,
        dirty_b.integrated_range_profile
    );
}

/// Lane I (Wave 4) — `synthesize_scene` must produce the same
/// products as `synthesize_takeoff_episode` when invoked with the
/// equivalent single-entity `SceneDescriptor`. This is the
/// in-module byte-stability gate; a heavier-weight version with
/// IQ-bit-equal assertions lives in
/// `tests/physics_correctness.rs::c_unified_takeoff_wrapper_matches_scene_direct`.
#[test]
fn synthesize_scene_single_target_matches_prior() {
    use crate::scene::{SceneDescriptor, TargetClass, TargetEntity, TargetKinematics};

    let config = RadarSimConfig {
        pulse_count: 6,
        ..RadarSimConfig::default()
    };
    let profile = TakeoffProfile::default();
    let noise = NoiseProfile::real_world_proxy_v1();
    let seed = EpisodeSeed(2027);

    let via_wrapper = synthesize_takeoff_episode(config.clone(), profile, noise, seed);

    let scene = SceneDescriptor::from_radar_config(
        &config,
        &noise,
        vec![TargetEntity {
            class: TargetClass::ShahedClassPiston,
            kinematics: TargetKinematics::FromTakeoffProfile(profile),
            spawn_time_s: 0.0,
        }],
    );
    let via_scene = synthesize_scene(scene, config, noise, seed);

    assert_eq!(
        via_wrapper.integrated_range_profile, via_scene.integrated_range_profile,
        "wrapper and unified path must produce byte-equal integrated profile"
    );
    assert_eq!(
        via_wrapper.detections, via_scene.detections,
        "wrapper and unified path must produce byte-equal detections"
    );
}
