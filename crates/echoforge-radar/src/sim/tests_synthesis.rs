use super::tests_helpers::high_snr_episode;
use super::*;
use crate::{SceneDescriptor, TargetClass, TargetEntity, TargetKinematics};

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

#[test]
fn synthesize_scene_allows_empty_target_roster() {
    let config = RadarSimConfig {
        pulse_count: 4,
        ..RadarSimConfig::default()
    };
    let noise = NoiseProfile::real_world_proxy_v1();
    let scene = SceneDescriptor::from_radar_config(&config, &noise, Vec::new());
    let episode = synthesize_scene(scene, config, noise, EpisodeSeed(2026));

    assert_eq!(episode.profile, TakeoffProfile::default());
    assert!(episode.target_states.is_empty());
    assert!(episode.per_target_snr_db.is_empty());
    assert_eq!(episode.pulse_diagnostics.len(), 4);
    assert_eq!(episode.iq.len(), 4);
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
    let profile = TakeoffProfile {
        initial_range_m: 1_700.0,
        runway_heading_deg: 24.0,
        ground_speed_mps: 34.0,
        acceleration_mps2: 0.8,
        climb_rate_mps: 5.0,
        max_altitude_m: 420.0,
        radial_velocity_bias_mps: -15.0,
        pitch_jitter_deg: 0.9,
        yaw_jitter_deg: 1.1,
        propulsor_hz: 88.0,
        micro_doppler_hz: 36.0,
        rcs_scalar: 1.2,
        blade_count: None,
        blade_length_m: None,
    };
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
        via_scene.profile, profile,
        "scene path must preserve the first target's TakeoffProfile"
    );
    assert_eq!(
        via_wrapper.integrated_range_profile, via_scene.integrated_range_profile,
        "wrapper and unified path must produce byte-equal integrated profile"
    );
    assert_eq!(
        via_wrapper.detections, via_scene.detections,
        "wrapper and unified path must produce byte-equal detections"
    );
}

#[test]
fn synthesize_scene_respects_spawn_time_and_emits_diagnostics() {
    let config = RadarSimConfig {
        pulse_count: 3,
        pri_s: 1.0,
        sample_rate_hz: 1_000_000.0,
        bandwidth_hz: 1_000_000.0,
        carrier_hz: 3.0e9,
        ..RadarSimConfig::default()
    };
    let mut noise = NoiseProfile::real_world_proxy_v1();
    noise.awgn_sigma = 0.0;
    noise.clutter_sigma = 0.0;
    noise.rfi_probability = 0.0;
    noise.ground_glint_count = 0;
    noise.clutter_regime = None;

    let scene = SceneDescriptor::from_radar_config(
        &config,
        &noise,
        vec![TargetEntity {
            class: TargetClass::ShahedClassPiston,
            kinematics: TargetKinematics::FromTakeoffProfile(TakeoffProfile {
                initial_range_m: 2_000.0,
                runway_heading_deg: 0.0,
                ground_speed_mps: 50.0,
                acceleration_mps2: 0.0,
                climb_rate_mps: 0.0,
                max_altitude_m: 1.0,
                radial_velocity_bias_mps: 0.0,
                pitch_jitter_deg: 0.0,
                yaw_jitter_deg: 0.0,
                propulsor_hz: 95.0,
                micro_doppler_hz: 42.0,
                rcs_scalar: 1.0,
                blade_count: None,
                blade_length_m: None,
            }),
            spawn_time_s: 10.0,
        }],
    );
    let episode = synthesize_scene(scene, config, noise, EpisodeSeed(7));

    assert!(episode
        .pulse_diagnostics
        .iter()
        .all(|pulse| pulse.source_diagnostics.len() == 1));
    assert!(episode
        .pulse_diagnostics
        .iter()
        .all(|pulse| !pulse.source_diagnostics[0].active));
    assert!(episode
        .pulse_diagnostics
        .iter()
        .all(|pulse| pulse.source_diagnostics[0].received_power_w == 0.0));
    assert!(episode.integrated_range_profile.iter().all(|v| *v == 0.0));
}

#[test]
fn synthesize_scene_dynamic_link_budget_tracks_range() {
    let config = RadarSimConfig {
        pulse_count: 3,
        pri_s: 1.0,
        sample_rate_hz: 1_000_000.0,
        bandwidth_hz: 1_000_000.0,
        carrier_hz: 3.0e9,
        ..RadarSimConfig::default()
    };
    let mut noise = NoiseProfile::real_world_proxy_v1();
    noise.awgn_sigma = 0.0;
    noise.clutter_sigma = 0.0;
    noise.rfi_probability = 0.0;
    noise.ground_glint_count = 0;
    noise.clutter_regime = None;

    let scene = SceneDescriptor::from_radar_config(
        &config,
        &noise,
        vec![TargetEntity {
            class: TargetClass::ShahedClassPiston,
            kinematics: TargetKinematics::FromTakeoffProfile(TakeoffProfile {
                initial_range_m: 2_000.0,
                runway_heading_deg: 0.0,
                ground_speed_mps: 50.0,
                acceleration_mps2: 0.0,
                climb_rate_mps: 0.0,
                max_altitude_m: 2_000.0,
                radial_velocity_bias_mps: 0.0,
                pitch_jitter_deg: 0.0,
                yaw_jitter_deg: 0.0,
                propulsor_hz: 95.0,
                micro_doppler_hz: 42.0,
                rcs_scalar: 1.0,
                blade_count: None,
                blade_length_m: None,
            }),
            spawn_time_s: 0.0,
        }],
    );
    let episode = synthesize_scene(scene, config, noise, EpisodeSeed(11));

    let p0 = episode.pulse_diagnostics[0].source_diagnostics[0].received_power_w;
    let p2 = episode.pulse_diagnostics[2].source_diagnostics[0].received_power_w;
    assert!(
        p0.is_finite() && p2.is_finite() && p0 > p2,
        "received power should fall as range increases; p0={p0}, p2={p2}"
    );
}

#[test]
fn synthesize_scene_emits_normalized_course_aspect_diagnostics() {
    let config = RadarSimConfig {
        pulse_count: 1,
        pri_s: 1.0,
        sample_rate_hz: 1_000_000.0,
        bandwidth_hz: 1_000_000.0,
        carrier_hz: 3.0e9,
        ..RadarSimConfig::default()
    };
    let mut noise = NoiseProfile::real_world_proxy_v1();
    noise.awgn_sigma = 0.0;
    noise.clutter_sigma = 0.0;
    noise.rfi_probability = 0.0;
    noise.ground_glint_count = 0;
    noise.clutter_regime = None;

    let scene = SceneDescriptor::from_radar_config(
        &config,
        &noise,
        vec![TargetEntity {
            class: TargetClass::Bird,
            kinematics: TargetKinematics::Bird {
                cruise_speed_mps: 20.0,
                altitude_agl_m: 120.0,
                heading_deg: -30.0,
                wingbeat_hz: 4.0,
                wing_length_m: 0.5,
            },
            spawn_time_s: 0.0,
        }],
    );
    let episode = synthesize_scene(scene, config, noise, EpisodeSeed(17));

    assert_eq!(episode.pulse_diagnostics.len(), 1);
    let pulse = &episode.pulse_diagnostics[0];
    assert_eq!(pulse.pulse_index, 0);
    assert!((pulse.time_s - 0.0).abs() < 1e-12);
    assert_eq!(pulse.source_diagnostics.len(), 1);

    let diag = &pulse.source_diagnostics[0];
    assert_eq!(diag.entity_index, 0);
    assert_eq!(diag.class_name, "Bird");
    assert!(diag.active);
    assert!(
        (diag.aspect_deg - 330.0).abs() < 1e-9,
        "expected normalized aspect of 330deg from course_deg=-30deg, got {}",
        diag.aspect_deg
    );
    assert!(diag.received_power_w.is_finite());
}
