use super::*;
use crate::{SceneDescriptor, TargetClass, TargetEntity, TargetKinematics};

#[test]
fn synthesize_scene_applies_scene_environment_overrides() {
    let config = RadarSimConfig {
        pulse_count: 1,
        sample_rate_hz: 1_000_000.0,
        pulse_width_s: 64e-6,
        bandwidth_hz: 1_000_000.0,
        carrier_hz: 3.0e9,
        atmospheric_one_way_db_per_km: 0.0,
        rain_rate_mm_per_h: 0.0,
        ground_reflection_coefficient_magnitude: 0.0,
        ..RadarSimConfig::default()
    };
    let mut noise = NoiseProfile::real_world_proxy_v1();
    noise.awgn_sigma = 0.0;
    noise.clutter_sigma = 0.0;
    noise.rfi_probability = 0.0;
    noise.ground_glint_count = 0;
    noise.clutter_regime = None;

    let mut scene = SceneDescriptor::from_radar_config(
        &config,
        &noise,
        vec![TargetEntity {
            class: TargetClass::ShahedClassPiston,
            kinematics: TargetKinematics::FromTakeoffProfile(TakeoffProfile::default()),
            spawn_time_s: 0.0,
        }],
    );
    scene.environment.atmospheric_one_way_db_per_km = 0.8;
    scene.environment.rain_rate_mm_per_h = 24.0;
    scene.environment.ground_reflection_coefficient_magnitude = 0.5;

    let episode = synthesize_scene(scene, config, noise, EpisodeSeed(55));
    assert_eq!(episode.config.atmospheric_one_way_db_per_km, 0.8);
    assert_eq!(episode.config.rain_rate_mm_per_h, 24.0);
    assert_eq!(episode.config.ground_reflection_coefficient_magnitude, 0.5);
    let diag = &episode.pulse_diagnostics[0].source_diagnostics[0];
    assert!(
        diag.propagation_loss_db.is_finite() && diag.propagation_loss_db > 0.0,
        "scene environment should feed propagation diagnostics"
    );
}

#[test]
fn receive_window_masks_out_of_swath_returns() {
    let config = RadarSimConfig {
        pulse_count: 2,
        sample_rate_hz: 1_000_000.0,
        pulse_width_s: 32e-6,
        bandwidth_hz: 1_000_000.0,
        carrier_hz: 3.0e9,
        receive_window_start_m: 0.0,
        receive_window_end_m: Some(500.0),
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
            class: TargetClass::TerrainGlint,
            kinematics: TargetKinematics::TerrainGlint {
                range_m: 2_000.0,
                altitude_agl_m: 25.0,
            },
            spawn_time_s: 0.0,
        }],
    );

    let episode = synthesize_scene(scene, config, noise, EpisodeSeed(56));
    assert!(episode
        .pulse_diagnostics
        .iter()
        .all(|pulse| pulse.source_diagnostics[0].masked_by_receive_window));
    assert!(episode
        .pulse_diagnostics
        .iter()
        .all(|pulse| pulse.source_diagnostics[0].received_power_w == 0.0));
    assert!(episode.integrated_range_profile.iter().all(|v| *v == 0.0));
}

#[test]
fn disabled_realism_hooks_are_byte_stable_and_enabled_hooks_change_iq() {
    let base_config = RadarSimConfig {
        pulse_count: 4,
        ..RadarSimConfig::default()
    };
    let mut noise = NoiseProfile::real_world_proxy_v1();
    noise.awgn_sigma = 0.0;
    noise.clutter_sigma = 0.0;
    noise.rfi_probability = 0.0;
    noise.ground_glint_count = 0;
    noise.clutter_regime = None;
    let profile = TakeoffProfile::default();
    let seed = EpisodeSeed(57);

    let base = synthesize_takeoff_episode(base_config.clone(), profile, noise, seed);
    let disabled = synthesize_takeoff_episode(base_config.clone(), profile, noise, seed);
    assert_eq!(base.iq, disabled.iq);

    let enabled_config = RadarSimConfig {
        receiver_impairment: Some(crate::ReceiverImpairmentProfile {
            awgn_sigma: 0.0,
            phase_noise_std_rad: 0.0,
            amplitude_scintillation_sigma: 0.0,
            adc_bits: 12,
            clipping_level: 2.5,
            timing_jitter_samples: 0.0,
            dropped_pulse_probability: 0.0,
            gain_imbalance_db: 1.0,
            phase_imbalance_rad: 0.1,
            calibration_drift_db: 0.0,
        }),
        interference_profile: Some(crate::RfiProfile {
            burst_probability: 1.0,
            burst_amplitude: 0.1,
            narrowband_cw_power: 0.02,
            cochannel_emitters: 1,
            sidelobe_pressure: 0.02,
        }),
        transient_events: vec![TransientEvent {
            kind: TransientEventKind::Dropout,
            start_s: 0.0,
            duration_s: 10.0,
            strength: 0.25,
            target_ref: None,
        }],
        ..base_config
    };
    let enabled = synthesize_takeoff_episode(enabled_config, profile, noise, seed);
    assert_ne!(base.iq, enabled.iq);
}
