use crate::cfar::ca_cfar_1d;
use crate::link_budget::{evaluate_link_budget, snr_to_target_amplitude, LinkBudgetResult};
use crate::rcs::{Polarization, Rcs};
use crate::scene::{SceneDescriptor, TargetClass, TargetEntity, TargetKinematics};

use super::config::{NoiseProfile, RadarSimConfig, TakeoffProfile};
use super::dft::{slow_time_complex_dft, slow_time_dft_magnitude};
use super::episode::{
    DetectionRecord, EpisodeSeed, SourceDiagnostics, SplitMix64, SyntheticEpisode,
};
use super::helpers::{
    build_ground_glints, class_default_rcs_scalar, integrate_profiles, public_proxy_rcs_class_name,
    range_bin_to_m, target_aspect_deg, target_elevation_deg,
};

#[path = "synthesize_loop.rs"]
mod synthesize_loop;

pub(super) struct EntityLinkEvaluation {
    pub amp: f32,
    pub link_budget: LinkBudgetResult,
    pub diagnostics: SourceDiagnostics,
}

pub(super) struct EntityLinkRequest<'a> {
    pub scene: &'a SceneDescriptor,
    pub config: &'a RadarSimConfig,
    pub noise: &'a NoiseProfile,
    pub first_profile: &'a TakeoffProfile,
    pub rcs_catalog: &'a Rcs,
    pub entity_idx: usize,
    pub t_s: f64,
    pub seed: u64,
    pub pulse_index: usize,
    pub active: bool,
    pub masked_by_receive_window: bool,
}

fn empty_link_budget_result() -> LinkBudgetResult {
    LinkBudgetResult {
        received_power_w: 0.0,
        noise_power_w: 0.0,
        snr_linear: 0.0,
        snr_db: f64::NEG_INFINITY,
        propagation_factor_db: 0.0,
        atmospheric_loss_db: 0.0,
        rain_loss_db: 0.0,
        free_space_path_loss_db: 0.0,
        coherent_integration_gain_db: 0.0,
        above_horizon: false,
    }
}

pub(super) fn evaluate_entity_link(request: EntityLinkRequest<'_>) -> EntityLinkEvaluation {
    let EntityLinkRequest {
        scene,
        config,
        noise,
        first_profile,
        rcs_catalog,
        entity_idx,
        t_s,
        seed,
        pulse_index,
        active,
        masked_by_receive_window,
    } = request;
    let entity = &scene.targets[entity_idx];
    let (state, range_offset_m) = super::helpers::resolve_entity_state(
        scene.targets.as_slice(),
        entity_idx,
        t_s,
        config.radar_altitude_agl_m,
    );
    let effective_range_m = (state.range_m + range_offset_m).max(0.0);
    let effective_state = super::episode::TargetState {
        range_m: effective_range_m,
        ..state
    };
    let prop_ctx = config.propagation_context(&effective_state);
    let tx_pol = config.polarization_for_pulse(pulse_index).0;
    let rx_pol = config.polarization_for_pulse(pulse_index).1;
    let lookup_pol = if tx_pol == rx_pol {
        tx_pol
    } else {
        Polarization::Cross
    };
    let aspect_deg = target_aspect_deg(&effective_state);
    let elevation_deg = target_elevation_deg(&effective_state);
    let class_name = format!("{:?}", entity.class);
    let base_rcs_dbsm = match public_proxy_rcs_class_name(entity, scene.targets.as_slice()) {
        Some(table_name) => {
            let evaluated = rcs_catalog.evaluate(
                table_name,
                aspect_deg,
                elevation_deg,
                config.carrier_hz / 1.0e9,
                lookup_pol,
                seed ^ (entity_idx as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15),
                pulse_index,
            );
            if evaluated.is_finite() {
                evaluated
            } else {
                10.0 * class_default_rcs_scalar(&entity.class, &entity.kinematics, first_profile)
                    .max(1e-12)
                    .log10()
            }
        }
        None => {
            10.0 * class_default_rcs_scalar(&entity.class, &entity.kinematics, first_profile)
                .max(1e-12)
                .log10()
        }
    };
    let mut rcs_m2 = 10f64.powf(base_rcs_dbsm / 10.0).max(0.0);
    if let TargetKinematics::MultipathGhost {
        reflection_coefficient_magnitude,
        ..
    } = &entity.kinematics
    {
        let gamma = reflection_coefficient_magnitude.max(0.0);
        rcs_m2 *= gamma * gamma;
    }
    let link = evaluate_link_budget(&config.link_budget(), &prop_ctx, rcs_m2);
    let amp =
        if active && !masked_by_receive_window && link.above_horizon && link.snr_db.is_finite() {
            snr_to_target_amplitude(link.snr_db, noise.awgn_sigma)
        } else {
            0.0
        };
    let (tx_scale, rx_scale) = (tx_pol, rx_pol);
    let amp = if active && !masked_by_receive_window {
        amp * super::config::polarization_amplitude_scale(tx_scale, rx_scale)
    } else {
        0.0
    };
    let signal_after_processing_w = if link.received_power_w > 0.0 {
        link.received_power_w / 10f64.powf(config.processing_loss_db.max(0.0) / 10.0)
    } else {
        0.0
    };
    let clutter_power_w = if let Some(regime) = noise.clutter_regime {
        let range_resolution_m = super::helpers::C_M_PER_S / (2.0 * config.bandwidth_hz.max(1.0));
        let grazing_factor = (effective_state.altitude_m + config.radar_altitude_agl_m)
            .abs()
            .max(1.0)
            / effective_state.range_m.max(1.0);
        let illuminated_span_m = (effective_state.range_m * grazing_factor)
            .abs()
            .max(range_resolution_m);
        let sigma0_linear = 10f64.powf(regime.mean_power_dbsm_per_m2 / 10.0)
            * f64::from(noise.clutter_sigma_0_scale.max(0.0));
        let clutter_rcs_m2 = sigma0_linear * range_resolution_m * illuminated_span_m;
        let clutter_link = evaluate_link_budget(&config.link_budget(), &prop_ctx, clutter_rcs_m2);
        clutter_link.received_power_w
    } else {
        f64::from(noise.clutter_sigma.max(0.0)).powi(2)
    };
    let structured_rfi_pressure = config
        .interference_profile
        .map(|profile| profile.pressure())
        .unwrap_or(0.0);
    let transient_rfi_pressure = config
        .transient_events
        .iter()
        .filter(|event| event.active_at(t_s))
        .filter(|event| {
            matches!(
                event.kind,
                super::config::TransientEventKind::RfiBurst
                    | super::config::TransientEventKind::WeatherVolume
            )
        })
        .map(|event| event.bounded_strength())
        .sum::<f32>()
        .clamp(0.0, 16.0);
    let interference_power_w = f64::from(noise.rfi_probability.max(0.0))
        * f64::from(noise.rfi_amplitude.max(0.0)).powi(2)
        * link.noise_power_w.max(1.0)
        + f64::from(structured_rfi_pressure + transient_rfi_pressure) * link.noise_power_w.max(1.0);
    let combined_noise_w = (link.noise_power_w + clutter_power_w + interference_power_w).max(1e-30);
    let sinr_db = if signal_after_processing_w > 0.0 {
        10.0 * (signal_after_processing_w / combined_noise_w).log10()
    } else {
        f64::NEG_INFINITY
    };
    let propagation_loss_db = if link.free_space_path_loss_db.is_finite() {
        link.free_space_path_loss_db + link.atmospheric_loss_db + link.rain_loss_db
            - link.propagation_factor_db
    } else {
        f64::INFINITY
    };
    EntityLinkEvaluation {
        amp,
        link_budget: link,
        diagnostics: SourceDiagnostics {
            pulse_index,
            entity_index: entity_idx,
            class_name,
            active,
            time_s: t_s,
            range_m: effective_state.range_m,
            altitude_m: effective_state.altitude_m,
            aspect_deg,
            elevation_deg,
            rcs_dbsm: base_rcs_dbsm,
            rcs_m2,
            received_power_w: if active && !masked_by_receive_window {
                link.received_power_w
            } else {
                0.0
            },
            thermal_noise_power_w: link.noise_power_w,
            clutter_power_w,
            interference_power_w,
            propagation_loss_db,
            processing_loss_db: config.processing_loss_db,
            sinr_db: if active && !masked_by_receive_window {
                sinr_db
            } else {
                f64::NEG_INFINITY
            },
            masked_by_receive_window,
        },
    }
}

/// bridged wrapper around the unified [`synthesize_scene`] path.
///
/// **Lane I (Wave 4) refactor:** before this lane, this function was
/// the only positive-target physics generator; confusers traversed a
/// separate envelope-statistics generator in
/// `echoforge-dataset/src/ml_training.rs::build_frame_products`. The
/// bifurcation meant the generator identity labelled the class. This
/// wrapper now constructs a single-entity [`SceneDescriptor`] (with
/// [`TargetClass::ShahedClassPiston`] + [`TargetKinematics::FromTakeoffProfile`])
/// and forwards through [`synthesize_scene`]. The output is
/// byte-stable with the pre-Lane-I implementation, so every existing
/// reproduction fixture and physics-correctness test passes
/// unchanged. The byte-stability gate lives in
/// `tests/physics_correctness.rs::c_unified_takeoff_wrapper_matches_scene_direct`.
///
/// Lane J extends `synthesize_scene` with native per-class kinematics
/// for confusers (Bird, GroundVehicle, WindTurbine, Balloon, Kite, Helicopter)
/// and multi-entity dispatch; this wrapper stays as the canonical
/// single-target API for downstream code.
pub fn synthesize_takeoff_episode(
    config: RadarSimConfig,
    profile: TakeoffProfile,
    noise: NoiseProfile,
    seed: EpisodeSeed,
) -> SyntheticEpisode {
    let scene = SceneDescriptor::from_radar_config(
        &config,
        &noise,
        vec![TargetEntity {
            class: TargetClass::ShahedClassPiston,
            kinematics: TargetKinematics::FromTakeoffProfile(profile),
            spawn_time_s: 0.0,
        }],
    );
    synthesize_scene(scene, config, noise, seed)
}

/// Unified scene-physics entry point — Lane I (Wave 4).
///
/// Consumes a [`SceneDescriptor`] (geometry + environment + per-target
/// entities) plus the sensor config and noise profile, returns the
/// same [`SyntheticEpisode`] product as the prior
/// [`synthesize_takeoff_episode`].
///
/// **Scope of Lane I:** only the single-entity [`TargetKinematics::FromTakeoffProfile`]
/// path was honoured by the physics chain.
///
/// **Wave 5 Lane J extension (this lane):** the gate is lifted. The
/// scene MAY now contain N entities, each with its own
/// [`TargetKinematics`] variant. Confuser classes (Bird, GroundVehicle,
/// WindTurbine, Balloon, Kite, Helicopter) traverse the same chain as
/// positive Shahed-class entities; the MultipathGhost variant
/// synthesises a phantom return offset by `2·h_r·h_t/R` from its
/// parent's geometry (Skolnik 3rd ed. §1.6). Per-class RCS is
/// resolved by a first-order proxy keyed on the dossier RCS table;
/// full per-class RCS-aspect lookup against
/// [`crate::rcs::Rcs::seeded_public_proxy_v1`] is downstream Lane K
/// work.
///
/// **Byte-stability guarantee:** when called with a single-entity
/// scene containing a [`TargetClass::ShahedClassPiston`] +
/// [`TargetKinematics::FromTakeoffProfile`] entity, the synthesis loop
/// reproduces pre-Lane-J byte-stable IQ output. The single-target RNG
/// consumption order is preserved: one scintillation draw per entity
/// per pulse (the first entity's draw is the same draw the prior
/// per pulse (the first entity's draw is the same draw the prior
/// path would have consumed), then the per-bin clutter/awgn/RFI loop
/// runs unchanged.
pub fn synthesize_scene(
    scene: SceneDescriptor,
    mut config: RadarSimConfig,
    mut noise: NoiseProfile,
    seed: EpisodeSeed,
) -> SyntheticEpisode {
    // Wave 5 Lane J: lifted the Lane I single-entity gate. The scene
    // MAY contain multiple TargetEntities; each entity dispatches its
    // own kinematic state via TargetKinematics::state_at and the
    // synthesis loop accumulates per-entity returns into a shared IQ
    // stream. Multipath ghost entities resolve their parent's geometry
    // and contribute a phantom return offset by the two-ray multipath
    // term (Skolnik 3rd ed. §1.6).
    config.radar_altitude_agl_m = scene.geometry.antenna_altitude_agl_m;
    config.atmospheric_one_way_db_per_km = scene.environment.atmospheric_one_way_db_per_km;
    config.rain_rate_mm_per_h = scene.environment.rain_rate_mm_per_h;
    config.ground_reflection_coefficient_magnitude =
        scene.environment.ground_reflection_coefficient_magnitude;
    noise.clutter_regime = scene.environment.clutter_regime;

    // Resolve a bridged `TakeoffProfile` for the SyntheticEpisode
    // `profile` field. When the first entity is FromTakeoffProfile the
    // prior code path round-trips byte-stably; otherwise we populate
    // a default-shaped profile so downstream consumers that read
    // `episode.profile` don't blow up. The authoritative per-entity
    // state lives in `target_states` (first entity) and the per-class
    // SNR list in `per_target_snr_db`.
    let first_profile = if let Some(first_entity) = scene.targets.first() {
        match &first_entity.kinematics {
            TargetKinematics::FromTakeoffProfile(profile) => *profile,
            _ => TakeoffProfile::default(),
        }
    } else {
        TakeoffProfile::default()
    };

    let mut rng = SplitMix64::new(seed.0);

    let rcs_catalog = Rcs::seeded_public_proxy_v1();

    let mut per_entity_snr_db: Vec<f64> = Vec::with_capacity(scene.targets.len());
    for (entity_idx, _) in scene.targets.iter().enumerate() {
        let evaluation = evaluate_entity_link(EntityLinkRequest {
            scene: &scene,
            config: &config,
            noise: &noise,
            first_profile: &first_profile,
            rcs_catalog: &rcs_catalog,
            entity_idx,
            t_s: 0.0,
            seed: seed.0,
            pulse_index: 0,
            active: 0.0 >= scene.targets[entity_idx].spawn_time_s,
            masked_by_receive_window: false,
        });
        per_entity_snr_db.push(if evaluation.diagnostics.active {
            evaluation.link_budget.snr_db
        } else {
            f64::NEG_INFINITY
        });
    }
    let first_link_result = if scene.targets.is_empty() {
        let mut zero = empty_link_budget_result();
        let default_state = super::episode::TargetState {
            time_s: 0.0,
            range_m: 1_450.0,
            altitude_m: 0.0,
            radial_velocity_mps: 0.0,
            pitch_deg: 0.0,
            yaw_deg: 0.0,
            course_deg: 0.0,
            propulsor_phase_rad: 0.0,
        };
        let prop = config.propagation_context(&default_state);
        let budget = config.link_budget();
        let eval = evaluate_link_budget(&budget, &prop, 0.0);
        zero.noise_power_w = eval.noise_power_w;
        zero.free_space_path_loss_db = eval.free_space_path_loss_db;
        zero.atmospheric_loss_db = eval.atmospheric_loss_db;
        zero.rain_loss_db = eval.rain_loss_db;
        zero.propagation_factor_db = eval.propagation_factor_db;
        zero.coherent_integration_gain_db = eval.coherent_integration_gain_db;
        zero
    } else {
        let mut result = evaluate_entity_link(EntityLinkRequest {
            scene: &scene,
            config: &config,
            noise: &noise,
            first_profile: &first_profile,
            rcs_catalog: &rcs_catalog,
            entity_idx: 0,
            t_s: 0.0,
            seed: seed.0,
            pulse_index: 0,
            active: 0.0 >= scene.targets[0].spawn_time_s,
            masked_by_receive_window: false,
        })
        .link_budget;
        if scene.targets[0].spawn_time_s > 0.0 {
            result.received_power_w = 0.0;
            result.snr_linear = 0.0;
            result.snr_db = f64::NEG_INFINITY;
        }
        result
    };

    let waveform = config.waveform();
    let sample_count = waveform.samples().len();
    let compressed_len = sample_count.saturating_mul(2).saturating_sub(1);
    let glints = build_ground_glints(sample_count, &noise, &mut rng);

    let loop_output = synthesize_loop::run_synthesis_loop(
        &config,
        &noise,
        seed,
        &scene,
        &rcs_catalog,
        &glints,
        &first_profile,
    );

    let integrated = integrate_profiles(&loop_output.iq_magnitudes, compressed_len);
    let decisions = ca_cfar_1d(&integrated, config.cfar_params());
    let detections = decisions
        .iter()
        .filter(|decision| decision.evaluated && decision.detected)
        .map(|decision| {
            let range_m = range_bin_to_m(
                decision.index,
                sample_count.saturating_sub(1),
                config.sample_rate_hz,
            );
            let confidence = if decision.threshold.is_finite() && decision.threshold > 0.0 {
                decision.statistic / decision.threshold
            } else {
                0.0
            };
            DetectionRecord {
                range_bin: decision.index,
                range_m,
                statistic: decision.statistic,
                threshold: decision.threshold,
                noise_estimate: decision.noise_estimate,
                confidence,
            }
        })
        .collect();
    let range_doppler_proxy = slow_time_dft_magnitude(&loop_output.iq_magnitudes, compressed_len);
    let range_doppler_complex =
        slow_time_complex_dft(&loop_output.compressed_complex, config.pulse_count);

    SyntheticEpisode {
        seed,
        config,
        profile: first_profile,
        noise,
        target_states: loop_output.states_first,
        iq: loop_output.iq,
        range_profiles_by_pulse: loop_output.iq_magnitudes,
        integrated_range_profile: integrated,
        range_doppler_proxy,
        detections,
        diagnostic_snr_db: first_link_result.snr_db,
        diagnostic_link_budget: first_link_result,
        range_doppler_complex,
        per_target_snr_db: per_entity_snr_db,
        pulse_diagnostics: loop_output.pulse_diagnostics,
    }
}
