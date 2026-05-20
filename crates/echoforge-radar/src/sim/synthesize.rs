use crate::cfar::ca_cfar_1d;
use crate::link_budget::{evaluate_link_budget, snr_to_target_amplitude, LinkBudgetResult};
use crate::scene::{
    SceneDescriptor, TargetClass, TargetEntity, TargetKinematics,
};

use super::config::{NoiseProfile, RadarSimConfig, TakeoffProfile};
use super::dft::{slow_time_complex_dft, slow_time_dft_magnitude};
use super::episode::{DetectionRecord, EpisodeSeed, SplitMix64, SyntheticEpisode};
use super::helpers::{
    build_ground_glints, class_default_rcs_scalar, entity_initial_range_recovery,
    integrate_profiles, range_bin_to_m,
};

#[path = "synthesize_loop.rs"]
mod synthesize_loop;

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
    let scene = SceneDescriptor::from_radar_config(&config, &noise, vec![TargetEntity {
        class: TargetClass::ShahedClassPiston,
        kinematics: TargetKinematics::FromTakeoffProfile(profile),
        spawn_time_s: 0.0,
    }]);
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
    config: RadarSimConfig,
    noise: NoiseProfile,
    seed: EpisodeSeed,
) -> SyntheticEpisode {
    // Wave 5 Lane J: lifted the Lane I single-entity gate. The scene
    // MAY contain multiple TargetEntities; each entity dispatches its
    // own kinematic state via TargetKinematics::state_at and the
    // synthesis loop accumulates per-entity returns into a shared IQ
    // stream. Multipath ghost entities resolve their parent's geometry
    // and contribute a phantom return offset by the two-ray multipath
    // term (Skolnik 3rd ed. §1.6).
    assert!(
        !scene.targets.is_empty(),
        "synthesize_scene requires at least one TargetEntity in SceneDescriptor::targets"
    );

    // Lane I/J bridged note: scene.geometry / scene.environment
    // are preserved on the descriptor for serde + Lane K follow-up
    // (per-scene RCS dispatch); the physics path still sources antenna
    // height, atmospherics, multipath coefficient, and clutter regime
    // from `config` / `noise` so pre-Lane-I reproduction fixtures
    // remain byte-stable.
    let _ = (scene.geometry, scene.environment);

    // Resolve a bridged `TakeoffProfile` for the SyntheticEpisode
    // `profile` field. When the first entity is FromTakeoffProfile the
    // prior code path round-trips byte-stably; otherwise we populate
    // a default-shaped profile so downstream consumers that read
    // `episode.profile` don't blow up. The authoritative per-entity
    // state lives in `target_states` (first entity) and the per-class
    // SNR list in `per_target_snr_db`.
    let first_entity = &scene.targets[0];
    let first_profile = match &first_entity.kinematics {
        TargetKinematics::FromTakeoffProfile(profile) => *profile,
        _ => TakeoffProfile::default(),
    };

    let mut rng = SplitMix64::new(seed.0);

    // Per-entity initial-state link-budget evaluation. Replace the
    // prior `target_snr_db` knob with a transparent radar-equation
    // result evaluated at each entity's initial geometry. Sub-horizon
    // entities contribute zero return (target_amp == 0), which the
    // rest of the chain treats identically to a missing target.
    let mut per_entity_target_amp: Vec<f32> = Vec::with_capacity(scene.targets.len());
    let mut per_entity_snr_db: Vec<f64> = Vec::with_capacity(scene.targets.len());
    let mut first_link_result: Option<LinkBudgetResult> = None;

    // Resolve per-entity initial state (for the link-budget pass) and
    // per-entity RCS scalar. Ghosts inherit their parent's amp scaled
    // by |Γ|; this matches the Skolnik §1.6 multipath convention.
    let entity_initial_states: Vec<super::episode::TargetState> = scene
        .targets
        .iter()
        .map(|entity| match &entity.kinematics {
            TargetKinematics::MultipathGhost { parent_idx, .. } => {
                // Resolve parent's initial state for diagnostic SNR.
                let parent = scene.targets.get(*parent_idx).unwrap_or(first_entity);
                let parent_initial_range = entity_initial_range_recovery(parent);
                parent
                    .kinematics
                    .state_at(0.0, parent_initial_range, config.radar_altitude_agl_m)
            }
            _ => {
                let initial_range = entity_initial_range_recovery(entity);
                entity
                    .kinematics
                    .state_at(0.0, initial_range, config.radar_altitude_agl_m)
            }
        })
        .collect();

    for (idx, entity) in scene.targets.iter().enumerate() {
        let entity_initial = entity_initial_states[idx];
        let prop_ctx = config.propagation_context(&entity_initial);
        let rcs_scalar = class_default_rcs_scalar(&entity.class, &entity.kinematics, &first_profile);
        let link = evaluate_link_budget(&config.link_budget(), &prop_ctx, rcs_scalar.max(0.0));
        let amp_full = if link.above_horizon && link.snr_db.is_finite() {
            snr_to_target_amplitude(link.snr_db, noise.awgn_sigma)
        } else {
            0.0
        };
        // Ghosts inherit a scaled amplitude — their kinematics carry
        // the reflection coefficient |Γ|. Apply the linear scale and
        // the dB-equivalent to the SNR.
        let (amp_effective, snr_effective) = match &entity.kinematics {
            TargetKinematics::MultipathGhost {
                reflection_coefficient_magnitude,
                ..
            } => {
                let gamma = reflection_coefficient_magnitude.max(0.0) as f32;
                let snr_offset_db = if *reflection_coefficient_magnitude > 0.0 {
                    20.0 * reflection_coefficient_magnitude.log10()
                } else {
                    f64::NEG_INFINITY
                };
                (amp_full * gamma, link.snr_db + snr_offset_db)
            }
            _ => (amp_full, link.snr_db),
        };
        per_entity_target_amp.push(amp_effective);
        per_entity_snr_db.push(snr_effective);
        if idx == 0 {
            first_link_result = Some(link);
        }
    }
    let link_result_first = first_link_result.expect("at least one entity");

    let waveform = config.waveform();
    let sample_count = waveform.samples().len();
    let compressed_len = sample_count.saturating_mul(2).saturating_sub(1);
    let glints = build_ground_glints(sample_count, &noise, &mut rng);

    let (iq, profiles, compressed_complex, states_first) =
        synthesize_loop::run_synthesis_loop(
            &config, &noise, seed, &scene,
            &per_entity_target_amp, &glints, &first_profile,
        );

    let integrated = integrate_profiles(&profiles, compressed_len);
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
    let range_doppler_proxy = slow_time_dft_magnitude(&profiles, compressed_len);
    let range_doppler_complex = slow_time_complex_dft(&compressed_complex, config.pulse_count);

    SyntheticEpisode {
        seed,
        config,
        profile: first_profile,
        noise,
        target_states: states_first,
        iq,
        range_profiles_by_pulse: profiles,
        integrated_range_profile: integrated,
        range_doppler_proxy,
        detections,
        diagnostic_snr_db: link_result_first.snr_db,
        diagnostic_link_budget: link_result_first,
        range_doppler_complex,
        per_target_snr_db: per_entity_snr_db,
    }
}
