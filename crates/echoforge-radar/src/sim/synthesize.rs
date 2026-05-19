use std::f32::consts::PI;

use crate::cfar::ca_cfar_1d;
use crate::clutter::generate_clutter_sequence;
use crate::link_budget::{evaluate_link_budget, snr_to_target_amplitude, LinkBudgetResult};
use crate::pulse_compression::{magnitude, pulse_compress_windowed, CompressionWindow};
use crate::scene::{
    EnvironmentDescriptor, SceneDescriptor, SiteGeometry, TargetClass, TargetEntity,
    TargetKinematics,
};
use crate::ComplexSample;

use super::config::{NoiseProfile, RadarSimConfig, TakeoffProfile};
use super::dft::{slow_time_complex_dft, slow_time_dft_magnitude};
use super::episode::{DetectionRecord, EpisodeSeed, SplitMix64, SyntheticEpisode};
use super::helpers::{
    build_ground_glints, class_default_rcs_scalar, entity_initial_range_recovery,
    integrate_profiles, micro_doppler_envelope, range_bin_to_m, resolve_entity_state, C_M_PER_S,
};
use super::config::polarization_amplitude_scale;

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
    let scene = SceneDescriptor {
        geometry: SiteGeometry {
            antenna_altitude_agl_m: config.radar_altitude_agl_m,
        },
        environment: EnvironmentDescriptor {
            clutter_regime: noise.clutter_regime,
            atmospheric_one_way_db_per_km: config.atmospheric_one_way_db_per_km,
            rain_rate_mm_per_h: config.rain_rate_mm_per_h,
            ground_reflection_coefficient_magnitude: config
                .ground_reflection_coefficient_magnitude,
        },
        targets: vec![TargetEntity {
            class: TargetClass::ShahedClassPiston,
            kinematics: TargetKinematics::FromTakeoffProfile(profile),
            spawn_time_s: 0.0,
        }],
    };
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

    let waveform = config.waveform();
    let reference = waveform.samples();
    let sample_count = reference.len();
    let compressed_len = sample_count.saturating_mul(2).saturating_sub(1);
    let mut rng = SplitMix64::new(seed.0);
    let mut phase_walk = 0.0f32;

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

    let glints = build_ground_glints(sample_count, &noise, &mut rng);

    let mut iq = Vec::with_capacity(config.pulse_count);
    let mut profiles = Vec::with_capacity(config.pulse_count);
    let mut compressed_complex: Vec<Vec<ComplexSample>> = Vec::with_capacity(config.pulse_count);
    // target_states (bridged) holds the FIRST entity's per-pulse
    // states only. Multi-entity per-pulse state vectors are downstream
    // Lane K work; the first-entity contract preserves the prior
    // single-target consumer surface.
    let mut states_first: Vec<super::episode::TargetState> = Vec::with_capacity(config.pulse_count);
    let mut clutter_state = 0.0f32;

    // When a ClutterRegime is configured, pre-generate the full
    // per-pulse per-range-bin clutter cube using the cited K /
    // Weibull / log-normal samplers (with proper AR(1) spatial and
    // temporal correlation). The Gaussian AR(1) path remains the
    // fall-back when no regime is set, preserving byte-for-byte
    // bridged with pre-Lane-C fixtures.
    //
    // Pre-generating (rather than calling `sample_clutter_amplitude`
    // per cell) keeps the inner loop tight and uses
    // `generate_clutter_sequence`'s proper AR(1) recurrence across
    // both range and pulse axes — `sample_clutter_amplitude` is
    // per-sample only and would lose the cross-pulse correlation.
    let clutter_cube: Option<Vec<f32>> = noise.clutter_regime.as_ref().map(|regime| {
        // Mix the episode seed with a stable lane-C tag so the clutter
        // stream is decorrelated from the rest of the RNG draws in the
        // synthesis loop. The tag is arbitrary but fixed.
        let clutter_seed = seed.0 ^ 0xC10C_C0DE_u64;
        generate_clutter_sequence(regime, sample_count, config.pulse_count, clutter_seed)
    });

    for pulse in 0..config.pulse_count {
        let t_s = pulse as f64 * config.pri_s;
        let mut received = vec![ComplexSample::new(0.0, 0.0); sample_count];

        // Wave 5 Lane J: per-entity contribution loop. Entities are
        // visited in `scene.targets` order, which makes the first
        // entity's scintillation draw the FIRST RNG consumer of this
        // pulse — preserving byte-stable bridged with the
        // single-entity Lane I synth (which only draws one
        // scintillation per pulse).
        //
        // Multipath ghost entities skip their own scintillation draw
        // (they're a phantom of the parent) and use the parent's
        // already-resolved state. Pure-static entities (WindTurbine,
        // tethered Balloon) still go through the same draw sequence
        // so the RNG consumption is uniform across entity types.
        for (entity_idx, entity) in scene.targets.iter().enumerate() {
            let amp_full = per_entity_target_amp[entity_idx];
            if amp_full == 0.0 {
                // Even a zero-amp entity must consume a scintillation
                // draw so single-entity vs multi-entity ordering with
                // mixed sub-horizon scenes remains predictable. The
                // first entity ALWAYS draws so pre-Lane-J fixtures
                // (single ShahedClassPiston) byte-match.
                let _ = rng.normal_f32();
                continue;
            }

            // Resolve this entity's per-pulse state. Ghosts inherit
            // parent's state then apply the multipath range offset.
            let (state, range_offset_m) = resolve_entity_state(
                scene.targets.as_slice(),
                entity_idx,
                t_s,
                config.radar_altitude_agl_m,
            );

            let effective_range_m = (state.range_m + range_offset_m).max(0.0);
            let delay_samples =
                ((2.0 * effective_range_m / C_M_PER_S) * config.sample_rate_hz).round()
                    as isize;
            let doppler_hz =
                2.0 * state.radial_velocity_mps * config.carrier_hz / C_M_PER_S;
            let pulse_phase = 2.0 * std::f64::consts::PI * doppler_hz * t_s;

            // Per-entity micro-Doppler dispatch. Entities that carry
            // a TakeoffProfile route through the prior multi-blade /
            // single-sinusoid dispatch so pre-Lane-J fixtures
            // byte-match. Confuser variants are handled by their
            // native generators downstream of this synthesis loop;
            // at the synthesis surface their AM envelope is the unit
            // multiplier (1.0) — a conservative first-order proxy
            // that does not over-claim micro-Doppler fidelity for
            // confusers.
            let micro = micro_doppler_envelope(entity, t_s, &state, &first_profile);

            let scintillation = (noise.amplitude_scintillation_sigma * rng.normal_f32())
                .exp()
                .clamp(0.4, 2.5);

            // Wave 4.5 H2: per-pulse polarization scaling. When
            // neither sequence is set we skip the multiplication
            // entirely — preserving exact bit-equality with
            // pre-Lane-H2 fixtures.
            let amp_base = amp_full * micro as f32 * scintillation;
            let amp = if config.pol_tx_sequence.is_some() || config.pol_rx_sequence.is_some() {
                let (pol_tx, pol_rx) = config.polarization_for_pulse(pulse);
                amp_base * polarization_amplitude_scale(pol_tx, pol_rx)
            } else {
                amp_base
            };

            for (i, sample) in reference.iter().enumerate() {
                let dst = i as isize + delay_samples;
                if dst < 0 || dst >= sample_count as isize {
                    continue;
                }
                let phase = pulse_phase as f32 + phase_walk;
                let phasor = ComplexSample::new(phase.cos(), phase.sin());
                received[dst as usize] += *sample * phasor * amp;
            }

            if entity_idx == 0 {
                states_first.push(state);
            }
        }

        // Stochastic terms (clutter, AWGN, RFI, phase walk) are
        // applied AFTER per-entity returns are accumulated. The RNG
        // consumption order is identical to the pre-Lane-J single-
        // entity synth so byte-stable fixtures replay unchanged.
        for (index, sample) in received.iter_mut().enumerate() {
            let clutter_raw = match clutter_cube.as_ref() {
                Some(cube) => cube[pulse * sample_count + index] * noise.clutter_sigma_0_scale,
                None => {
                    clutter_state = noise.clutter_correlation * clutter_state
                        + (1.0 - noise.clutter_correlation)
                            * rng.normal_scaled(noise.clutter_sigma);
                    clutter_state
                }
            };
            let glint = glints
                .iter()
                .find(|(bin, _)| *bin == index)
                .map(|(_, amp)| *amp)
                .unwrap_or(0.0);
            let clutter = clutter_raw + glint;
            sample.re += clutter + rng.normal_scaled(noise.awgn_sigma);
            sample.im += clutter * 0.35 + rng.normal_scaled(noise.awgn_sigma);

            if rng.unit_f32() < noise.rfi_probability {
                let phase = 2.0 * PI * rng.unit_f32();
                *sample += ComplexSample::new(phase.cos(), phase.sin()) * noise.rfi_amplitude;
            }
        }

        phase_walk += rng.normal_scaled(noise.phase_noise_std_rad);
        let compressed =
            pulse_compress_windowed(&received, &reference, CompressionWindow::taylor_default());
        let mag = magnitude(&compressed);
        iq.push(received);
        profiles.push(mag);
        compressed_complex.push(compressed);
    }

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
