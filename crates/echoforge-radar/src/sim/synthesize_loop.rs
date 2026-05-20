//! Per-pulse synthesis loop — extracted from synthesize.rs for LOC compliance.

use std::f32::consts::PI;

use crate::clutter::generate_clutter_sequence;
use crate::pulse_compression::{magnitude, pulse_compress_windowed, CompressionWindow};
use crate::scene::SceneDescriptor;
use crate::ComplexSample;

use crate::sim::config::{NoiseProfile, RadarSimConfig, TakeoffProfile};
use crate::sim::episode::{EpisodeSeed, PulseDiagnostics, SplitMix64, TargetState};
use crate::sim::helpers::{micro_doppler_envelope, resolve_entity_state, C_M_PER_S};

/// Run the per-pulse synthesis loop.
/// Returns (iq, range_profiles_by_pulse, compressed_complex, target_states_first).
pub(super) fn run_synthesis_loop(
    config: &RadarSimConfig,
    noise: &NoiseProfile,
    seed: EpisodeSeed,
    scene: &SceneDescriptor,
    rcs_catalog: &crate::rcs::Rcs,
    glints: &[(usize, f32)],
    first_profile: &TakeoffProfile,
) -> (
    Vec<Vec<ComplexSample>>,
    Vec<Vec<f32>>,
    Vec<Vec<ComplexSample>>,
    Vec<TargetState>,
    Vec<PulseDiagnostics>,
) {
    let waveform = config.waveform();
    let reference = waveform.samples();
    let sample_count = reference.len();
    let mut rng = SplitMix64::new(seed.0);
    let mut phase_walk = 0.0f32;

    let clutter_cube: Option<Vec<f32>> = noise.clutter_regime.as_ref().map(|regime| {
        let clutter_seed = seed.0 ^ 0xC10C_C0DE_u64;
        generate_clutter_sequence(regime, sample_count, config.pulse_count, clutter_seed)
    });
    let mut clutter_state = 0.0f32;

    let mut iq = Vec::with_capacity(config.pulse_count);
    let mut profiles = Vec::with_capacity(config.pulse_count);
    let mut compressed_complex: Vec<Vec<ComplexSample>> = Vec::with_capacity(config.pulse_count);
    let mut states_first: Vec<TargetState> = Vec::with_capacity(config.pulse_count);
    let mut pulse_diagnostics: Vec<PulseDiagnostics> = Vec::with_capacity(config.pulse_count);

    for pulse in 0..config.pulse_count {
        let t_s = pulse as f64 * config.pri_s;
        let mut received = vec![ComplexSample::new(0.0, 0.0); sample_count];
        let mut source_diagnostics = Vec::with_capacity(scene.targets.len());

        for (entity_idx, entity) in scene.targets.iter().enumerate() {
            let active = t_s >= entity.spawn_time_s;
            let eval = super::evaluate_entity_link(
                scene,
                config,
                noise,
                first_profile,
                rcs_catalog,
                entity_idx,
                t_s,
                seed.0,
                pulse,
                active,
            );
            let state = resolve_entity_state(
                scene.targets.as_slice(),
                entity_idx,
                t_s,
                config.radar_altitude_agl_m,
            )
            .0;
            let effective_range_m = eval.diagnostics.range_m.max(0.0);
            let delay_samples =
                ((2.0 * effective_range_m / C_M_PER_S) * config.sample_rate_hz).round() as isize;
            let doppler_hz = 2.0 * state.radial_velocity_mps * config.carrier_hz / C_M_PER_S;
            let pulse_phase = 2.0 * std::f64::consts::PI * doppler_hz * t_s;

            let micro = micro_doppler_envelope(entity, t_s, &state, first_profile);
            let scintillation = (noise.amplitude_scintillation_sigma * rng.normal_f32())
                .exp()
                .clamp(0.4, 2.5);

            let amp = eval.amp * micro as f32 * scintillation;

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
            source_diagnostics.push(eval.diagnostics);
        }

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
        pulse_diagnostics.push(PulseDiagnostics {
            pulse_index: pulse,
            time_s: t_s,
            source_diagnostics,
        });
    }

    (
        iq,
        profiles,
        compressed_complex,
        states_first,
        pulse_diagnostics,
    )
}
