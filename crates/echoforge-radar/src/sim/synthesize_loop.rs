//! Per-pulse synthesis loop — extracted from synthesize.rs for LOC compliance.

use std::f32::consts::PI;

use crate::clutter::generate_clutter_sequence;
use crate::impairments::apply_receiver_impairments;
use crate::pulse_compression::{magnitude, pulse_compress_windowed, CompressionWindow};
use crate::rfi::sample_rfi_frame;
use crate::scene::SceneDescriptor;
use crate::sim::config::TransientEventKind;
use crate::ComplexSample;

use crate::sim::config::{NoiseProfile, RadarSimConfig, TakeoffProfile};
use crate::sim::episode::{EpisodeSeed, PulseDiagnostics, SplitMix64, TargetState};
use crate::sim::helpers::{micro_doppler_envelope, resolve_entity_state, C_M_PER_S};

pub(super) struct SynthesisLoopOutput {
    pub iq: Vec<Vec<ComplexSample>>,
    pub iq_magnitudes: Vec<Vec<f32>>,
    pub compressed_complex: Vec<Vec<ComplexSample>>,
    pub states_first: Vec<TargetState>,
    pub pulse_diagnostics: Vec<PulseDiagnostics>,
}

/// Run the per-pulse synthesis loop.
pub(super) fn run_synthesis_loop(
    config: &RadarSimConfig,
    noise: &NoiseProfile,
    seed: EpisodeSeed,
    scene: &SceneDescriptor,
    rcs_catalog: &crate::rcs::Rcs,
    glints: &[(usize, f32)],
    first_profile: &TakeoffProfile,
) -> SynthesisLoopOutput {
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
            let (state, range_offset_m) = resolve_entity_state(
                scene.targets.as_slice(),
                entity_idx,
                t_s,
                config.radar_altitude_agl_m,
            );
            let effective_range_m = (state.range_m + range_offset_m).max(0.0);
            let masked_by_receive_window = !config.receive_window_contains(effective_range_m);
            let active = t_s >= entity.spawn_time_s;
            let eval = super::evaluate_entity_link(super::EntityLinkRequest {
                scene,
                config,
                noise,
                first_profile,
                rcs_catalog,
                entity_idx,
                t_s,
                seed: seed.0,
                pulse_index: pulse,
                active,
                masked_by_receive_window,
            });
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

        let clutter_surge = transient_strength(config, t_s, TransientEventKind::ClutterSurge);
        for (index, sample) in received.iter_mut().enumerate() {
            let clutter_raw = match clutter_cube.as_ref() {
                Some(cube) => {
                    cube[pulse * sample_count + index]
                        * noise.clutter_sigma_0_scale
                        * (1.0 + clutter_surge)
                }
                None => {
                    clutter_state = noise.clutter_correlation * clutter_state
                        + (1.0 - noise.clutter_correlation)
                            * rng.normal_scaled(noise.clutter_sigma);
                    clutter_state * (1.0 + clutter_surge)
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

        apply_structured_rfi(config, seed.0, pulse, &mut received);
        apply_transients(config, scene, seed.0, pulse, t_s, &mut received);
        if let Some(profile) = config.receiver_impairment {
            apply_receiver_impairments(&mut received, profile, seed.0 ^ 0x51a9_2b6d, pulse);
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

    SynthesisLoopOutput {
        iq,
        iq_magnitudes: profiles,
        compressed_complex,
        states_first,
        pulse_diagnostics,
    }
}

fn transient_strength(config: &RadarSimConfig, t_s: f64, kind: TransientEventKind) -> f32 {
    config
        .transient_events
        .iter()
        .filter(|event| event.kind == kind && event.active_at(t_s))
        .map(|event| event.bounded_strength())
        .sum::<f32>()
        .clamp(0.0, 16.0)
}

fn apply_structured_rfi(
    config: &RadarSimConfig,
    seed: u64,
    pulse: usize,
    received: &mut [ComplexSample],
) {
    let Some(profile) = config.interference_profile else {
        return;
    };
    if received.is_empty() {
        return;
    }
    let sample = sample_rfi_frame(profile, seed ^ 0x5246_495f, pulse, received.len());
    let profile = profile.bounded();
    let cw_amp = profile.narrowband_cw_power.sqrt();
    let len = received.len() as f32;
    for (idx, value) in received.iter_mut().enumerate() {
        let phase = 2.0 * PI * (idx as f32 / len) * (sample.narrowband_bin.max(1) as f32);
        *value += ComplexSample::new(phase.cos(), phase.sin())
            * (cw_amp + profile.sidelobe_pressure * 0.05);
        if sample.burst_active {
            let burst_phase = 2.0 * PI * ((idx + pulse) as f32 * 0.173).fract();
            *value +=
                ComplexSample::new(burst_phase.cos(), burst_phase.sin()) * profile.burst_amplitude;
        }
    }
}

fn apply_transients(
    config: &RadarSimConfig,
    scene: &SceneDescriptor,
    seed: u64,
    pulse: usize,
    t_s: f64,
    received: &mut [ComplexSample],
) {
    if received.is_empty() {
        return;
    }
    for event in config
        .transient_events
        .iter()
        .filter(|event| event.active_at(t_s))
    {
        let strength = event.bounded_strength();
        match event.kind {
            TransientEventKind::RfiBurst => {
                for (idx, value) in received.iter_mut().enumerate() {
                    let phase =
                        2.0 * PI * ((idx as u64 ^ seed ^ pulse as u64) as f32 * 0.001).fract();
                    *value += ComplexSample::new(phase.cos(), phase.sin()) * strength;
                }
            }
            TransientEventKind::Dropout => {
                let scale = (1.0 - strength.clamp(0.0, 1.0)).max(0.0);
                for value in received.iter_mut() {
                    *value *= scale;
                }
            }
            TransientEventKind::Glint
            | TransientEventKind::MultipathGhost
            | TransientEventKind::WeatherVolume => {
                let bin = event_target_bin(config, scene, event.target_ref, t_s)
                    .unwrap_or(received.len() / 2)
                    .min(received.len() - 1);
                let width = if event.kind == TransientEventKind::WeatherVolume {
                    8usize
                } else {
                    1usize
                };
                for offset in 0..width {
                    let idx = (bin + offset).min(received.len() - 1);
                    let phase = 2.0 * PI * ((pulse + offset) as f32 * 0.137).fract();
                    received[idx] += ComplexSample::new(phase.cos(), phase.sin()) * strength;
                }
            }
            TransientEventKind::ClutterSurge => {}
        }
    }
}

fn event_target_bin(
    config: &RadarSimConfig,
    scene: &SceneDescriptor,
    target_ref: Option<usize>,
    t_s: f64,
) -> Option<usize> {
    let idx = target_ref?;
    if idx >= scene.targets.len() {
        return None;
    }
    let (state, offset) = resolve_entity_state(
        scene.targets.as_slice(),
        idx,
        t_s,
        config.radar_altitude_agl_m,
    );
    let range_m = state.range_m + offset;
    if !config.receive_window_contains(range_m) {
        return None;
    }
    let sample = ((2.0 * range_m.max(0.0) / C_M_PER_S) * config.sample_rate_hz).round();
    Some(sample.max(0.0) as usize)
}
