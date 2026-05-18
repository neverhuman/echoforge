use std::f32::consts::PI;

use serde::{Deserialize, Serialize};

use crate::cfar::{ca_cfar_1d, CfarParams};
use crate::pulse_compression::{magnitude, pulse_compress};
use crate::waveform::LfmChirp;
use crate::ComplexSample;

const C_M_PER_S: f64 = 299_792_458.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct EpisodeSeed(pub u64);

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TakeoffProfile {
    pub initial_range_m: f64,
    pub runway_heading_deg: f64,
    pub ground_speed_mps: f64,
    pub acceleration_mps2: f64,
    pub climb_rate_mps: f64,
    pub max_altitude_m: f64,
    pub radial_velocity_bias_mps: f64,
    pub pitch_jitter_deg: f64,
    pub yaw_jitter_deg: f64,
    pub propulsor_hz: f64,
    pub micro_doppler_hz: f64,
    pub rcs_scalar: f64,
}

impl Default for TakeoffProfile {
    fn default() -> Self {
        Self {
            initial_range_m: 1_450.0,
            runway_heading_deg: 18.0,
            ground_speed_mps: 31.0,
            acceleration_mps2: 1.2,
            climb_rate_mps: 4.5,
            max_altitude_m: 380.0,
            radial_velocity_bias_mps: -18.0,
            pitch_jitter_deg: 1.2,
            yaw_jitter_deg: 1.7,
            propulsor_hz: 95.0,
            micro_doppler_hz: 42.0,
            rcs_scalar: 1.0,
        }
    }
}

impl TakeoffProfile {
    pub fn state_at(&self, t_s: f64) -> TargetState {
        let speed = self.ground_speed_mps + self.acceleration_mps2 * t_s;
        let along_runway_m = self.ground_speed_mps * t_s + 0.5 * self.acceleration_mps2 * t_s * t_s;
        let heading_rad = self.runway_heading_deg.to_radians();
        let cross_range_m = along_runway_m * heading_rad.sin();
        let down_range_m = self.initial_range_m + along_runway_m * heading_rad.cos();
        let altitude_m = (self.climb_rate_mps * t_s).min(self.max_altitude_m);
        let slant_range_m =
            (down_range_m * down_range_m + cross_range_m * cross_range_m + altitude_m * altitude_m)
                .sqrt();
        let radial_velocity_mps = self.radial_velocity_bias_mps + speed * heading_rad.cos() * 0.35;
        let pitch_deg = (altitude_m / self.max_altitude_m.max(1.0) * 7.0)
            + self.pitch_jitter_deg * (2.0 * std::f64::consts::PI * 0.31 * t_s).sin();
        let yaw_deg = self.yaw_jitter_deg * (2.0 * std::f64::consts::PI * 0.17 * t_s + 0.4).sin();
        let propulsor_phase_rad = 2.0 * std::f64::consts::PI * self.propulsor_hz * t_s;

        TargetState {
            time_s: t_s,
            range_m: slant_range_m,
            altitude_m,
            radial_velocity_mps,
            pitch_deg,
            yaw_deg,
            propulsor_phase_rad,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RadarSimConfig {
    pub sample_rate_hz: f64,
    pub pulse_width_s: f64,
    pub bandwidth_hz: f64,
    pub carrier_hz: f64,
    pub pulse_count: usize,
    pub pri_s: f64,
    pub target_snr_db: f64,
    pub cfar_training_cells: usize,
    pub cfar_guard_cells: usize,
    pub cfar_pfa: f32,
}

impl Default for RadarSimConfig {
    fn default() -> Self {
        Self {
            sample_rate_hz: 2_000_000.0,
            pulse_width_s: 128e-6,
            bandwidth_hz: 1_000_000.0,
            carrier_hz: 9_600_000_000.0,
            pulse_count: 32,
            pri_s: 900e-6,
            target_snr_db: 18.0,
            cfar_training_cells: 10,
            cfar_guard_cells: 3,
            cfar_pfa: 1e-3,
        }
    }
}

impl RadarSimConfig {
    pub fn waveform(&self) -> LfmChirp {
        LfmChirp {
            sample_rate_hz: self.sample_rate_hz,
            pulse_width_s: self.pulse_width_s,
            bandwidth_hz: self.bandwidth_hz,
            carrier_hz: 0.0,
            initial_phase_rad: 0.0,
        }
    }

    pub fn cfar_params(&self) -> CfarParams {
        CfarParams::new(
            self.cfar_training_cells,
            self.cfar_guard_cells,
            self.cfar_pfa,
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct NoiseProfile {
    pub awgn_sigma: f32,
    pub phase_noise_std_rad: f32,
    pub amplitude_scintillation_sigma: f32,
    pub rfi_probability: f32,
    pub rfi_amplitude: f32,
    pub clutter_sigma: f32,
    pub clutter_correlation: f32,
    pub ground_glint_count: usize,
    pub ground_glint_amplitude: f32,
}

impl NoiseProfile {
    pub fn real_world_proxy_v1() -> Self {
        Self {
            awgn_sigma: 0.055,
            phase_noise_std_rad: 0.018,
            amplitude_scintillation_sigma: 0.11,
            rfi_probability: 0.006,
            rfi_amplitude: 0.9,
            clutter_sigma: 0.045,
            clutter_correlation: 0.94,
            ground_glint_count: 5,
            ground_glint_amplitude: 0.16,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TargetState {
    pub time_s: f64,
    pub range_m: f64,
    pub altitude_m: f64,
    pub radial_velocity_mps: f64,
    pub pitch_deg: f64,
    pub yaw_deg: f64,
    pub propulsor_phase_rad: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DetectionRecord {
    pub range_bin: usize,
    pub range_m: f64,
    pub statistic: f32,
    pub threshold: f32,
    pub noise_estimate: f32,
    pub confidence: f32,
}

#[derive(Debug, Clone)]
pub struct SyntheticEpisode {
    pub seed: EpisodeSeed,
    pub config: RadarSimConfig,
    pub profile: TakeoffProfile,
    pub noise: NoiseProfile,
    pub target_states: Vec<TargetState>,
    pub iq: Vec<Vec<ComplexSample>>,
    pub range_profiles_by_pulse: Vec<Vec<f32>>,
    pub integrated_range_profile: Vec<f32>,
    pub range_doppler_proxy: Vec<Vec<f32>>,
    pub detections: Vec<DetectionRecord>,
}

#[derive(Debug, Clone)]
struct SplitMix64 {
    state: u64,
    cached_normal: Option<f32>,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self {
            state: seed,
            cached_normal: None,
        }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }

    fn unit_f32(&mut self) -> f32 {
        let bits = (self.next_u64() >> 40) as u32;
        (bits as f32) / ((1u32 << 24) as f32)
    }

    fn normal_f32(&mut self) -> f32 {
        if let Some(value) = self.cached_normal.take() {
            return value;
        }

        let u1 = self.unit_f32().clamp(1e-7, 1.0 - 1e-7);
        let u2 = self.unit_f32().clamp(1e-7, 1.0 - 1e-7);
        let radius = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * PI * u2;
        let z0 = radius * theta.cos();
        let z1 = radius * theta.sin();
        self.cached_normal = Some(z1);
        z0
    }

    fn normal_scaled(&mut self, sigma: f32) -> f32 {
        self.normal_f32() * sigma
    }
}

pub fn synthesize_takeoff_episode(
    config: RadarSimConfig,
    profile: TakeoffProfile,
    noise: NoiseProfile,
    seed: EpisodeSeed,
) -> SyntheticEpisode {
    let waveform = config.waveform();
    let reference = waveform.samples();
    let sample_count = reference.len();
    let compressed_len = sample_count.saturating_mul(2).saturating_sub(1);
    let mut rng = SplitMix64::new(seed.0);
    let mut phase_walk = 0.0f32;
    let target_amp = 10f32.powf((config.target_snr_db as f32) / 20.0)
        * noise.awgn_sigma.max(1e-4)
        * profile.rcs_scalar.max(0.01) as f32;
    let glints = build_ground_glints(sample_count, &noise, &mut rng);

    let mut iq = Vec::with_capacity(config.pulse_count);
    let mut profiles = Vec::with_capacity(config.pulse_count);
    let mut states = Vec::with_capacity(config.pulse_count);
    let mut clutter_state = 0.0f32;

    for pulse in 0..config.pulse_count {
        let t_s = pulse as f64 * config.pri_s;
        let state = profile.state_at(t_s);
        let mut received = vec![ComplexSample::new(0.0, 0.0); sample_count];
        let delay_samples =
            ((2.0 * state.range_m / C_M_PER_S) * config.sample_rate_hz).round() as isize;
        let doppler_hz = 2.0 * state.radial_velocity_mps * config.carrier_hz / C_M_PER_S;
        let pulse_phase = 2.0 * std::f64::consts::PI * doppler_hz * t_s;
        let micro = 1.0
            + 0.15
                * (2.0 * std::f64::consts::PI * profile.micro_doppler_hz * t_s
                    + state.propulsor_phase_rad)
                    .sin();
        let scintillation = (noise.amplitude_scintillation_sigma * rng.normal_f32())
            .exp()
            .clamp(0.4, 2.5);
        let amp = target_amp * micro as f32 * scintillation;

        for (i, sample) in reference.iter().enumerate() {
            let dst = i as isize + delay_samples;
            if dst < 0 || dst >= sample_count as isize {
                continue;
            }
            let phase = pulse_phase as f32 + phase_walk;
            let phasor = ComplexSample::new(phase.cos(), phase.sin());
            received[dst as usize] += *sample * phasor * amp;
        }

        for (index, sample) in received.iter_mut().enumerate() {
            clutter_state = noise.clutter_correlation * clutter_state
                + (1.0 - noise.clutter_correlation) * rng.normal_scaled(noise.clutter_sigma);
            let glint = glints
                .iter()
                .find(|(bin, _)| *bin == index)
                .map(|(_, amp)| *amp)
                .unwrap_or(0.0);
            let clutter = clutter_state + glint;
            sample.re += clutter + rng.normal_scaled(noise.awgn_sigma);
            sample.im += clutter * 0.35 + rng.normal_scaled(noise.awgn_sigma);

            if rng.unit_f32() < noise.rfi_probability {
                let phase = 2.0 * PI * rng.unit_f32();
                *sample += ComplexSample::new(phase.cos(), phase.sin()) * noise.rfi_amplitude;
            }
        }

        phase_walk += rng.normal_scaled(noise.phase_noise_std_rad);
        let compressed = pulse_compress(&received, &reference);
        let mag = magnitude(&compressed);
        iq.push(received);
        profiles.push(mag);
        states.push(state);
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

    SyntheticEpisode {
        seed,
        config,
        profile,
        noise,
        target_states: states,
        iq,
        range_profiles_by_pulse: profiles,
        integrated_range_profile: integrated,
        range_doppler_proxy,
        detections,
    }
}

fn build_ground_glints(
    sample_count: usize,
    noise: &NoiseProfile,
    rng: &mut SplitMix64,
) -> Vec<(usize, f32)> {
    if sample_count == 0 {
        return Vec::new();
    }

    (0..noise.ground_glint_count)
        .map(|_| {
            let bin = (rng.unit_f32() * sample_count as f32) as usize;
            let amp = noise.ground_glint_amplitude * (0.5 + rng.unit_f32());
            (bin.min(sample_count - 1), amp)
        })
        .collect()
}

fn integrate_profiles(profiles: &[Vec<f32>], len: usize) -> Vec<f32> {
    if profiles.is_empty() {
        return Vec::new();
    }

    let mut integrated = vec![0.0f32; len];
    for profile in profiles {
        for (index, value) in profile.iter().enumerate().take(len) {
            integrated[index] += *value;
        }
    }
    let scale = 1.0 / profiles.len() as f32;
    for value in &mut integrated {
        *value *= scale;
    }
    integrated
}

fn slow_time_dft_magnitude(profiles: &[Vec<f32>], range_len: usize) -> Vec<Vec<f32>> {
    let pulses = profiles.len();
    if pulses == 0 || range_len == 0 {
        return Vec::new();
    }

    let mut output = vec![vec![0.0f32; range_len]; pulses];
    for doppler in 0..pulses {
        for range in 0..range_len {
            let mut re = 0.0f32;
            let mut im = 0.0f32;
            for (pulse, profile) in profiles.iter().enumerate() {
                let angle = -2.0 * PI * (doppler as f32) * (pulse as f32) / pulses as f32;
                let value = profile.get(range).copied().unwrap_or(0.0);
                re += value * angle.cos();
                im += value * angle.sin();
            }
            output[doppler][range] = (re * re + im * im).sqrt() / pulses as f32;
        }
    }
    output
}

pub fn range_bin_to_m(index: usize, zero_delay_bin: usize, sample_rate_hz: f64) -> f64 {
    let delay = index as isize - zero_delay_bin as isize;
    if delay <= 0 {
        0.0
    } else {
        (delay as f64) * C_M_PER_S / (2.0 * sample_rate_hz)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_episode_repeats_for_seed() {
        let config = RadarSimConfig {
            pulse_count: 8,
            ..RadarSimConfig::default()
        };
        let a = synthesize_takeoff_episode(
            config,
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
        let config = RadarSimConfig {
            pulse_count: 12,
            target_snr_db: 28.0,
            ..RadarSimConfig::default()
        };
        let mut noise = NoiseProfile::real_world_proxy_v1();
        noise.awgn_sigma = 0.025;
        let episode =
            synthesize_takeoff_episode(config, TakeoffProfile::default(), noise, EpisodeSeed(7));
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
            config,
            TakeoffProfile::default(),
            clean_noise,
            EpisodeSeed(12),
        );
        let dirty_a = synthesize_takeoff_episode(
            config,
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
}
