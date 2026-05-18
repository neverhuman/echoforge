use serde::{Deserialize, Serialize};

use crate::ComplexSample;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ReceiverImpairmentProfile {
    pub awgn_sigma: f32,
    pub phase_noise_std_rad: f32,
    pub amplitude_scintillation_sigma: f32,
    pub adc_bits: u8,
    pub clipping_level: f32,
    pub timing_jitter_samples: f32,
    pub dropped_pulse_probability: f32,
    pub gain_imbalance_db: f32,
    pub phase_imbalance_rad: f32,
    pub calibration_drift_db: f32,
}

impl ReceiverImpairmentProfile {
    pub fn public_proxy_default() -> Self {
        Self {
            awgn_sigma: 0.025,
            phase_noise_std_rad: 0.012,
            amplitude_scintillation_sigma: 0.06,
            adc_bits: 12,
            clipping_level: 2.5,
            timing_jitter_samples: 0.15,
            dropped_pulse_probability: 0.002,
            gain_imbalance_db: 0.4,
            phase_imbalance_rad: 0.015,
            calibration_drift_db: 0.25,
        }
    }

    pub fn bounded(self) -> Self {
        Self {
            awgn_sigma: self.awgn_sigma.clamp(0.0, 4.0),
            phase_noise_std_rad: self.phase_noise_std_rad.clamp(0.0, 1.0),
            amplitude_scintillation_sigma: self.amplitude_scintillation_sigma.clamp(0.0, 1.0),
            adc_bits: self.adc_bits.clamp(4, 24),
            clipping_level: self.clipping_level.clamp(0.05, 64.0),
            timing_jitter_samples: self.timing_jitter_samples.clamp(0.0, 8.0),
            dropped_pulse_probability: self.dropped_pulse_probability.clamp(0.0, 1.0),
            gain_imbalance_db: self.gain_imbalance_db.clamp(-12.0, 12.0),
            phase_imbalance_rad: self.phase_imbalance_rad.clamp(-1.0, 1.0),
            calibration_drift_db: self.calibration_drift_db.clamp(-12.0, 12.0),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ReceiverImpairmentSample {
    pub dropped_pulse: bool,
    pub timing_jitter_samples: f32,
    pub gain_scale_i: f32,
    pub gain_scale_q: f32,
    pub phase_offset_rad: f32,
}

pub fn sample_receiver_impairments(
    profile: ReceiverImpairmentProfile,
    seed: u64,
    pulse_index: usize,
) -> ReceiverImpairmentSample {
    let profile = profile.bounded();
    let mut rng = SplitMix64::new(seed ^ (pulse_index as u64).wrapping_mul(0x8fb9_2f3d_5e12_77d5));
    let gain = 10f32.powf(profile.calibration_drift_db / 20.0);
    let imbalance = 10f32.powf(profile.gain_imbalance_db / 20.0);
    ReceiverImpairmentSample {
        dropped_pulse: rng.unit_f32() < profile.dropped_pulse_probability,
        timing_jitter_samples: profile.timing_jitter_samples * (2.0 * rng.unit_f32() - 1.0),
        gain_scale_i: gain * imbalance.sqrt(),
        gain_scale_q: gain / imbalance.sqrt(),
        phase_offset_rad: profile.phase_imbalance_rad
            + profile.phase_noise_std_rad * rng.normal_f32(),
    }
}

pub fn apply_receiver_impairments(
    samples: &mut [ComplexSample],
    profile: ReceiverImpairmentProfile,
    seed: u64,
    pulse_index: usize,
) {
    if samples.is_empty() {
        return;
    }
    let profile = profile.bounded();
    let sample = sample_receiver_impairments(profile, seed, pulse_index);
    if sample.dropped_pulse {
        for value in samples {
            *value = ComplexSample::new(0.0, 0.0);
        }
        return;
    }

    let mut rng = SplitMix64::new(seed ^ 0xa076_1d64_78bd_642f ^ pulse_index as u64);
    let levels = ((1u32 << profile.adc_bits.min(20)) - 1).max(15) as f32;
    let q_step = (2.0 * profile.clipping_level) / levels;
    let phase = ComplexSample::new(sample.phase_offset_rad.cos(), sample.phase_offset_rad.sin());
    let scintillation = (profile.amplitude_scintillation_sigma * rng.normal_f32())
        .exp()
        .clamp(0.25, 4.0);

    for value in samples {
        let mut corrected = ComplexSample::new(
            value.re * sample.gain_scale_i + rng.normal_f32() * profile.awgn_sigma,
            value.im * sample.gain_scale_q + rng.normal_f32() * profile.awgn_sigma,
        ) * phase
            * scintillation;
        corrected.re = quantize_clip(corrected.re, profile.clipping_level, q_step);
        corrected.im = quantize_clip(corrected.im, profile.clipping_level, q_step);
        *value = corrected;
    }
}

fn quantize_clip(value: f32, limit: f32, step: f32) -> f32 {
    (value.clamp(-limit, limit) / step).round() * step
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
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
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
        let theta = 2.0 * std::f32::consts::PI * u2;
        let z0 = radius * theta.cos();
        let z1 = radius * theta.sin();
        self.cached_normal = Some(z1);
        z0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn receiver_impairment_sampling_is_deterministic() {
        let profile = ReceiverImpairmentProfile::public_proxy_default();
        let a = sample_receiver_impairments(profile, 17, 4);
        let b = sample_receiver_impairments(profile, 17, 4);
        assert_eq!(a, b);
        assert!(a.timing_jitter_samples.is_finite());
        assert!(a.phase_offset_rad.is_finite());
    }

    #[test]
    fn receiver_impairment_application_is_finite() {
        let mut a = vec![ComplexSample::new(0.3, -0.2); 16];
        let mut b = a.clone();
        let profile = ReceiverImpairmentProfile::public_proxy_default();
        apply_receiver_impairments(&mut a, profile, 99, 2);
        apply_receiver_impairments(&mut b, profile, 99, 2);
        assert_eq!(a, b);
        assert!(a
            .iter()
            .all(|v| v.re.is_finite() && v.im.is_finite() && v.norm() <= 4.0));
    }
}
