//! Channel-domain beamformers for the EchoForge radar chain.
//!
//! The beamformers in this module operate on per-channel IQ samples shaped as
//! `channel_iq[ch][sample]` and emit a single combined channel of the same
//! sample length. The contract is intentionally narrow so that downstream
//! code (e.g. detectors, matched filters) can ignore whether the front-end
//! used a single channel or an array.
//!
//! Three flavours ship with this scaffold:
//!
//! * [`SumBeamformer`] — coherent (uniform-weight) sum.
//! * [`DelayAndSumBeamformer`] — applies a per-channel steering vector
//!   before summation, biasing toward the steered direction.
//! * [`CaponUnimplementedBeamformer`] — applies a pre-computed weight vector. This
//!   is an unimplemented adapter: the adaptive Capon weight estimation (sample covariance
//!   inversion, etc.) is intentionally out of scope here; this struct only
//!   pins down the interface so a follow-up packet can fill it in.
//!
//! [`steering_vector`] is a convenience for the common ULA case used by the
//! antenna manifold model.

use num_complex::Complex;

use crate::ComplexSample;

const C_M_PER_S: f64 = 299_792_458.0;

/// Common interface for any beamformer that consumes per-channel IQ and
/// produces a single combined channel.
pub trait Beamformer {
    /// Combine `channel_iq` (indexed `[channel][sample]`) into a single
    /// sample stream of length `channel_iq[0].len()`. If `channel_iq` is
    /// empty the implementation returns an empty vector.
    fn beamform(&self, channel_iq: &[Vec<ComplexSample>]) -> Vec<ComplexSample>;
}

/// Coherent (uniform-weight) sum across channels — the simplest possible
/// combiner. Useful as a baseline reference.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct SumBeamformer;

impl SumBeamformer {
    pub fn new() -> Self {
        Self
    }
}

impl Beamformer for SumBeamformer {
    fn beamform(&self, channel_iq: &[Vec<ComplexSample>]) -> Vec<ComplexSample> {
        let Some(first) = channel_iq.first() else {
            return Vec::new();
        };
        let samples = first.len();
        let mut out = vec![ComplexSample::new(0.0, 0.0); samples];
        for channel in channel_iq {
            let len = channel.len().min(samples);
            for i in 0..len {
                out[i] += channel[i];
            }
        }
        out
    }
}

/// Phase-only (delay-and-sum) beamformer. Each channel sample is multiplied
/// by the conjugate of the per-channel steering vector before summation; on
/// the steered direction the channels add up coherently, off-steering they
/// cancel.
///
/// Construct via [`DelayAndSumBeamformer::new`] or via
/// [`DelayAndSumBeamformer::for_ula`] (which delegates to [`steering_vector`]).
#[derive(Debug, Clone, PartialEq)]
pub struct DelayAndSumBeamformer {
    pub steering_vector: Vec<Complex<f64>>,
}

impl DelayAndSumBeamformer {
    pub fn new(steering_vector: Vec<Complex<f64>>) -> Self {
        Self { steering_vector }
    }

    pub fn for_ula(
        n_elements: usize,
        element_spacing_m: f64,
        frequency_hz: f64,
        target_az_deg: f64,
    ) -> Self {
        Self {
            steering_vector: steering_vector(
                n_elements,
                element_spacing_m,
                frequency_hz,
                target_az_deg,
            ),
        }
    }
}

impl Beamformer for DelayAndSumBeamformer {
    fn beamform(&self, channel_iq: &[Vec<ComplexSample>]) -> Vec<ComplexSample> {
        let Some(first) = channel_iq.first() else {
            return Vec::new();
        };
        let samples = first.len();
        let mut out = vec![ComplexSample::new(0.0, 0.0); samples];
        let channel_count = channel_iq.len();
        let weight_norm = (channel_count.max(1)) as f32;
        for (ch_idx, channel) in channel_iq.iter().enumerate() {
            let weight = self
                .steering_vector
                .get(ch_idx)
                .copied()
                .unwrap_or(Complex::new(1.0, 0.0));
            // Conjugate so that an incoming wavefront matching the
            // steering vector becomes phase-aligned across channels.
            let w_conj = weight.conj();
            let weight_sample = ComplexSample::new(w_conj.re as f32, w_conj.im as f32);
            let len = channel.len().min(samples);
            for i in 0..len {
                out[i] += channel[i] * weight_sample;
            }
        }
        for sample in &mut out {
            sample.re /= weight_norm;
            sample.im /= weight_norm;
        }
        out
    }
}

/// Unimplemented adapter for the Capon (minimum-variance distortionless response) beamformer.
///
/// The full Capon adaptive estimator requires per-frame sample covariance
/// inversion and is left for a follow-up packet. This unimplemented adapter locks in the
/// interface by simply applying a pre-computed weight vector to each
/// channel; callers can supply Capon weights produced offline (or any
/// other adaptive scheme) and exercise the same code path that the
/// production estimator will use.
#[derive(Debug, Clone, PartialEq)]
pub struct CaponUnimplementedBeamformer {
    pub weights: Vec<Complex<f64>>,
}

impl CaponUnimplementedBeamformer {
    pub fn new(weights: Vec<Complex<f64>>) -> Self {
        Self { weights }
    }
}

impl Beamformer for CaponUnimplementedBeamformer {
    fn beamform(&self, channel_iq: &[Vec<ComplexSample>]) -> Vec<ComplexSample> {
        let Some(first) = channel_iq.first() else {
            return Vec::new();
        };
        let samples = first.len();
        let mut out = vec![ComplexSample::new(0.0, 0.0); samples];
        for (ch_idx, channel) in channel_iq.iter().enumerate() {
            let weight = self
                .weights
                .get(ch_idx)
                .copied()
                .unwrap_or(Complex::new(0.0, 0.0));
            let weight_sample = ComplexSample::new(weight.re as f32, weight.im as f32);
            let len = channel.len().min(samples);
            for i in 0..len {
                out[i] += channel[i] * weight_sample;
            }
        }
        out
    }
}

/// Per-element steering vector for a uniform linear array (ULA).
///
/// The vector is built from the geometric phase progression across an
/// `n_elements` ULA spaced `element_spacing_m` apart, looking at
/// `target_az_deg` measured from boresight. Each element has unit
/// magnitude; only the phase changes. Useful as the weight vector for
/// [`DelayAndSumBeamformer`].
pub fn steering_vector(
    n_elements: usize,
    element_spacing_m: f64,
    frequency_hz: f64,
    target_az_deg: f64,
) -> Vec<Complex<f64>> {
    let n = n_elements.max(1);
    let lambda = C_M_PER_S / frequency_hz.max(1e-3);
    let k = 2.0 * std::f64::consts::PI / lambda;
    let sin_theta = target_az_deg.to_radians().sin();
    (0..n)
        .map(|i| {
            let pos = (i as f64) - (n as f64 - 1.0) * 0.5;
            let phase = k * element_spacing_m * pos * sin_theta;
            Complex::new(phase.cos(), phase.sin())
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI as PI32;

    fn channels_from_plane_wave(
        n_channels: usize,
        n_samples: usize,
        spacing_m: f64,
        freq_hz: f64,
        az_deg: f64,
    ) -> Vec<Vec<ComplexSample>> {
        let lambda = C_M_PER_S / freq_hz;
        let k = 2.0 * std::f64::consts::PI / lambda;
        let sin_theta = az_deg.to_radians().sin();
        (0..n_channels)
            .map(|ch| {
                let pos = (ch as f64) - (n_channels as f64 - 1.0) * 0.5;
                let phase = (k * spacing_m * pos * sin_theta) as f32;
                let phasor = ComplexSample::new(phase.cos(), phase.sin());
                vec![phasor; n_samples]
            })
            .collect()
    }

    #[test]
    fn sum_beamformer_handles_empty_input() {
        let bf = SumBeamformer::new();
        let out = bf.beamform(&[]);
        assert!(out.is_empty());
    }

    #[test]
    fn sum_beamformer_adds_channels_sample_wise() {
        let channels = vec![
            vec![
                ComplexSample::new(1.0, 0.0),
                ComplexSample::new(2.0, 1.0),
                ComplexSample::new(0.0, -1.0),
            ],
            vec![
                ComplexSample::new(0.0, 1.0),
                ComplexSample::new(-1.0, 0.0),
                ComplexSample::new(3.0, 4.0),
            ],
            vec![
                ComplexSample::new(0.5, 0.5),
                ComplexSample::new(0.0, 0.0),
                ComplexSample::new(-2.0, -3.0),
            ],
        ];
        let bf = SumBeamformer::new();
        let out = bf.beamform(&channels);
        assert_eq!(out.len(), 3);
        assert!((out[0].re - 1.5).abs() < 1e-6);
        assert!((out[0].im - 1.5).abs() < 1e-6);
        assert!((out[1].re - 1.0).abs() < 1e-6);
        assert!((out[1].im - 1.0).abs() < 1e-6);
        assert!((out[2].re - 1.0).abs() < 1e-6);
        assert!((out[2].im - 0.0).abs() < 1e-6);
    }

    #[test]
    fn delay_and_sum_passes_on_steering_signal() {
        let n_channels = 8;
        let n_samples = 4;
        let spacing = 0.015;
        let freq = 10_000_000_000.0;
        let target = 25.0;

        let on_channels = channels_from_plane_wave(n_channels, n_samples, spacing, freq, target);
        let bf = DelayAndSumBeamformer::for_ula(n_channels, spacing, freq, target);
        let out = bf.beamform(&on_channels);
        let on_mag: f32 = out.iter().map(|c| c.norm()).sum::<f32>() / n_samples as f32;
        assert!(on_mag > 0.95, "on-steering magnitude too low: {on_mag}");
    }

    #[test]
    fn delay_and_sum_rejects_off_steering_signal() {
        let n_channels = 16;
        let n_samples = 4;
        let spacing = 0.015;
        let freq = 10_000_000_000.0;
        let target = 0.0;
        let interferer_az = 30.0;

        let on_channels = channels_from_plane_wave(n_channels, n_samples, spacing, freq, target);
        let off_channels =
            channels_from_plane_wave(n_channels, n_samples, spacing, freq, interferer_az);
        let bf = DelayAndSumBeamformer::for_ula(n_channels, spacing, freq, target);
        let on_out = bf.beamform(&on_channels);
        let off_out = bf.beamform(&off_channels);

        let on_mag: f32 = on_out.iter().map(|c| c.norm()).sum::<f32>() / n_samples as f32;
        let off_mag: f32 = off_out.iter().map(|c| c.norm()).sum::<f32>() / n_samples as f32;
        assert!(
            on_mag > off_mag * 4.0,
            "on={on_mag} should dominate off={off_mag}"
        );
    }

    #[test]
    fn delay_and_sum_empty_input_returns_empty() {
        let bf = DelayAndSumBeamformer::new(vec![Complex::new(1.0, 0.0); 4]);
        let out = bf.beamform(&[]);
        assert!(out.is_empty());
    }

    #[test]
    fn capon_unimplemented_applies_supplied_weights() {
        let weights = vec![
            Complex::new(0.5, 0.0),
            Complex::new(0.0, 0.5),
            Complex::new(-0.5, 0.0),
        ];
        let channels = vec![
            vec![ComplexSample::new(1.0, 0.0)],
            vec![ComplexSample::new(1.0, 0.0)],
            vec![ComplexSample::new(1.0, 0.0)],
        ];
        let bf = CaponUnimplementedBeamformer::new(weights);
        let out = bf.beamform(&channels);
        assert_eq!(out.len(), 1);
        // (1+0j)*0.5 + (1+0j)*0.5j + (1+0j)*(-0.5) = 0.0 + 0.5j
        assert!(out[0].re.abs() < 1e-6, "re={}", out[0].re);
        assert!((out[0].im - 0.5).abs() < 1e-6, "im={}", out[0].im);
    }

    #[test]
    fn capon_unimplemented_handles_empty_input() {
        let bf = CaponUnimplementedBeamformer::new(vec![Complex::new(1.0, 0.0); 2]);
        assert!(bf.beamform(&[]).is_empty());
    }

    #[test]
    fn steering_vector_has_unit_magnitudes() {
        let sv = steering_vector(12, 0.015, 10_000_000_000.0, 17.0);
        assert_eq!(sv.len(), 12);
        for entry in &sv {
            let mag = (entry.re * entry.re + entry.im * entry.im).sqrt();
            assert!((mag - 1.0).abs() < 1e-9, "magnitude {mag} not unit");
        }
    }

    #[test]
    fn steering_vector_zero_az_is_all_ones() {
        let sv = steering_vector(8, 0.015, 10_000_000_000.0, 0.0);
        for entry in &sv {
            assert!((entry.re - 1.0).abs() < 1e-9);
            assert!(entry.im.abs() < 1e-9);
        }
    }

    #[test]
    fn steering_vector_phase_progression_is_linear() {
        let sv = steering_vector(4, 0.015, 10_000_000_000.0, 12.0);
        // Phases should be in arithmetic progression — successive deltas
        // are constant for a ULA. Compare deltas (modulo wrap) within a
        // tight tolerance.
        let phases: Vec<f32> = sv
            .iter()
            .map(|c| (c.im as f32).atan2(c.re as f32))
            .collect();
        let mut deltas: Vec<f32> = phases.windows(2).map(|w| w[1] - w[0]).collect();
        // Unwrap deltas around ±π so wrap-arounds don't fool the diff.
        for d in &mut deltas {
            while *d > PI32 {
                *d -= 2.0 * PI32;
            }
            while *d < -PI32 {
                *d += 2.0 * PI32;
            }
        }
        let reference = deltas[0];
        for d in &deltas {
            assert!(
                (d - reference).abs() < 1e-3,
                "delta {d} differs from reference {reference}"
            );
        }
    }

    #[test]
    fn steering_vector_single_element_returns_unit() {
        let sv = steering_vector(1, 0.015, 10_000_000_000.0, 45.0);
        assert_eq!(sv.len(), 1);
        assert!((sv[0].re - 1.0).abs() < 1e-9);
        assert!(sv[0].im.abs() < 1e-9);
    }
}
