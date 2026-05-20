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
#[path = "beamforming_tests.rs"]
mod tests;
