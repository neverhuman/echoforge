//! Channel-domain beamformers for the EchoForge radar chain.
//!
//! The beamformers in this module operate on per-channel IQ samples shaped as
//! `channel_iq[ch][sample]` and emit a single combined channel of the same
//! sample length. The contract is intentionally narrow so that downstream
//! code (e.g. detectors, matched filters) can ignore whether the front-end
//! used a single channel or an array.
//!
//! Four flavours ship here:
//!
//! * [`SumBeamformer`] — coherent (uniform-weight) sum.
//! * [`DelayAndSumBeamformer`] — applies a per-channel steering vector
//!   before summation, biasing toward the steered direction.
//! * [`StaticWeightBeamformer`] — applies a caller-supplied fixed weight
//!   vector (e.g. weights computed offline).
//! * [`CaponBeamformer`] — the adaptive minimum-variance distortionless
//!   response (MVDR) estimator: it estimates the spatial sample
//!   covariance from the channel snapshots, applies diagonal loading,
//!   and forms `w = R⁻¹a / (aᴴR⁻¹a)` so interference and clutter
//!   off the look direction are nulled while the look direction is
//!   passed distortionless.
//!
//! References: J. Capon, "High-resolution frequency-wavenumber spectrum
//! analysis," Proc. IEEE 57(8), 1969; H. L. Van Trees, *Optimum Array
//! Processing*, Wiley 2002, ch. 6-7; B. D. Carlson, "Covariance matrix
//! estimation errors and diagonal loading in adaptive arrays," IEEE
//! Trans. AES 24(4), 1988.
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

/// Applies a caller-supplied fixed weight vector to each channel. Useful
/// for weights computed offline (Capon, eigen-beamforming, or any other
/// scheme) when adaptive estimation is not wanted at run time.
#[derive(Debug, Clone, PartialEq)]
pub struct StaticWeightBeamformer {
    pub weights: Vec<Complex<f64>>,
}

impl StaticWeightBeamformer {
    pub fn new(weights: Vec<Complex<f64>>) -> Self {
        Self { weights }
    }
}

impl Beamformer for StaticWeightBeamformer {
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

/// Estimate the `n × n` spatial sample covariance `R = (1/K) Σ x xᴴ`
/// from per-channel snapshots `channel_iq[channel][sample]`. `R` is
/// Hermitian positive-semidefinite.
pub fn estimate_covariance(channel_iq: &[Vec<ComplexSample>]) -> Vec<Vec<Complex<f64>>> {
    let n = channel_iq.len();
    let mut r = vec![vec![Complex::new(0.0, 0.0); n]; n];
    if n == 0 {
        return r;
    }
    let snapshots = channel_iq.iter().map(|c| c.len()).min().unwrap_or(0);
    if snapshots == 0 {
        return r;
    }
    for t in 0..snapshots {
        for i in 0..n {
            let xi = Complex::new(channel_iq[i][t].re as f64, channel_iq[i][t].im as f64);
            for j in 0..n {
                let xj = Complex::new(channel_iq[j][t].re as f64, channel_iq[j][t].im as f64);
                r[i][j] += xi * xj.conj();
            }
        }
    }
    let scale = 1.0 / snapshots as f64;
    for row in &mut r {
        for cell in row {
            *cell *= scale;
        }
    }
    r
}

/// Add diagonal loading `R += ε·(tr(R)/n)·I` in place. Loading bounds the
/// inverse when the covariance is rank-deficient (short snapshot support),
/// trading a little white-noise gain for robustness (Carlson 1988).
fn apply_diagonal_loading(r: &mut [Vec<Complex<f64>>], epsilon: f64) {
    let n = r.len();
    if n == 0 {
        return;
    }
    let trace: f64 = (0..n).map(|i| r[i][i].re).sum();
    let load = (epsilon.max(0.0) * trace / n as f64).max(1e-12);
    for (i, row) in r.iter_mut().enumerate() {
        row[i] += Complex::new(load, 0.0);
    }
}

/// Solve the Hermitian positive-definite system `R x = b` via a complex
/// Cholesky factorisation `R = L Lᴴ`. Returns `None` if `R` is not
/// positive-definite (a non-positive pivot is encountered).
pub fn hermitian_solve(r: &[Vec<Complex<f64>>], b: &[Complex<f64>]) -> Option<Vec<Complex<f64>>> {
    let n = r.len();
    if n == 0 || b.len() != n {
        return None;
    }
    let mut l = vec![vec![Complex::new(0.0, 0.0); n]; n];
    for j in 0..n {
        let mut diag = r[j][j].re;
        for k in 0..j {
            diag -= l[j][k].norm_sqr();
        }
        if !(diag > 0.0) {
            return None;
        }
        let ljj = diag.sqrt();
        l[j][j] = Complex::new(ljj, 0.0);
        for i in (j + 1)..n {
            let mut s = r[i][j];
            for k in 0..j {
                s -= l[i][k] * l[j][k].conj();
            }
            l[i][j] = s / ljj;
        }
    }
    // Forward solve L y = b.
    let mut y = vec![Complex::new(0.0, 0.0); n];
    for i in 0..n {
        let mut s = b[i];
        for k in 0..i {
            s -= l[i][k] * y[k];
        }
        y[i] = s / l[i][i].re;
    }
    // Backward solve Lᴴ x = y.
    let mut x = vec![Complex::new(0.0, 0.0); n];
    for i in (0..n).rev() {
        let mut s = y[i];
        for k in (i + 1)..n {
            s -= l[k][i].conj() * x[k];
        }
        x[i] = s / l[i][i].re;
    }
    Some(x)
}

/// Adaptive minimum-variance distortionless-response (MVDR / Capon)
/// beamformer. The weight vector `w = R⁻¹a / (aᴴR⁻¹a)` minimises output
/// power subject to `wᴴa = 1`, so a signal from the look direction is
/// passed undistorted while interference is nulled.
#[derive(Debug, Clone, PartialEq)]
pub struct CaponBeamformer {
    /// Look-direction steering vector `a`.
    pub steering_vector: Vec<Complex<f64>>,
    /// Diagonal-loading fraction `ε` of `tr(R)/n`.
    pub diagonal_loading: f64,
}

impl CaponBeamformer {
    pub fn new(steering_vector: Vec<Complex<f64>>, diagonal_loading: f64) -> Self {
        Self {
            steering_vector,
            diagonal_loading,
        }
    }

    /// Construct for a uniform linear array looking at `target_az_deg`.
    pub fn for_ula(
        n_elements: usize,
        element_spacing_m: f64,
        frequency_hz: f64,
        target_az_deg: f64,
        diagonal_loading: f64,
    ) -> Self {
        Self {
            steering_vector: steering_vector(
                n_elements,
                element_spacing_m,
                frequency_hz,
                target_az_deg,
            ),
            diagonal_loading,
        }
    }

    /// Compute the MVDR weight vector from channel snapshots. Falls back
    /// to the (normalised) steering vector when the loaded covariance is
    /// not positive-definite — a graceful conventional-beamformer
    /// degradation rather than a NaN.
    pub fn weights(&self, channel_iq: &[Vec<ComplexSample>]) -> Vec<Complex<f64>> {
        let n = channel_iq.len();
        let a = self.aligned_steering(n);
        if n == 0 {
            return a;
        }
        let mut r = estimate_covariance(channel_iq);
        apply_diagonal_loading(&mut r, self.diagonal_loading);
        match hermitian_solve(&r, &a) {
            Some(u) => {
                // denom = aᴴ u  (real positive for Hermitian PD R).
                let mut denom = Complex::new(0.0, 0.0);
                for i in 0..n {
                    denom += a[i].conj() * u[i];
                }
                if denom.norm() < 1e-30 {
                    return conventional_weights(&a);
                }
                u.iter().map(|&ui| ui / denom).collect()
            }
            None => conventional_weights(&a),
        }
    }

    fn aligned_steering(&self, n: usize) -> Vec<Complex<f64>> {
        (0..n)
            .map(|i| {
                self.steering_vector
                    .get(i)
                    .copied()
                    .unwrap_or(Complex::new(1.0, 0.0))
            })
            .collect()
    }
}

/// Conventional (delay-and-sum) weights normalised to `wᴴa = 1`.
fn conventional_weights(a: &[Complex<f64>]) -> Vec<Complex<f64>> {
    let norm: f64 = a.iter().map(|c| c.norm_sqr()).sum::<f64>().max(1e-30);
    a.iter().map(|&ai| ai / norm).collect()
}

impl Beamformer for CaponBeamformer {
    fn beamform(&self, channel_iq: &[Vec<ComplexSample>]) -> Vec<ComplexSample> {
        let Some(first) = channel_iq.first() else {
            return Vec::new();
        };
        let samples = first.len();
        let weights = self.weights(channel_iq);
        let mut out = vec![ComplexSample::new(0.0, 0.0); samples];
        // y[t] = wᴴ x[t].
        for (ch_idx, channel) in channel_iq.iter().enumerate() {
            let w = weights
                .get(ch_idx)
                .copied()
                .unwrap_or(Complex::new(0.0, 0.0));
            let wc = w.conj();
            let weight_sample = ComplexSample::new(wc.re as f32, wc.im as f32);
            let len = channel.len().min(samples);
            for i in 0..len {
                out[i] += channel[i] * weight_sample;
            }
        }
        out
    }
}

/// Capon spatial power estimate `P = 1 / (aᴴ R⁻¹ a)` for a look-direction
/// steering vector `a`. The classic high-resolution direction-of-arrival
/// spectrum (Capon 1969). Returns `0.0` if the covariance is singular.
pub fn capon_spectrum(
    channel_iq: &[Vec<ComplexSample>],
    steering: &[Complex<f64>],
    diagonal_loading: f64,
) -> f64 {
    let n = channel_iq.len();
    if n == 0 || steering.len() != n {
        return 0.0;
    }
    let mut r = estimate_covariance(channel_iq);
    apply_diagonal_loading(&mut r, diagonal_loading);
    match hermitian_solve(&r, steering) {
        Some(u) => {
            let mut denom = Complex::new(0.0, 0.0);
            for i in 0..n {
                denom += steering[i].conj() * u[i];
            }
            let d = denom.re.max(1e-30);
            1.0 / d
        }
        None => 0.0,
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
