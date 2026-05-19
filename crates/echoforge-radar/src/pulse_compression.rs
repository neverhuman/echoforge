//! Pulse-compression primitives with optional amplitude weighting.
//!
//! Real radars apply a window function (Taylor, Hamming, Dolph-Chebyshev,
//! etc.) to the matched-filter reference before correlation in order to
//! suppress range sidelobes. Without windowing the compressed peak is a
//! sinc-like response whose first sidelobe is at -13.3 dB; that level is
//! not believable to a radar engineer because it implies CFAR thresholds
//! barely below the main-lobe shoulders.
//!
//! This module provides:
//!   * [`CompressionWindow`] — the supported amplitude tapers.
//!   * [`coefficients`] — closed-form weighting vectors per window kind.
//!   * [`matched_filter`] — naive O(N*M) time-domain matched filter,
//!     byte-stable with the original implementation.
//!   * [`pulse_compress`] — backward-bridged entry point, no window
//!     (returns raw matched-filter output).
//!   * [`pulse_compress_windowed`] — preferred entry point that takes a
//!     [`CompressionWindow`] and applies the taper to the reference
//!     before correlation.
//!
//! References:
//!   - F. Harris, "On the Use of Windows for Harmonic Analysis with the
//!     Discrete Fourier Transform", *Proc. IEEE*, Jan 1978 — survey of
//!     window functions including textbook endpoint values for Hamming
//!     (0.08) and Hann (0.0).
//!   - Carrara, Goodman, Majewski, *Spotlight Synthetic Aperture Radar*,
//!     Artech House 1995, §7.2.4 — Taylor weighting design.
//!   - Dolph, "A Current Distribution for Broadside Arrays Which
//!     Optimizes the Relationship Between Beam Width and Side-Lobe
//!     Level", *Proc. IRE*, June 1946 — Dolph-Chebyshev tapers.
//!   - Skolnik, *Introduction to Radar Systems*, 3rd ed., §6.5 — matched
//!     filtering and the role of amplitude weighting for sidelobe
//!     control.

use std::f64::consts::PI;

use crate::ComplexSample;

/// Amplitude tapers applied to the matched-filter reference before
/// correlation. Real systems prefer a Taylor weighting (typically
/// -35 dB peak sidelobe with `nbar = 4` passband ripples) as a default;
/// Hamming and Hann are kept as well-known textbook recovery options; the
/// Dolph-Chebyshev option lets callers dial in an arbitrary sidelobe
/// level explicitly.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CompressionWindow {
    /// No weighting. Equivalent to the raw matched filter; first sidelobe
    /// of an LFM compressed pulse sits near -13.3 dB.
    None,
    /// Hamming taper. `0.54 - 0.46 cos(2 pi n / (N - 1))`. Endpoint
    /// value 0.08 per Harris 1978.
    Hamming,
    /// Hann (raised-cosine) taper. `0.5 (1 - cos(2 pi n / (N - 1)))`.
    /// Endpoints exactly zero per Harris 1978.
    Hann,
    /// Taylor weighting with -35 dB peak sidelobe target and `nbar`
    /// equi-ripple bands in the design passband (Carrara, Goodman,
    /// Majewski 1995 §7.2.4). Default for [`CompressionWindow::taylor_default`].
    TaylorN35 { nbar: usize },
    /// Taylor weighting with -40 dB peak sidelobe target.
    TaylorN40 { nbar: usize },
    /// Dolph-Chebyshev taper with arbitrary peak sidelobe level (in dB,
    /// negative). All sidelobes are equi-ripple at this level.
    DolphChebyshev { sll_db: f32 },
}

impl CompressionWindow {
    /// The default Taylor-35 with `nbar = 4`. This is the chain-level
    /// default used by [`pulse_compress_windowed`] callers and the
    /// in-tree simulator pipeline (`sim.rs`).
    pub fn taylor_default() -> Self {
        CompressionWindow::TaylorN35 { nbar: 4 }
    }
}

/// Naive O(N*M) time-domain matched filter. The reference is used as-is;
/// callers that want amplitude weighting must pre-window the reference
/// or call [`pulse_compress_windowed`].
///
/// This function is byte-stable with the original pre-windowing
/// implementation so existing callers (CPU backend, GPU backend unimplemented,
/// detector graph tests) continue to produce bit-identical output.
pub fn matched_filter(
    received: &[ComplexSample],
    reference: &[ComplexSample],
) -> Vec<ComplexSample> {
    if received.is_empty() || reference.is_empty() {
        return Vec::new();
    }

    let mut output = vec![ComplexSample::new(0.0, 0.0); received.len() + reference.len() - 1];
    let reference_conj_rev: Vec<_> = reference.iter().rev().map(|sample| sample.conj()).collect();

    for (i, sample) in received.iter().enumerate() {
        for (j, ref_sample) in reference_conj_rev.iter().enumerate() {
            output[i + j] += *sample * *ref_sample;
        }
    }

    output
}

/// Backward-bridged entry point. Applies no window — equivalent to
/// [`matched_filter`]. Use [`pulse_compress_windowed`] for new code; the
/// default chain (`sim.rs`) calls the windowed variant with
/// [`CompressionWindow::taylor_default`].
pub fn pulse_compress(
    received: &[ComplexSample],
    reference: &[ComplexSample],
) -> Vec<ComplexSample> {
    matched_filter(received, reference)
}

/// Pulse-compress `received` against a windowed copy of `reference`.
///
/// The window is applied to the reference as a real-valued multiplicative
/// taper before correlation; the matched filter itself is unchanged. The
/// returned vector has length `received.len() + reference.len() - 1` (same
/// as [`matched_filter`]).
pub fn pulse_compress_windowed(
    received: &[ComplexSample],
    reference: &[ComplexSample],
    window: CompressionWindow,
) -> Vec<ComplexSample> {
    if matches!(window, CompressionWindow::None) {
        return matched_filter(received, reference);
    }
    let coeffs = coefficients(window, reference.len());
    let windowed_reference: Vec<ComplexSample> = reference
        .iter()
        .zip(coeffs.iter())
        .map(|(sample, w)| ComplexSample::new(sample.re * w, sample.im * w))
        .collect();
    matched_filter(received, &windowed_reference)
}

/// Magnitude (|z|) of a complex sample sequence. Returned values are
/// strictly non-negative.
pub fn magnitude(samples: &[ComplexSample]) -> Vec<f32> {
    samples.iter().map(|sample| sample.norm()).collect()
}

/// Real-valued amplitude weights of length `length` for the requested
/// window kind. Edge cases (length 0 or 1) return short trivial vectors
/// rather than panicking so window-aware code can be called on arbitrary
/// reference lengths without guards.
pub fn coefficients(window: CompressionWindow, length: usize) -> Vec<f32> {
    if length == 0 {
        return Vec::new();
    }
    if length == 1 {
        return vec![1.0];
    }
    match window {
        CompressionWindow::None => vec![1.0; length],
        CompressionWindow::Hamming => hamming(length),
        CompressionWindow::Hann => hann(length),
        CompressionWindow::TaylorN35 { nbar } => taylor(length, -35.0, nbar.max(2)),
        CompressionWindow::TaylorN40 { nbar } => taylor(length, -40.0, nbar.max(2)),
        CompressionWindow::DolphChebyshev { sll_db } => {
            dolph_chebyshev(length, sll_db.abs() as f64)
        }
    }
}

fn hamming(length: usize) -> Vec<f32> {
    let n_minus_1 = (length - 1) as f64;
    (0..length)
        .map(|n| (0.54 - 0.46 * (2.0 * PI * (n as f64) / n_minus_1).cos()) as f32)
        .collect()
}

fn hann(length: usize) -> Vec<f32> {
    let n_minus_1 = (length - 1) as f64;
    (0..length)
        .map(|n| (0.5 * (1.0 - (2.0 * PI * (n as f64) / n_minus_1).cos())) as f32)
        .collect()
}

/// Taylor weighting per Carrara, Goodman, Majewski 1995 §7.2.4. The
/// closed form is
///
/// ```text
/// w(n) = 1 + 2 * sum_{m=1}^{nbar-1} F_m * cos(2 pi m (n - (N-1)/2) / N)
/// ```
///
/// with
///
/// ```text
/// A    = acosh(10^(|sll_db|/20)) / pi
/// sigma^2 = nbar^2 / (A^2 + (nbar - 0.5)^2)
/// F_m  = ((-1)^(m+1) / 2) * prod_{n=1}^{nbar-1} (1 - m^2 / (sigma^2 (A^2 + (n - 0.5)^2)))
///        / prod_{n=1, n != m}^{nbar-1} (1 - m^2 / n^2)
/// ```
///
/// This is the standard formulation reproduced in scipy's
/// `signal.windows.taylor`; targeted at peak sidelobe `sll_db` (negative)
/// with `nbar` equi-ripple bands inside the design passband.
fn taylor(length: usize, sll_db: f64, nbar: usize) -> Vec<f32> {
    debug_assert!(sll_db < 0.0, "sll_db must be negative");
    debug_assert!(nbar >= 2, "nbar must be at least 2");

    let r = 10f64.powf(sll_db.abs() / 20.0); // linear sidelobe ratio (> 1)
    let a = r.ln_acosh() / PI; // A = acosh(r) / pi
    let nbar_f = nbar as f64;
    let sigma_sq = nbar_f * nbar_f / (a * a + (nbar_f - 0.5).powi(2));

    // Pre-compute F_m for m = 1 .. nbar-1.
    let mut f_coeffs = vec![0.0f64; nbar.saturating_sub(1)];
    for m in 1..nbar {
        let m_f = m as f64;
        // Numerator product: n = 1 .. nbar-1
        let mut num = 1.0f64;
        for n in 1..nbar {
            let n_f = n as f64;
            num *= 1.0 - (m_f * m_f) / (sigma_sq * (a * a + (n_f - 0.5).powi(2)));
        }
        // Denominator product: n = 1 .. nbar-1, n != m
        let mut den = 1.0f64;
        for n in 1..nbar {
            if n == m {
                continue;
            }
            let n_f = n as f64;
            den *= 1.0 - (m_f * m_f) / (n_f * n_f);
        }
        let sign = if m % 2 == 1 { 1.0 } else { -1.0 }; // (-1)^(m+1)
        f_coeffs[m - 1] = 0.5 * sign * (num / den);
    }

    // Evaluate w(n) on a centered grid. We use the symmetric definition
    // (peak at the center of the window) by referring to n' = n - (N-1)/2.
    let n_f = length as f64;
    let center = (length as f64 - 1.0) / 2.0;
    let mut weights = Vec::with_capacity(length);
    for n in 0..length {
        let np = (n as f64) - center;
        let mut w = 1.0f64;
        for m in 1..nbar {
            w += 2.0 * f_coeffs[m - 1] * (2.0 * PI * (m as f64) * np / n_f).cos();
        }
        weights.push(w as f32);
    }

    // Normalize so the maximum coefficient is 1.0 (cosmetic — the test
    // suite measures sidelobe levels relative to the matched-filter peak
    // so the absolute scale does not matter, but a max of 1.0 makes the
    // peak-SNR-loss accounting straightforward).
    let max = weights
        .iter()
        .copied()
        .fold(f32::MIN, f32::max)
        .max(f32::EPSILON);
    for w in weights.iter_mut() {
        *w /= max;
    }
    weights
}

/// Dolph-Chebyshev weights. Closed-form synthesis via the inverse DFT of
/// the Chebyshev frequency response. The resulting window is symmetric,
/// centred on the middle sample, with peak normalised to 1.
///
/// Closed-form derivation: the window's DTFT magnitude equals
/// `T_{N-1}(beta * cos(omega/2))`, sampled at the DFT bin frequencies
/// `omega_k = 2 pi k / N` for `k = 0..N`, giving
///
/// ```text
/// W_k = T_{N-1}(beta * cos(pi * k / N))
/// ```
///
/// with `beta = cosh(acosh(R) / (N-1))` and sidelobe ratio
/// `R = 10^(|sll_db|/20)`. The window coefficients are the inverse DFT
/// of `W_k` centred so the peak lies at the middle sample:
///
/// ```text
/// w[n] = (1/N) * sum_{k=0}^{N-1} W_k * exp(j * 2 pi k (n - (N-1)/2) / N)
/// ```
///
/// Because `W_k` is real and even-symmetric (`W_k = W_{N-k}`) the
/// imaginary parts cancel and we get a real, symmetric window directly.
/// We compute the DFT by brute force — reference lengths are small
/// (~64 to a few hundred samples) and we avoid pulling in an FFT crate.
fn dolph_chebyshev(length: usize, sll_db_abs: f64) -> Vec<f32> {
    debug_assert!(sll_db_abs > 0.0, "sll_db_abs must be positive");
    debug_assert!(length >= 2);

    let n = length;
    let m = n - 1; // Chebyshev polynomial order
    let r = 10f64.powf(sll_db_abs / 20.0); // sidelobe ratio (> 1)
    // beta = cosh( acosh(r) / m )
    let beta = (r.ln_acosh() / (m as f64)).cosh();

    // Frequency-domain samples T_m(beta * cos(pi * k / N)) for k = 0..N.
    let w_freq: Vec<f64> = (0..n)
        .map(|k| {
            let x = beta * (PI * (k as f64) / (n as f64)).cos();
            chebyshev_t(m, x)
        })
        .collect();

    // Centred inverse DFT. The shift (n_idx - (N-1)/2) puts the peak
    // at the middle of the window. We use the cosine sum form because
    // W_k is real and W_k = W_{N-k} (so the imaginary parts cancel).
    let n_f = n as f64;
    let center = (n as f64 - 1.0) / 2.0;
    let mut w_time = vec![0.0f64; n];
    for n_idx in 0..n {
        let n_shift = (n_idx as f64) - center;
        let mut acc = 0.0f64;
        for k in 0..n {
            acc += w_freq[k] * (2.0 * PI * (k as f64) * n_shift / n_f).cos();
        }
        w_time[n_idx] = acc;
    }

    let max = w_time
        .iter()
        .copied()
        .fold(f64::MIN, f64::max)
        .max(f64::EPSILON);
    w_time.iter().map(|w| (w / max) as f32).collect()
}

/// `T_n(x)` — Chebyshev polynomial of the first kind. Uses the
/// trigonometric/hyperbolic identities so it works for all real `x`.
fn chebyshev_t(n: usize, x: f64) -> f64 {
    if x.abs() <= 1.0 {
        let theta = x.acos();
        ((n as f64) * theta).cos()
    } else if x > 1.0 {
        let theta = x.ln_acosh();
        ((n as f64) * theta).cosh()
    } else {
        // x < -1
        let theta = (-x).ln_acosh();
        let sign = if n % 2 == 0 { 1.0 } else { -1.0 };
        sign * ((n as f64) * theta).cosh()
    }
}

/// Numerically stable `acosh(x)` for `x >= 1`. Named with an `ln_`
/// prefix to make the call sites read as the textbook formula
/// `acosh(x) = ln(x + sqrt(x^2 - 1))`. Stable for `x` up to
/// `~1e100` because we factor out the dominant term.
trait LnAcosh {
    fn ln_acosh(self) -> f64;
}

impl LnAcosh for f64 {
    fn ln_acosh(self) -> f64 {
        if self <= 1.0 {
            0.0
        } else if self < 1e8 {
            (self + (self * self - 1.0).sqrt()).ln()
        } else {
            // Asymptote: acosh(x) ~= ln(2x) for large x.
            (2.0 * self).ln()
        }
    }
}

// =====================================================================
// Tests
// =====================================================================

#[cfg(test)]
#[path = "pulse_compression_tests.rs"]
mod tests;
