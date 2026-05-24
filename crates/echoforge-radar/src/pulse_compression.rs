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

#[path = "pulse_compression_windows.rs"]
mod pulse_compression_windows;
use pulse_compression_windows::{dolph_chebyshev, taylor};

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

// =====================================================================
// Tests
// =====================================================================

#[cfg(test)]
#[path = "pulse_compression_tests.rs"]
mod tests;
