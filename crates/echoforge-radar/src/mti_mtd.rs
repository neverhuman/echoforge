//! Moving Target Indicator (MTI) and Moving Target Detector (MTD)
//! processing for stationary-clutter suppression and coherent Doppler
//! integration.
//!
//! MTI suppresses the zero-Doppler line via binomial-weighted finite
//! impulse response (FIR) cancellers across slow time. MTD chains MTI
//! with a Doppler filter bank (windowed slow-time DFT) so each Doppler
//! bin can be CFAR-tested independently.
//!
//! References:
//!   - Skolnik, *Introduction to Radar Systems*, 3rd ed. (2001), §3.7
//!     (MTI), §3.8 (MTD).
//!   - Richards, *Fundamentals of Radar Signal Processing*, 2nd ed.,
//!     §5.6 (pulse-Doppler / MTD).
//!   - Schleher, *MTI and Pulsed Doppler Radar*, Artech House, 1991,
//!     §2.4 — derivation of MTI improvement factor under Gaussian
//!     clutter spectra.

use std::f64::consts::PI;

use crate::pulse_compression::{coefficients, CompressionWindow};
use crate::sim::slow_time_complex_dft;
use crate::ComplexSample;

/// Binomial-weighted single-delay-line MTI canceller order.
///
/// `Two` is the classical 2-pulse canceller (h = [-1, +1] in slow time).
/// `Three` is 3-pulse (h = [+1, -2, +1]). Higher orders are valid
/// (binomial weights with alternating signs) but uncommon in C-UAS
/// radars where coherent processing intervals (CPIs) are short.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MtiOrder {
    /// 2-pulse canceller. Coefficients [-1, +1] in slow time. Frequency
    /// response |H(f)| = 2|sin(π·f·T_pri)|.
    Two,
    /// 3-pulse canceller. Coefficients [+1, -2, +1] in slow time.
    /// Frequency response |H(f)| = 4·sin²(π·f·T_pri).
    Three,
}

impl MtiOrder {
    /// Binomial cancellation coefficients along slow time (Skolnik §3.7).
    /// The coefficients are written in causal order: index `i` multiplies
    /// the pulse `i` samples behind the current evaluation point.
    pub fn coefficients(self) -> &'static [f32] {
        match self {
            MtiOrder::Two => &[-1.0, 1.0],
            MtiOrder::Three => &[1.0, -2.0, 1.0],
        }
    }

    /// Number of FIR taps used by the canceller (filter order + 1).
    pub fn taps(self) -> usize {
        self.coefficients().len()
    }
}

/// Apply an MTI canceller across slow time (pulse axis) for each range
/// bin.
///
/// Input is a per-pulse complex range profile (`compressed_pulses[k]`
/// holds the range profile for pulse `k`); output is the MTI-filtered
/// complex range profile per pulse with the same shape `[n_pulses;
/// range_len]`. The first `(taps - 1)` output pulses are zero-vectors
/// of the same range length because the FIR convolution has not yet
/// filled its delay line.
///
/// Identical clutter at the same range bin across consecutive pulses
/// cancels to zero. Doppler-shifted targets survive with the binomial
/// frequency response (Skolnik §3.7).
pub fn apply_mti(
    compressed_pulses: &[Vec<ComplexSample>],
    order: MtiOrder,
) -> Vec<Vec<ComplexSample>> {
    let n_pulses = compressed_pulses.len();
    if n_pulses == 0 {
        return Vec::new();
    }
    let range_len = compressed_pulses
        .iter()
        .map(|profile| profile.len())
        .max()
        .unwrap_or(0);
    if range_len == 0 {
        // Preserve shape: return n_pulses empty rows.
        return vec![Vec::new(); n_pulses];
    }

    let taps = order.taps();
    let coeffs = order.coefficients();
    let mut output = vec![vec![ComplexSample::new(0.0, 0.0); range_len]; n_pulses];

    if n_pulses < taps {
        // Delay line never fills: all outputs are zero (already
        // initialized).
        return output;
    }

    // For each output pulse k in [(taps - 1) .. n_pulses - 1]:
    //   output[k][r] = sum_{i=0..taps} coeffs[i] * input[k - (taps - 1) + i][r]
    // With taps = 2, coeffs = [-1, +1] → output[k] = -input[k-1] + input[k]
    // With taps = 3, coeffs = [+1, -2, +1] → output[k] = input[k-2] - 2 input[k-1] + input[k]
    for (k, output_row) in output.iter_mut().enumerate().take(n_pulses).skip(taps - 1) {
        for (r, output_cell) in output_row.iter_mut().enumerate().take(range_len) {
            let mut acc = ComplexSample::new(0.0, 0.0);
            for (i, &c) in coeffs.iter().enumerate() {
                let pulse_idx = k + 1 + i - taps; // k - (taps - 1) + i
                let sample = compressed_pulses[pulse_idx]
                    .get(r)
                    .copied()
                    .unwrap_or(ComplexSample::new(0.0, 0.0));
                acc += sample * c;
            }
            *output_cell = acc;
        }
    }

    output
}

/// MTI improvement factor (dB) on a known-correlation clutter
/// realization with a Gaussian-shaped power spectrum of one-sigma
/// standard deviation `sigma_f_hz`.
///
/// Closed-form: under a Gaussian clutter spectrum with std `σ_f`, the
/// pulse-to-pulse correlation of the clutter return sampled at PRI
/// `T_pri` is
///
/// ```text
/// ρ_n = exp(-2 · (n · π · σ_f · T_pri)²)
/// ```
///
/// (Skolnik §3.7 eq. 3.31; Schleher 1991 §2.4). The improvement factors
/// follow Skolnik §3.7 eq. 3.32:
///
/// ```text
/// I_2 = 1 / (2 · (1 - ρ_1))                                 (2-pulse)
/// I_3 = 1 / (6 · (1 - (4/3)·ρ_1 + (1/3)·ρ_2))               (3-pulse)
/// ```
///
/// `σ_f = 0` (perfectly stationary clutter) gives `ρ_1 = ρ_2 = 1` and
/// the denominator collapses to zero. We saturate to a finite ceiling
/// of 200 dB (well beyond any physical detection floor) so the function
/// is total.
pub fn mti_improvement_factor_db(order: MtiOrder, sigma_f_hz: f64, pri_s: f64) -> f64 {
    // Ceiling for the saturating return when ρ -> 1.
    const SATURATION_DB: f64 = 200.0;

    // Closed-form correlation of a Gaussian-spectrum clutter return
    // sampled at lag n·T_pri.
    let rho = |n: f64| -> f64 {
        let arg = n * PI * sigma_f_hz * pri_s;
        (-2.0 * arg * arg).exp()
    };

    let denom = match order {
        MtiOrder::Two => {
            let rho1 = rho(1.0);
            2.0 * (1.0 - rho1)
        }
        MtiOrder::Three => {
            let rho1 = rho(1.0);
            let rho2 = rho(2.0);
            6.0 * (1.0 - (4.0 / 3.0) * rho1 + (1.0 / 3.0) * rho2)
        }
    };

    if !denom.is_finite() || denom <= 0.0 {
        return SATURATION_DB;
    }

    let i_linear = 1.0 / denom;
    let i_db = 10.0 * i_linear.log10();
    if !i_db.is_finite() || i_db > SATURATION_DB {
        SATURATION_DB
    } else {
        i_db
    }
}

/// Doppler filter bank: take the MTI-filtered slow-time vector at each
/// range bin and run a windowed DFT into `n_doppler` bins.
///
/// The window is applied along the pulse axis before the DFT to control
/// Doppler sidelobe leakage from off-bin targets (Skolnik §3.8;
/// Richards §5.6). The window length matches `mti_pulses.len()` so the
/// taper covers the full coherent processing interval (CPI).
///
/// Output layout is `[range][doppler]`, matching
/// [`slow_time_complex_dft`].
pub fn doppler_filter_bank(
    mti_pulses: &[Vec<ComplexSample>],
    n_doppler: usize,
    window: CompressionWindow,
) -> Vec<Vec<ComplexSample>> {
    let n_pulses = mti_pulses.len();
    if n_pulses == 0 || n_doppler == 0 {
        return Vec::new();
    }

    // No window → straight DFT.
    if matches!(window, CompressionWindow::None) {
        return slow_time_complex_dft(mti_pulses, n_doppler);
    }

    let weights = coefficients(window, n_pulses);
    // Apply the slow-time taper pulse-by-pulse before the DFT.
    let windowed: Vec<Vec<ComplexSample>> = mti_pulses
        .iter()
        .zip(weights.iter())
        .map(|(profile, w)| {
            let w_f32 = *w;
            profile
                .iter()
                .map(|sample| ComplexSample::new(sample.re * w_f32, sample.im * w_f32))
                .collect()
        })
        .collect();

    slow_time_complex_dft(&windowed, n_doppler)
}

/// Full MTD chain: MTI cancellation + Doppler filter bank.
///
/// Equivalent to `doppler_filter_bank(apply_mti(...), ...)`. Provided
/// as a convenience so callers do not need to allocate the intermediate
/// vector themselves.
pub fn mtd_chain(
    compressed_pulses: &[Vec<ComplexSample>],
    mti_order: MtiOrder,
    n_doppler: usize,
    doppler_window: CompressionWindow,
) -> Vec<Vec<ComplexSample>> {
    let mti = apply_mti(compressed_pulses, mti_order);
    doppler_filter_bank(&mti, n_doppler, doppler_window)
}

// =====================================================================
// Tests
// =====================================================================

#[cfg(test)]
#[path = "mti_mtd_tests.rs"]
mod tests;
