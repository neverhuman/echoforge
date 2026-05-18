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
    let mut output =
        vec![vec![ComplexSample::new(0.0, 0.0); range_len]; n_pulses];

    if n_pulses < taps {
        // Delay line never fills: all outputs are zero (already
        // initialized).
        return output;
    }

    // For each output pulse k in [(taps - 1) .. n_pulses - 1]:
    //   output[k][r] = sum_{i=0..taps} coeffs[i] * input[k - (taps - 1) + i][r]
    // With taps = 2, coeffs = [-1, +1] → output[k] = -input[k-1] + input[k]
    // With taps = 3, coeffs = [+1, -2, +1] → output[k] = input[k-2] - 2 input[k-1] + input[k]
    for k in (taps - 1)..n_pulses {
        for r in 0..range_len {
            let mut acc = ComplexSample::new(0.0, 0.0);
            for (i, &c) in coeffs.iter().enumerate() {
                let pulse_idx = k + 1 + i - taps; // k - (taps - 1) + i
                let sample = compressed_pulses[pulse_idx]
                    .get(r)
                    .copied()
                    .unwrap_or(ComplexSample::new(0.0, 0.0));
                acc += sample * c;
            }
            output[k][r] = acc;
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
pub fn mti_improvement_factor_db(
    order: MtiOrder,
    sigma_f_hz: f64,
    pri_s: f64,
) -> f64 {
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
mod tests {
    use super::*;

    fn make_constant_pulses(
        n_pulses: usize,
        range_len: usize,
        value: ComplexSample,
    ) -> Vec<Vec<ComplexSample>> {
        (0..n_pulses).map(|_| vec![value; range_len]).collect()
    }

    /// 2-pulse MTI on a DC clutter sequence (constant complex value
    /// across all pulses at a fixed range bin) must produce numerically
    /// zero output. This is the canonical zero-Doppler cancellation
    /// guarantee from Skolnik §3.7.
    #[test]
    fn mti_two_pulse_cancels_constant_dc() {
        let n_pulses = 16usize;
        let range_len = 4usize;
        let pulses = make_constant_pulses(
            n_pulses,
            range_len,
            ComplexSample::new(1.0, 0.0),
        );
        let out = apply_mti(&pulses, MtiOrder::Two);
        assert_eq!(out.len(), n_pulses);
        // Pulse 0 is the warm-up slot, always zero by construction.
        for k in 1..n_pulses {
            for r in 0..range_len {
                let cell = out[k][r];
                assert!(
                    cell.norm() < 1e-6,
                    "2-pulse MTI residue at pulse {k} range {r}: |{cell}| = {}",
                    cell.norm()
                );
            }
        }
    }

    /// 2-pulse MTI on a maximum-Doppler tone (alternating +1, -1 across
    /// pulses, i.e. f_d = PRF / 2) must double the amplitude.
    /// |H(f)| = 2|sin(π·f·T_pri)| evaluates to 2 at f = 1/(2 T_pri).
    #[test]
    fn mti_two_pulse_passes_high_doppler() {
        let n_pulses = 16usize;
        let range_len = 1usize;
        let mut pulses = Vec::with_capacity(n_pulses);
        for k in 0..n_pulses {
            let sign = if k % 2 == 0 { 1.0_f32 } else { -1.0_f32 };
            pulses.push(vec![ComplexSample::new(sign, 0.0); range_len]);
        }
        let out = apply_mti(&pulses, MtiOrder::Two);
        for k in 1..n_pulses {
            // output[k] = input[k] - input[k-1]
            // alternating signs give magnitude exactly 2.
            let mag = out[k][0].norm();
            assert!(
                (mag - 2.0).abs() < 1e-5,
                "2-pulse MTI gain at pulse {k}: expected 2.0, got {mag}"
            );
        }
    }

    /// 3-pulse MTI on a DC clutter sequence also cancels exactly.
    /// Coefficients [+1, -2, +1] sum to zero, so any constant sequence
    /// must produce zero output.
    #[test]
    fn mti_three_pulse_cancels_dc() {
        let n_pulses = 12usize;
        let range_len = 3usize;
        let pulses = make_constant_pulses(
            n_pulses,
            range_len,
            ComplexSample::new(-0.5, 0.75),
        );
        let out = apply_mti(&pulses, MtiOrder::Three);
        assert_eq!(out.len(), n_pulses);
        // First two pulses are warm-up; output starts at index 2.
        for k in 2..n_pulses {
            for r in 0..range_len {
                let cell = out[k][r];
                assert!(
                    cell.norm() < 1e-5,
                    "3-pulse MTI residue at pulse {k} range {r}: {cell}",
                );
            }
        }
    }

    /// σ_f = 0 → ρ_1 = 1 → denominator → 0. The function must return
    /// the saturating ceiling, not infinity or NaN. The contract is
    /// "improvement factor exceeds any physically meaningful detection
    /// margin"; we test it with a generous floor of 100 dB.
    #[test]
    fn improvement_factor_2_pulse_zero_sigma() {
        let i2 = mti_improvement_factor_db(MtiOrder::Two, 0.0, 900e-6);
        assert!(
            i2 > 100.0,
            "expected improvement factor > 100 dB on σ_f=0 clutter; got {i2}"
        );
        assert!(i2.is_finite(), "improvement factor must saturate to finite ceiling, got {i2}");
        // Same check for 3-pulse.
        let i3 = mti_improvement_factor_db(MtiOrder::Three, 0.0, 900e-6);
        assert!(
            i3 > 100.0,
            "expected 3-pulse improvement factor > 100 dB on σ_f=0 clutter; got {i3}"
        );
        assert!(i3.is_finite());
    }

    /// Canonical Skolnik §3.7 worked example: σ_f = 2 Hz, T_pri = 900 µs
    /// (S-band C-UAS / airport surveillance scenario).
    ///   ρ_1 = exp(-2·(π·2·900e-6)²) ≈ 0.99993632
    ///   I_2 ≈ 1 / (2 · 6.368e-5) ≈ 7,851 ≈ +38.95 dB
    ///
    /// **HEADLINE GATE C10**: assert improvement ≥ 25 dB (well below the
    /// closed-form ≈ 39 dB so we have margin against any rounding /
    /// numeric drift but still gate the expected order of magnitude).
    #[test]
    fn improvement_factor_2_pulse_skolnik_canonical() {
        let sigma_f_hz = 2.0_f64;
        let pri_s = 900e-6_f64;
        let i2 = mti_improvement_factor_db(MtiOrder::Two, sigma_f_hz, pri_s);
        // Closed-form expectation.
        let rho1 = (-2.0_f64 * (PI * sigma_f_hz * pri_s).powi(2)).exp();
        let expected_db = 10.0 * (1.0 / (2.0 * (1.0 - rho1))).log10();
        assert!(
            (i2 - expected_db).abs() < 0.2,
            "I_2 returned {i2} dB vs textbook closed form {expected_db} dB"
        );
        // Headline C10 gate.
        assert!(
            i2 >= 25.0,
            "2-pulse MTI improvement factor under canonical Gaussian \
             clutter = {i2} dB; gate requires >= 25 dB (Skolnik §3.7)"
        );
    }

    /// Full MTD chain: a complex tone at a known Doppler bin must light
    /// up the corresponding bin after MTI + Doppler filter bank. Because
    /// the 2-pulse MTI is a high-pass filter, a tone exactly at f_d =
    /// PRF / N · k_d (with k_d > 0) emerges scaled by |H(f_d)| = 2
    /// sin(π k_d / N) (cancellation passband) and concentrates in
    /// Doppler bin k_d after the DFT.
    #[test]
    fn mtd_chain_pure_tone_lands_in_correct_bin() {
        const N_PULSES: usize = 32;
        const RANGE_LEN: usize = 4;
        const K_D: usize = 7; // arbitrary non-DC Doppler bin
        const TARGET_RANGE: usize = 2;

        let mut pulses: Vec<Vec<ComplexSample>> = Vec::with_capacity(N_PULSES);
        for n in 0..N_PULSES {
            let phase =
                2.0 * std::f32::consts::PI * (K_D as f32) * (n as f32) / N_PULSES as f32;
            let mut profile = vec![ComplexSample::new(0.0, 0.0); RANGE_LEN];
            profile[TARGET_RANGE] = ComplexSample::new(phase.cos(), phase.sin());
            pulses.push(profile);
        }

        let grid = mtd_chain(
            &pulses,
            MtiOrder::Two,
            N_PULSES,
            CompressionWindow::None, // exclude windowing for clean peak
        );

        assert_eq!(grid.len(), RANGE_LEN);
        assert_eq!(grid[TARGET_RANGE].len(), N_PULSES);

        // Find the peak Doppler bin at the target range.
        let (peak_bin, peak_mag) = grid[TARGET_RANGE]
            .iter()
            .enumerate()
            .map(|(k, c)| (k, c.norm()))
            .fold((0usize, 0.0f32), |(best_k, best_m), (k, m)| {
                if m > best_m {
                    (k, m)
                } else {
                    (best_k, best_m)
                }
            });

        assert_eq!(
            peak_bin, K_D,
            "MTD peak landed at Doppler bin {peak_bin}, expected {K_D}"
        );
        // After MTI the spectrum should still be dominated by the
        // injected tone; require the peak to be > 5x the next-largest
        // bin to confirm the bank is concentrating energy correctly.
        let runner_up = grid[TARGET_RANGE]
            .iter()
            .enumerate()
            .filter(|(k, _)| *k != peak_bin)
            .map(|(_, c)| c.norm())
            .fold(0.0_f32, f32::max);
        assert!(
            peak_mag > 5.0 * runner_up,
            "MTD bin {K_D} peak={peak_mag} not dominant over runner-up={runner_up}"
        );

        // Empty range bins must remain numerically zero.
        for range in 0..RANGE_LEN {
            if range == TARGET_RANGE {
                continue;
            }
            for cell in &grid[range] {
                assert!(
                    cell.norm() < 1e-4,
                    "spurious energy {} in empty range {range}",
                    cell.norm()
                );
            }
        }
    }

    /// 3-pulse MTI improvement under the canonical scenario should
    /// exceed the 2-pulse improvement by tens of dB (Skolnik §3.7
    /// table 3.4). At σ_f=2 Hz, T_pri=900 µs the closed-form gives
    /// I_3 ≈ 78 dB.
    #[test]
    fn improvement_factor_3_pulse_dominates_2_pulse_canonical() {
        let i2 = mti_improvement_factor_db(MtiOrder::Two, 2.0, 900e-6);
        let i3 = mti_improvement_factor_db(MtiOrder::Three, 2.0, 900e-6);
        assert!(
            i3 > i2 + 20.0,
            "expected 3-pulse improvement to exceed 2-pulse by >20 dB \
             at canonical scenario; got I_2 = {i2}, I_3 = {i3}"
        );
    }

    /// Doppler filter bank with the default Taylor-35 window must still
    /// peak at the correct Doppler bin (window only suppresses
    /// sidelobes; it does not move the main lobe).
    #[test]
    fn doppler_filter_bank_window_preserves_peak_location() {
        const N_PULSES: usize = 32;
        const K_D: usize = 5;
        let mut pulses: Vec<Vec<ComplexSample>> = Vec::with_capacity(N_PULSES);
        for n in 0..N_PULSES {
            let phase =
                2.0 * std::f32::consts::PI * (K_D as f32) * (n as f32) / N_PULSES as f32;
            pulses.push(vec![ComplexSample::new(phase.cos(), phase.sin())]);
        }
        let grid = doppler_filter_bank(&pulses, N_PULSES, CompressionWindow::taylor_default());
        let (peak_bin, _) = grid[0]
            .iter()
            .enumerate()
            .map(|(k, c)| (k, c.norm()))
            .fold((0usize, 0.0f32), |(best_k, best_m), (k, m)| {
                if m > best_m {
                    (k, m)
                } else {
                    (best_k, best_m)
                }
            });
        assert_eq!(peak_bin, K_D, "window moved the main lobe (now at {peak_bin})");
    }
}
