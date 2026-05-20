//! Barrage noise jamming (Skolnik §11.6).
//!
//! A barrage jammer transmits broadband noise across a band wide enough to
//! cover the victim radar's tuned frequency regardless of where the radar
//! parks its receiver. The jammer-to-receiver link is *one-way* (radar →
//! target → radar is two-way; jammer → radar is one-way), so the received
//! jamming power obeys the standard Friis transmission equation rather
//! than the radar range-equation:
//!
//! ```text
//! P_J = ERP_J · G_r · λ² / ((4π)² · R_J²)
//! ```
//!
//! where `ERP_J` is the jammer's effective radiated power (transmit power
//! × transmit antenna gain), `G_r` is the victim radar antenna gain in the
//! direction of the jammer (assumed to be the main-beam gain unless the
//! caller passes a sidelobe gain), `λ` is the radar wavelength, and `R_J`
//! is the jammer-to-radar slant range.
//!
//! Bandwidth dilution: the jammer spreads its ERP across `B_J` Hz; the
//! radar's matched filter captures only the fraction `B_r / B_J` (radar
//! bandwidth over jammer bandwidth, clamped at 1). The effective jammer
//! noise power inside the radar receiver is therefore `P_J · (B_r / B_J)`.
//! The current model returns `P_J` (the full one-way received jamming
//! power); callers that need the matched-filter-aware J inside the
//! receiver should multiply by the bandwidth ratio themselves — keeping
//! the dilution out of the helper preserves composability with link-budget
//! code that already tracks receiver bandwidth.
//!
//! # References
//!
//! - Skolnik, *Introduction to Radar Systems* 3rd ed. (McGraw-Hill 2001),
//!   §11.6 — noise (barrage and spot) jamming.

/// One barrage noise jammer geometry. ERP is in dBW per the radar EW
/// convention; bandwidth is in Hz; range is in metres.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BarrageJammer {
    /// Effective Radiated Power (transmit power × transmit antenna gain),
    /// in dBW. `dBW = 10·log10(W)`.
    pub erp_dbw: f64,
    /// Total jammer bandwidth (Hz) across which the ERP is spread.
    pub bandwidth_hz: f64,
    /// Slant range from the jammer to the victim radar (m).
    pub jammer_to_radar_range_m: f64,
}

const FOUR_PI_SQUARED: f64 = 4.0 * std::f64::consts::PI * 4.0 * std::f64::consts::PI;

/// One-way received jamming power (W) at the victim radar antenna. Uses
/// the Friis transmission equation, with `victim_antenna_gain_dbi` the
/// radar receive-antenna gain in the direction of the jammer (so callers
/// can pass the main-beam gain when the jammer is in the boresight or the
/// sidelobe gain when the radar is looking away).
///
/// Returns zero when range, wavelength, or bandwidth is non-positive
/// (degenerate input — the caller almost certainly meant "no jammer").
pub fn jammer_received_power_w(
    jammer: &BarrageJammer,
    victim_antenna_gain_dbi: f64,
    wavelength_m: f64,
) -> f64 {
    if jammer.jammer_to_radar_range_m <= 0.0
        || wavelength_m <= 0.0
        || jammer.bandwidth_hz <= 0.0
    {
        return 0.0;
    }
    let erp_w = 10.0_f64.powf(jammer.erp_dbw / 10.0);
    let g_r = 10.0_f64.powf(victim_antenna_gain_dbi / 10.0);
    let range = jammer.jammer_to_radar_range_m;
    erp_w * g_r * wavelength_m * wavelength_m / (FOUR_PI_SQUARED * range * range)
}

/// Jamming-to-Signal ratio (dB) at the victim radar receiver. `J/S =
/// 10·log10(P_J / P_S)` where `P_J` is the one-way received jamming
/// power and `P_S` is the victim's target return power (already supplied
/// by the caller, typically from the radar range-equation in
/// [`crate::link_budget`]).
///
/// Returns `f64::NEG_INFINITY` when the signal power is non-positive
/// (degenerate); returns `f64::INFINITY` when the jamming power is
/// positive and the signal is zero.
pub fn jamming_to_signal_ratio_db(
    jammer: &BarrageJammer,
    signal_power_w: f64,
    victim_antenna_gain_dbi: f64,
    wavelength_m: f64,
) -> f64 {
    let p_j = jammer_received_power_w(jammer, victim_antenna_gain_dbi, wavelength_m);
    if signal_power_w <= 0.0 {
        if p_j > 0.0 {
            return f64::INFINITY;
        }
        return f64::NEG_INFINITY;
    }
    if p_j <= 0.0 {
        return f64::NEG_INFINITY;
    }
    10.0 * (p_j / signal_power_w).log10()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cfar::{ca_cfar_1d, CfarParams};

    #[test]
    fn barrage_friis_equation_units_are_correct() {
        // Sanity: ERP = 50 dBW (1e5 W), range = 100 km, λ = 0.03 m
        // (X-band), antenna gain = 30 dBi (1e3 linear). Compute manually
        // and check:
        //
        //   P_J = ERP · G_r · λ² / ((4π)² · R²)
        //       = 1e5 · 1e3 · 9e-4 / (158 · 1e10)
        //       ≈ 5.7e-8 W.
        let jammer = BarrageJammer {
            erp_dbw: 50.0,
            bandwidth_hz: 100e6,
            jammer_to_radar_range_m: 100_000.0,
        };
        let g_r = 30.0;
        let lambda = 0.03;
        let p_j = jammer_received_power_w(&jammer, g_r, lambda);
        assert!(
            (5.0e-8..=7.0e-8).contains(&p_j),
            "P_J = {p_j} W should be ~5.7e-8 W"
        );
    }

    #[test]
    fn barrage_zero_range_returns_zero_power() {
        let jammer = BarrageJammer {
            erp_dbw: 50.0,
            bandwidth_hz: 100e6,
            jammer_to_radar_range_m: 0.0,
        };
        assert_eq!(jammer_received_power_w(&jammer, 30.0, 0.03), 0.0);
    }

    #[test]
    fn js_ratio_at_plus_20db_degrades_ca_cfar_pd() {
        // Skolnik §11.6 worked-example style: pick a signal level S, then
        // pick the jammer geometry so that the helper reports J/S = +20 dB
        // (J ≈ 100·S). Add J as white noise to a 256-cell power trace
        // around an injected target at S, threshold with CA-CFAR, and
        // assert Pd < 0.5 (Skolnik §11.6 shows ~ -10 dB to -15 dB SNR
        // loss for J/S ≈ +20 dB, which drops Pd from ~ 0.9 to well below
        // 0.5 for any reasonable input SNR).
        //
        // The link math is intentionally abstracted away from the test:
        // we directly evaluate the helper, then degrade a synthetic CUT
        // by the same factor and verify the CFAR consequence. This keeps
        // the test focused on the J/S ratio rather than the full radar
        // link budget.
        let s_w = 1.0e-9; // 1 nW target return — typical.
        let jammer = BarrageJammer {
            erp_dbw: 50.0,
            bandwidth_hz: 100e6,
            // Tune range so J/S ≈ +20 dB. With g_r = 30 dBi, λ = 0.03 m,
            // ERP = 50 dBW: P_J(R₀=100 km) ≈ 5.7e-8 W; P_J scales as
            // (R₀ / R)². For J/S = 100 we need P_J = 1e-7 W, so
            // (100 km / R)² = 1e-7 / 5.7e-8 ≈ 1.754 → R ≈ 75.5 km.
            jammer_to_radar_range_m: 75_500.0,
        };
        let g_r = 30.0;
        let lambda = 0.03;
        let js_db = jamming_to_signal_ratio_db(&jammer, s_w, g_r, lambda);
        assert!(
            (15.0..=25.0).contains(&js_db),
            "expected J/S ≈ +20 dB, got {js_db}"
        );

        // CFAR consequence at J/S = +20 dB. The jamming raises the noise
        // floor by a factor of ~100, so the target-to-noise ratio drops
        // by ~ 20 dB. Build a 256-cell synthetic where the unjammed CUT
        // has SNR = 13 dB (a healthy 1-D CA-CFAR working point). After
        // +20 dB jamming the CUT-to-noise ratio is ~ -7 dB → Pd should
        // collapse well below 0.5.
        let mut hits = 0;
        let trials = 200;
        for trial in 0..trials {
            let mut power = vec![1.0f32; 256];
            // Unjammed CUT at index 128. The CA-CFAR threshold for
            // N = 16, Pfa = 1e-3 is alpha ≈ 8.6, so an unjammed SNR of
            // 20 (~ 13 dB) is comfortably detectable.
            power[128] = 20.0;
            // Apply +20 dB jamming as a multiplicative noise-floor lift
            // of ~ 100x. Tiny SplitMix64 PRNG so the noise is independent
            // per trial but deterministic per seed; avoids pulling in a
            // thread RNG and keeps the test bit-stable.
            let mut state =
                0xC2FA_2D06u64.wrapping_add((trial as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15));
            for cell in power.iter_mut() {
                state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
                let mut z = state;
                z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
                z ^= z >> 31;
                let u = ((z >> 11) as f64 + 0.5) / ((1u64 << 53) as f64);
                let n = -((1.0 - u).ln()) as f32;
                *cell += 100.0 * n;
            }
            let decisions = ca_cfar_1d(&power, CfarParams::new(16, 2, 1e-3));
            if decisions
                .iter()
                .any(|d| d.index == 128 && d.evaluated && d.detected)
            {
                hits += 1;
            }
        }
        let pd = hits as f64 / trials as f64;
        assert!(
            pd < 0.5,
            "CA-CFAR Pd at J/S = +20 dB should be < 0.5; got {pd}"
        );
    }

    #[test]
    fn js_ratio_negative_when_signal_dominates() {
        // Huge target, weak jammer → J/S should be negative.
        let jammer = BarrageJammer {
            erp_dbw: 10.0,
            bandwidth_hz: 100e6,
            jammer_to_radar_range_m: 1_000_000.0,
        };
        let js_db = jamming_to_signal_ratio_db(&jammer, 1.0, 30.0, 0.03);
        assert!(js_db < 0.0, "weak jammer / strong signal should give negative J/S: {js_db}");
    }

    #[test]
    fn js_ratio_handles_zero_signal_power() {
        let jammer = BarrageJammer {
            erp_dbw: 50.0,
            bandwidth_hz: 100e6,
            jammer_to_radar_range_m: 100_000.0,
        };
        let js_db = jamming_to_signal_ratio_db(&jammer, 0.0, 30.0, 0.03);
        assert!(js_db.is_infinite() && js_db > 0.0, "zero signal + nonzero jammer should give +inf: {js_db}");
    }
}
