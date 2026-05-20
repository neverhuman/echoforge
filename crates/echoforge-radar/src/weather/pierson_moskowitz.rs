//! Pierson-Moskowitz fully-developed wind-sea spectrum and Beaufort
//! classification.
//!
//! Pierson & Moskowitz (1964) proposed a one-dimensional frequency
//! spectrum for ocean gravity waves under a fully developed wind sea
//! (infinite fetch, steady wind):
//!
//! ```text
//! S(ω) = (α · g²) / ω⁵ · exp(-β · (g / (U · ω))⁴)
//! ```
//!
//! with the Phillips constant `α = 8.1 × 10⁻³`, `β = 0.74`, gravitational
//! acceleration `g = 9.81 m/s²`, and `U` the wind speed at 19.5 m above
//! the sea surface. For radar-clutter work the wind speed is conventionally
//! reported at the 10 m reference height (`U_10`); the standard
//! engineering substitution `U ≈ U_10` is adopted here, with the
//! understanding that a small (~5 %) low bias in `H_s` results — within
//! the variance of the published spectral fit.
//!
//! Derived quantities of interest to the radar-chain consumer:
//!
//! - Significant wave height: `H_s = 0.21 · U_10² / g`.
//! - Peak angular frequency: `ω_peak = 0.877 · g / U_10`.
//! - Beaufort force from wind speed: standard table breakpoints.
//!
//! # References
//!
//! - Pierson W. J. & Moskowitz L., *A proposed spectral form for fully
//!   developed wind seas based on the similarity theory of S. A.
//!   Kitaigorodskii*, J. Geophys. Res. 69(24):5181-5190, 1964.
//! - Ochi M. K., *Ocean Waves: The Stochastic Approach*, Cambridge
//!   University Press, 1998 — derivations of `H_s` and `ω_peak` from the
//!   spectrum.
//! - WMO Sea State / Beaufort Wind Force scale (WMO Code Tables) — wind
//!   speed breakpoints used by [`beaufort_force_from_u10`].

/// Gravitational acceleration (m/s²) used by the Pierson-Moskowitz fit.
pub const GRAVITY_M_PER_S2: f64 = 9.81;

/// Phillips constant α (dimensionless) for the Pierson-Moskowitz form.
pub const PHILLIPS_ALPHA: f64 = 8.1e-3;

/// β coefficient inside the high-frequency cutoff exponential.
pub const PM_BETA: f64 = 0.74;

/// Pierson-Moskowitz one-dimensional gravity-wave variance density `S(ω)`
/// (m² · s / rad) evaluated at angular frequency `omega_rad_s` for wind
/// speed `u10_mps` (10-m reference height).
///
/// Closed form (Pierson & Moskowitz 1964 eq. 13):
///
/// ```text
/// S(ω) = α · g² / ω⁵ · exp(-β · (g / (U·ω))⁴)
/// ```
///
/// Returns 0 for non-positive `omega_rad_s` or `u10_mps` so that
/// numerical-integration call sites can integrate over `[0, ∞)` without
/// special-casing the endpoint.
pub fn pierson_moskowitz_spectrum(omega_rad_s: f64, u10_mps: f64) -> f64 {
    if omega_rad_s <= 0.0 || u10_mps <= 0.0 {
        return 0.0;
    }
    let g = GRAVITY_M_PER_S2;
    let head = PHILLIPS_ALPHA * g * g / omega_rad_s.powi(5);
    let ratio = g / (u10_mps * omega_rad_s);
    let exponent = -PM_BETA * ratio.powi(4);
    head * exponent.exp()
}

/// Significant wave height `H_s` (m) under the Pierson-Moskowitz
/// fully-developed-sea fit.
///
/// Integrating the spectrum yields `H_s = 4·sqrt(m_0)` where
/// `m_0 = α·U_10⁴ / (4·β·g²)`. Substituting the published constants gives
/// the engineering form `H_s = 0.21 · U_10² / g` (Ochi 1998 §3.3, eq.
/// 3.39). The numerical constant `0.21` carries the Pierson-Moskowitz
/// integration; we use it directly to avoid round-trip noise from
/// re-deriving `m_0` at each call.
pub fn significant_wave_height_m(u10_mps: f64) -> f64 {
    if u10_mps <= 0.0 {
        return 0.0;
    }
    0.21 * u10_mps * u10_mps / GRAVITY_M_PER_S2
}

/// Angular frequency of the Pierson-Moskowitz spectral peak (rad/s).
///
/// Setting `dS/dω = 0` yields `ω_peak = (4·β/5)^(1/4) · g / U_10`. The
/// numerical prefactor evaluates to ~0.877; we ship the canonical Ochi
/// (1998) figure of `0.877·g / U_10` for direct comparison with published
/// tables.
pub fn peak_frequency_rad_s(u10_mps: f64) -> f64 {
    if u10_mps <= 0.0 {
        return 0.0;
    }
    0.877 * GRAVITY_M_PER_S2 / u10_mps
}

/// Beaufort wind force (0-12) derived from 10-m reference wind speed.
///
/// Breakpoints (m/s) per the WMO standard table:
///
/// | Force | Upper bound U_10 (m/s) | Descriptor |
/// |-------|------------------------|------------|
/// | 0     | 0.3                    | calm       |
/// | 1     | 1.5                    | light air  |
/// | 2     | 3.4                    | light breeze |
/// | 3     | 5.5                    | gentle breeze |
/// | 4     | 8.0                    | moderate breeze |
/// | 5     | 10.8                   | fresh breeze |
/// | 6     | 13.9                   | strong breeze |
/// | 7     | 17.2                   | near gale |
/// | 8     | 20.7                   | gale |
/// | 9     | 24.5                   | strong gale |
/// | 10    | 28.5                   | storm |
/// | 11    | 32.7                   | violent storm |
/// | 12    | ∞                      | hurricane |
pub fn beaufort_force_from_u10(u10_mps: f64) -> u8 {
    const BREAKPOINTS: &[f64] = &[0.3, 1.5, 3.4, 5.5, 8.0, 10.8, 13.9, 17.2, 20.7, 24.5, 28.5, 32.7];
    if u10_mps < 0.0 {
        return 0;
    }
    for (idx, &upper) in BREAKPOINTS.iter().enumerate() {
        if u10_mps <= upper {
            return idx as u8;
        }
    }
    12
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Test 1 — At U_10 = 10 m/s the published Pierson-Moskowitz fit
    /// gives H_s = 0.21·100/9.81 ≈ 2.140 m and ω_peak ≈ 0.877·9.81/10 ≈
    /// 0.860 rad/s. Both must land within 5 %.
    #[test]
    fn pierson_moskowitz_hs_and_peak_at_10_mps() {
        let hs = significant_wave_height_m(10.0);
        let omega = peak_frequency_rad_s(10.0);
        let hs_expected = 2.14;
        let omega_expected = 0.86;
        assert!(
            (hs - hs_expected).abs() / hs_expected < 0.05,
            "H_s at U_10=10 m/s was {hs:.3} m, expected ≈ {hs_expected} (±5 %)"
        );
        assert!(
            (omega - omega_expected).abs() / omega_expected < 0.05,
            "ω_peak at U_10=10 m/s was {omega:.3} rad/s, expected ≈ {omega_expected} (±5 %)"
        );
    }

    /// Test 2 — At U_10 = 15 m/s the fit gives H_s = 0.21·225/9.81 ≈
    /// 4.815 m. Tolerance 5 %.
    #[test]
    fn pierson_moskowitz_hs_at_15_mps() {
        let hs = significant_wave_height_m(15.0);
        let expected = 4.81;
        assert!(
            (hs - expected).abs() / expected < 0.05,
            "H_s at U_10=15 m/s was {hs:.3} m, expected ≈ {expected} (±5 %)"
        );
    }

    /// Test 3 — Beaufort force matches the published breakpoints:
    /// Beaufort(0.0)=0, Beaufort(10.0)=5, Beaufort(20.0)=8, Beaufort(40)=12.
    #[test]
    fn beaufort_breakpoints() {
        assert_eq!(beaufort_force_from_u10(0.0), 0);
        assert_eq!(beaufort_force_from_u10(0.4), 1);
        assert_eq!(beaufort_force_from_u10(2.0), 2);
        assert_eq!(beaufort_force_from_u10(4.0), 3);
        assert_eq!(beaufort_force_from_u10(7.0), 4);
        assert_eq!(beaufort_force_from_u10(10.0), 5);
        assert_eq!(beaufort_force_from_u10(13.0), 6);
        assert_eq!(beaufort_force_from_u10(16.0), 7);
        assert_eq!(beaufort_force_from_u10(20.0), 8);
        assert_eq!(beaufort_force_from_u10(24.0), 9);
        assert_eq!(beaufort_force_from_u10(28.0), 10);
        assert_eq!(beaufort_force_from_u10(32.0), 11);
        assert_eq!(beaufort_force_from_u10(40.0), 12);
    }

    /// Test 4 — The Pierson-Moskowitz spectrum is non-negative for every
    /// `(ω > 0, U > 0)`, and the peak of `S(ω)` at U_10 = 10 m/s falls
    /// near the analytic `ω_peak`. We sample around the peak and check
    /// that `S(ω_peak)` exceeds `S(0.5·ω_peak)` and `S(2·ω_peak)`.
    #[test]
    fn pierson_moskowitz_spectrum_peaks_near_omega_peak() {
        let u10 = 10.0;
        let omega_peak = peak_frequency_rad_s(u10);
        let s_peak = pierson_moskowitz_spectrum(omega_peak, u10);
        let s_below = pierson_moskowitz_spectrum(0.5 * omega_peak, u10);
        let s_above = pierson_moskowitz_spectrum(2.0 * omega_peak, u10);
        assert!(s_peak > 0.0);
        assert!(s_below >= 0.0);
        assert!(s_above >= 0.0);
        assert!(
            s_peak > s_below,
            "expected S(ω_peak)={s_peak:.4} > S(0.5·ω_peak)={s_below:.4}"
        );
        assert!(
            s_peak > s_above,
            "expected S(ω_peak)={s_peak:.4} > S(2·ω_peak)={s_above:.4}"
        );
    }

    /// Test 5 — Zero wind speed must yield a zero spectrum and zero
    /// derived quantities (avoids singularity at U_10 = 0 inside the
    /// (g/(U·ω))⁴ exponent term).
    #[test]
    fn pierson_moskowitz_zero_wind_returns_zero() {
        assert_eq!(pierson_moskowitz_spectrum(1.0, 0.0), 0.0);
        assert_eq!(significant_wave_height_m(0.0), 0.0);
        assert_eq!(peak_frequency_rad_s(0.0), 0.0);
    }
}
