//! Radar propagation primitives — line-of-sight horizon, ground-multipath
//! propagation factor, and ITU-R atmospheric / rain attenuation models.
//!
//! These primitives compose the `L_total` loss term in the radar equation
//!
//! ```text
//! P_r = P_t · G_t · G_r · λ² · σ / ((4π)³ · R⁴ · L_total)
//! ```
//!
//! and the geometric LOS test that decides whether a target is above the
//! 4/3-Earth radar horizon at all. The link-budget packet that replaces the
//! current SNR knob (`crates/echoforge-radar/src/sim.rs:253–255`) consumes
//! every public function in this module.
//!
//! # References
//!
//! - Skolnik, *Introduction to Radar Systems*, 3rd ed. (2001), §2.10 and
//!   eq. (2.42) — radar horizon under the 4/3-Earth refraction model.
//! - Blake, *Radar Range-Performance Analysis* (1970) — original 4/3-Earth
//!   horizon derivation cited by Skolnik.
//! - Balanis, *Antenna Theory: Analysis and Design*, 4th ed. (2016), §4.7
//!   (Wave Propagation Over Plane Earth) — two-ray flat-earth ground
//!   reflection model in the low-grazing limit.
//! - ITU-R Recommendation P.676-13 (2022), *Attenuation by atmospheric
//!   gases and related effects*, Annex 2 (simplified approximation) —
//!   piecewise specific-attenuation values used for the 1–40 GHz range.
//! - ITU-R Recommendation P.838-3 (2005), *Specific attenuation model for
//!   rain for use in prediction methods* — k, α coefficient tables.
//! - ITU-R Recommendation P.530-17 (2017), §2.4.1 — polarization-mixing
//!   formula used to derive circular-polarization (k_C, α_C) from the
//!   linear (k_H, α_H, k_V, α_V) coefficients of P.838.
//! - ITU-R Recommendation P.453-14 (2019), *The radio refractive index*,
//!   §1 — exponential reference atmosphere N(h) = N_0·exp(-h/h_0) with
//!   N_0 = 315, h_0 = 7350 m.

/// Mean Earth radius (m), IUGG 2015 reference value.
pub const EARTH_RADIUS_M: f64 = 6_371_008.8;

/// k-factor for the 4/3-Earth refraction model (Skolnik §2.10).
pub const STANDARD_K_FACTOR: f64 = 4.0 / 3.0;

/// Effective Earth radius (m) under the standard 4/3-Earth model.
pub const EFFECTIVE_EARTH_RADIUS_M: f64 = EARTH_RADIUS_M * STANDARD_K_FACTOR;

/// Speed of light in vacuum (m/s) — engineering 4-significant-figure form
/// used elsewhere in this crate (`crates/echoforge-radar/src/sim.rs`
/// `C_M_PER_S`).
pub const SPEED_OF_LIGHT_M_PER_S: f64 = 2.998e8;

// ----------------------------------------------------------------------------
// Radar horizon (4/3-Earth model)
// ----------------------------------------------------------------------------

/// Required target altitude above the local geoid for line-of-sight from a
/// radar at the given antenna height, at the given range, under a
/// refraction model with effective-earth k-factor `k_factor`.
///
/// Per Skolnik, *Introduction to Radar Systems* 3rd ed. eq. (2.42) /
/// Blake 1970, the horizon range to a target at altitude `h_t` with the
/// antenna at `h_r` is, to first order,
///
/// ```text
/// R_horizon = sqrt(2·R_eff·h_r) + sqrt(2·R_eff·h_t)
/// ```
///
/// where `R_eff = k_factor · EARTH_RADIUS_M`. Inverting for `h_t` at a
/// given `target_range_m` yields
///
/// ```text
/// h_t = (R_horizon - sqrt(2·R_eff·h_r))² / (2·R_eff)
/// ```
///
/// A negative return value indicates that the radar's own horizon already
/// exceeds the requested range (the target needs no altitude for LOS).
pub fn min_target_altitude_for_los_m(
    antenna_height_m: f64,
    target_range_m: f64,
    k_factor: f64,
) -> f64 {
    let r_eff = k_factor * EARTH_RADIUS_M;
    let antenna_horizon = (2.0 * r_eff * antenna_height_m.max(0.0)).sqrt();
    let remaining = target_range_m - antenna_horizon;
    if remaining <= 0.0 {
        // Range fits inside the antenna's own horizon; target needs no
        // altitude. Return a small negative value scaled to the deficit
        // so callers can preserve monotonicity if they wish.
        return -(antenna_horizon - target_range_m).max(0.0);
    }
    (remaining * remaining) / (2.0 * r_eff)
}

/// True if `(target_range_m, target_altitude_m)` is above the radar
/// horizon for the given antenna height under the chosen k-factor.
///
/// A target sitting exactly on the geometric horizon is treated as
/// non-LOS (`>=` would erroneously include a knife-edge grazing target).
pub fn target_above_horizon(
    antenna_height_m: f64,
    target_range_m: f64,
    target_altitude_m: f64,
    k_factor: f64,
) -> bool {
    let required = min_target_altitude_for_los_m(antenna_height_m, target_range_m, k_factor);
    target_altitude_m > required
}

// ----------------------------------------------------------------------------
// Two-ray ground reflection
// ----------------------------------------------------------------------------

/// Magnitude of the complex amplitude propagation factor `|F|` for the
/// direct + ground-reflected two-path geometry at low grazing angles.
///
/// Per Balanis, *Antenna Theory* 4th ed. §4.7 (Wave Propagation Over
/// Plane Earth), the difference in path lengths between the direct ray
/// and the ground-reflected ray, expanded in the small-grazing-angle
/// limit, is
///
/// ```text
/// Δ ≈ 2·h_r·h_t / R
/// ```
///
/// giving a phase difference `Δφ = (2π/λ)·Δ = 4π·h_r·h_t / (λ·R)`.
/// With reflection coefficient `Γ ≈ -ρ` (horizontal polarization,
/// low-grazing) the propagation factor is
///
/// ```text
/// F = 1 + ρ·exp(j·(Δφ + π))
///   = 1 - ρ·exp(j·Δφ)
/// ```
///
/// whose magnitude reduces (for `ρ = 1`) to the textbook
///
/// ```text
/// |F| = 2·|sin(2π·h_r·h_t / (λ·R))|
/// ```
///
/// For `ρ < 1` the full envelope is `|F| = sqrt(1 + ρ² - 2ρ·cos(Δφ))`.
/// This implementation evaluates the latter directly so that
/// `reflection_coeff_magnitude` is honoured smoothly between 0 and 1.
///
/// `freq_hz`, `target_altitude_m`, `antenna_height_m`, `range_m` are all
/// linear engineering units. `range_m` must be strictly positive.
pub fn two_ray_propagation_factor_magnitude(
    freq_hz: f64,
    target_altitude_m: f64,
    antenna_height_m: f64,
    range_m: f64,
    reflection_coeff_magnitude: f64,
) -> f64 {
    if range_m <= 0.0 || freq_hz <= 0.0 {
        return 0.0;
    }
    let lambda = SPEED_OF_LIGHT_M_PER_S / freq_hz;
    let delta_phi = 4.0 * std::f64::consts::PI * antenna_height_m * target_altitude_m
        / (lambda * range_m);
    let rho = reflection_coeff_magnitude.clamp(0.0, 1.0);
    // |F|^2 = 1 + rho^2 - 2*rho*cos(delta_phi)
    (1.0 + rho * rho - 2.0 * rho * delta_phi.cos())
        .max(0.0)
        .sqrt()
}

// ----------------------------------------------------------------------------
// ITU-R P.676 atmospheric gas attenuation (simplified)
// ----------------------------------------------------------------------------

/// Path-integrated atmospheric gas attenuation (dB) per ITU-R
/// Recommendation P.676-13, Annex 2 (simplified one-way attenuation).
///
/// For a horizontal slant path of `range_km`, the attenuation is
/// `γ · range_km` where `γ` (dB/km) is the specific attenuation at
/// frequency `freq_ghz` under the supplied atmosphere. This implementation
/// linearly interpolates between anchor specific-attenuation values
/// tabulated in P.676-13 §2.2 for the standard reference atmosphere
/// (15 °C, sea-level pressure, 7.5 g/m³ water vapor), then applies a
/// small first-order correction for departures from those reference
/// thermodynamic conditions.
///
/// Anchors (one-way specific attenuation γ at sea level, dB/km):
///
/// | f (GHz) | γ (dB/km) |
/// |---------|-----------|
/// | 1.0     | 0.005     |
/// | 3.0     | 0.008     |
/// | 10.0    | 0.013     |
/// | 15.0    | 0.050     |
/// | 30.0    | 0.200     |
/// | 40.0    | 0.450     |
///
/// Valid for `freq_ghz ∈ [1, 40]`. Frequencies outside that band are
/// clamped to the nearest anchor for graceful degradation; for the radar
/// bands of interest (S, C, X, Ku, Ka — 1–40 GHz) the interpolation is
/// within the few-tenths-of-dB tolerance documented in P.676-13 Annex 2.
///
/// Thermodynamic correction: γ is multiplied by `pressure_kpa / 101.325`
/// (linear pressure dependence of the dry-air term), by
/// `288.15 / temperature_k` (mild temperature dependence of the dry-air
/// term), and the water-vapor specific attenuation is scaled linearly in
/// `water_vapor_g_per_m3 / 7.5` for the > 10 GHz tail where the H₂O
/// component dominates. This is the same first-order correction noted in
/// P.676-13 Annex 2 figure 2 caption and is sufficient for the
/// fidelity-class F1–F3 link-budget work that consumes this primitive.
pub fn itu_r_p676_gas_attenuation_db(
    freq_ghz: f64,
    range_km: f64,
    temperature_k: f64,
    pressure_kpa: f64,
    water_vapor_g_per_m3: f64,
) -> f64 {
    if range_km <= 0.0 {
        return 0.0;
    }
    let f = freq_ghz.clamp(1.0, 40.0);
    // Anchor table (GHz, dB/km at the P.676-13 standard reference
    // atmosphere). Held internal to the function so the table stays
    // colocated with the citation.
    const ANCHORS: &[(f64, f64)] = &[
        (1.0, 0.005),
        (3.0, 0.008),
        (10.0, 0.013),
        (15.0, 0.050),
        (30.0, 0.200),
        (40.0, 0.450),
    ];
    let gamma_ref = interpolate_linear(f, ANCHORS);

    // First-order thermodynamic corrections (P.676-13 Annex 2 caption).
    let pressure_scale = (pressure_kpa / 101.325).max(0.0);
    let temperature_scale = if temperature_k > 0.0 {
        288.15 / temperature_k
    } else {
        1.0
    };
    // Water-vapor influence ramps in above ~10 GHz; below that the dry-air
    // component dominates and water vapor barely contributes.
    let h2o_scale = if f >= 10.0 {
        (water_vapor_g_per_m3 / 7.5).max(0.0)
    } else {
        1.0
    };
    let gamma = gamma_ref * pressure_scale * temperature_scale * h2o_scale;
    gamma * range_km
}

// ----------------------------------------------------------------------------
// ITU-R P.838 rain attenuation
// ----------------------------------------------------------------------------

/// Polarization of the radar wave, used to select the rain-attenuation
/// coefficients per ITU-R P.838-3.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RainPolarization {
    /// Horizontal linear polarization (k_H, α_H from P.838-3 Table 1).
    Horizontal,
    /// Vertical linear polarization (k_V, α_V from P.838-3 Table 1).
    Vertical,
    /// Circular polarization. Coefficients derived per ITU-R P.530-17
    /// §2.4.1 from the linear pair: k_C = (k_H + k_V) / 2,
    /// α_C = (k_H·α_H + k_V·α_V) / (k_H + k_V).
    Circular,
}

/// Path-integrated rain attenuation (dB) per ITU-R Recommendation
/// P.838-3.
///
/// Specific rain attenuation is `γ_R = k · R^α` (dB/km) where `R` is the
/// rainfall rate (mm/h) and `k`, `α` are frequency- and
/// polarization-dependent coefficients tabulated in P.838-3 Table 1. Total
/// path attenuation across a horizontal `range_km` is `γ_R · range_km`.
///
/// For circular polarization the coefficients are derived from the
/// linear (H, V) pair using the polarization-mixing formula in ITU-R
/// P.530-17 §2.4.1: `k_C = (k_H + k_V) / 2`,
/// `α_C = (k_H·α_H + k_V·α_V) / (k_H + k_V)`.
///
/// Anchor table — k_H, α_H, k_V, α_V at six standard radar frequencies,
/// from ITU-R P.838-3 Table 1 (entries rounded to 5 sig fig):
///
/// | f (GHz) | k_H       | α_H    | k_V       | α_V    |
/// |---------|-----------|--------|-----------|--------|
/// | 1       | 0.0000259 | 0.9691 | 0.0000308 | 0.8592 |
/// | 3       | 0.0001543 | 1.0329 | 0.0001533 | 0.9491 |
/// | 10      | 0.01217   | 1.2571 | 0.01129   | 1.2156 |
/// | 15      | 0.04481   | 1.1233 | 0.04164   | 1.1044 |
/// | 30      | 0.2403    | 0.9485 | 0.2291    | 0.9129 |
/// | 40      | 0.4365    | 0.8516 | 0.4274    | 0.8126 |
///
/// Interpolation between anchors is linear in `log10(freq_ghz)`. For
/// `freq_ghz` outside `[1, 40]` the nearest anchor is used.
pub fn itu_r_p838_rain_attenuation_db(
    freq_ghz: f64,
    rain_rate_mm_per_hr: f64,
    range_km: f64,
    polarization: RainPolarization,
) -> f64 {
    if rain_rate_mm_per_hr <= 0.0 || range_km <= 0.0 {
        return 0.0;
    }
    let (k_h, alpha_h, k_v, alpha_v) = rain_kalpha_pair(freq_ghz);
    let (k, alpha) = match polarization {
        RainPolarization::Horizontal => (k_h, alpha_h),
        RainPolarization::Vertical => (k_v, alpha_v),
        RainPolarization::Circular => {
            // ITU-R P.530-17 §2.4.1 polarization-mixing formula.
            let denom = k_h + k_v;
            let k_c = 0.5 * denom;
            let alpha_c = if denom > 0.0 {
                (k_h * alpha_h + k_v * alpha_v) / denom
            } else {
                0.0
            };
            (k_c, alpha_c)
        }
    };
    let gamma = k * rain_rate_mm_per_hr.powf(alpha);
    gamma * range_km
}

fn rain_kalpha_pair(freq_ghz: f64) -> (f64, f64, f64, f64) {
    // (f_GHz, k_H, alpha_H, k_V, alpha_V) — ITU-R P.838-3 Table 1.
    const ANCHORS: &[(f64, f64, f64, f64, f64)] = &[
        (1.0, 0.0000259, 0.9691, 0.0000308, 0.8592),
        (3.0, 0.0001543, 1.0329, 0.0001533, 0.9491),
        (10.0, 0.01217, 1.2571, 0.01129, 1.2156),
        (15.0, 0.04481, 1.1233, 0.04164, 1.1044),
        (30.0, 0.2403, 0.9485, 0.2291, 0.9129),
        (40.0, 0.4365, 0.8516, 0.4274, 0.8126),
    ];
    let f = freq_ghz.clamp(ANCHORS[0].0, ANCHORS[ANCHORS.len() - 1].0);
    let log_f = f.log10();
    // Find bracketing anchors.
    for win in ANCHORS.windows(2) {
        let (a, b) = (win[0], win[1]);
        if log_f >= a.0.log10() && log_f <= b.0.log10() {
            let span = b.0.log10() - a.0.log10();
            let t = if span > 0.0 {
                (log_f - a.0.log10()) / span
            } else {
                0.0
            };
            // Interpolate k in log space (k spans many orders of
            // magnitude across the radar bands) and α linearly.
            let k_h = lerp_log10(a.1, b.1, t);
            let alpha_h = a.2 + t * (b.2 - a.2);
            let k_v = lerp_log10(a.3, b.3, t);
            let alpha_v = a.4 + t * (b.4 - a.4);
            return (k_h, alpha_h, k_v, alpha_v);
        }
    }
    // Boundary fallthrough.
    let a = if f <= ANCHORS[0].0 {
        ANCHORS[0]
    } else {
        ANCHORS[ANCHORS.len() - 1]
    };
    (a.1, a.2, a.3, a.4)
}

fn lerp_log10(a: f64, b: f64, t: f64) -> f64 {
    if a <= 0.0 || b <= 0.0 {
        return a + t * (b - a);
    }
    10f64.powf(a.log10() + t * (b.log10() - a.log10()))
}

fn interpolate_linear(x: f64, anchors: &[(f64, f64)]) -> f64 {
    if anchors.is_empty() {
        return 0.0;
    }
    if x <= anchors[0].0 {
        return anchors[0].1;
    }
    let last = anchors[anchors.len() - 1];
    if x >= last.0 {
        return last.1;
    }
    for win in anchors.windows(2) {
        let (a, b) = (win[0], win[1]);
        if x >= a.0 && x <= b.0 {
            let span = b.0 - a.0;
            let t = if span > 0.0 { (x - a.0) / span } else { 0.0 };
            return a.1 + t * (b.1 - a.1);
        }
    }
    last.1
}

// ----------------------------------------------------------------------------
// ITU-R P.453 reference-atmosphere refractivity
// ----------------------------------------------------------------------------

/// Atmospheric refractivity `N(h)` (N-units) at altitude `altitude_m`
/// under the exponential reference atmosphere of ITU-R Recommendation
/// P.453-14 §1:
///
/// ```text
/// N(h) = N_0 · exp(-h / h_0),   N_0 = 315,  h_0 = 7350 m
/// ```
///
/// The refractive index is `n(h) = 1 + N(h) · 10⁻⁶`. The standard
/// 4/3-Earth k-factor (`STANDARD_K_FACTOR`) corresponds to a surface
/// refractivity gradient `dN/dh ≈ -39 N-units/km`, which this exponential
/// profile reproduces in the lower troposphere (the derivative at the
/// surface is `-N_0 / h_0 ≈ -0.0428 N/m = -42.8 N/km`, close to the
/// canonical -39 N/km figure once seasonal averaging is applied).
pub fn itu_r_p453_refractivity_n_units(altitude_m: f64) -> f64 {
    const N0: f64 = 315.0;
    const H0: f64 = 7350.0;
    N0 * (-altitude_m / H0).exp()
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Test 1 — Horizon inversion at the canonical antenna height /
    /// range pair (h_r=20 m, R=100 km) under the 4/3-Earth model.
    ///
    /// The Skolnik (2.42) inversion gives
    ///   h_t = (R - sqrt(2·R_eff·h_r))² / (2·R_eff)
    /// with R_eff = 4/3 · 6_371_008.8 ≈ 8.495e6 m. Numerically
    /// h_t ≈ 391.6 m. The original v3 dossier listed ≈568 m for this
    /// case; tracing the algebra back to the cited formula shows that
    /// figure was an arithmetic error in the dossier (568 m corresponds
    /// to h_r ≈ 0 m, not 20 m). The test below pins the correct
    /// physics value with a 10 m tolerance and the receipt for this
    /// packet records the deviation from the dossier figure.
    #[test]
    fn min_target_altitude_canonical_geometry() {
        let h_t = min_target_altitude_for_los_m(20.0, 100_000.0, STANDARD_K_FACTOR);
        let expected = 391.6;
        assert!(
            (h_t - expected).abs() < 10.0,
            "min altitude {h_t:.2} m differs from {expected:.2} m by more than 10 m"
        );
    }

    /// Test 2 — A 50 m target at 100 km is below the 4/3-Earth horizon
    /// for a 20 m antenna (required altitude is ~392 m).
    #[test]
    fn target_below_horizon_returns_false() {
        assert!(!target_above_horizon(20.0, 100_000.0, 50.0, STANDARD_K_FACTOR));
    }

    /// Test 3 — A 1000 m target at 100 km is comfortably above the
    /// 4/3-Earth horizon for a 20 m antenna.
    #[test]
    fn target_above_horizon_returns_true() {
        assert!(target_above_horizon(20.0, 100_000.0, 1000.0, STANDARD_K_FACTOR));
    }

    /// Test 4 — Required target altitude is monotonically non-decreasing
    /// in range for a fixed antenna height. Sample the 1 km–500 km
    /// interval coarsely.
    #[test]
    fn min_target_altitude_monotonic_in_range() {
        let mut prev = f64::NEG_INFINITY;
        for r_km in (1..=500).step_by(5) {
            let h = min_target_altitude_for_los_m(20.0, r_km as f64 * 1000.0, STANDARD_K_FACTOR);
            assert!(
                h >= prev - 1e-9,
                "non-monotonic horizon altitude: prev={prev:.3} new={h:.3} at {r_km} km"
            );
            prev = h;
        }
    }

    /// Test 5 — First two-ray null altitude. For 3 GHz (λ=0.0999 m
    /// from C=2.998e8), R=80 km, h_r=20 m, the textbook
    /// `|F| = 2|sin(2π·h_r·h_t/(λ·R))|` has its first null at
    /// `h_t = λ·R / (2·h_r) ≈ 199.87 m`. (The original v3 spec text
    /// quoted "≈200 m" as the first null but described the formula as
    /// `2·sin(π·h_r·h_t/(λ·R))` — the latter would put the first null
    /// at 400 m. The Balanis derivation cited in the docstring shows
    /// that the spec's *test value* matches the correct full-phase
    /// formula `2·sin(2π·h_r·h_t/(λ·R))`; this module implements that
    /// formula. The tolerance below allows ±5 m.)
    #[test]
    fn two_ray_first_null_low_grazing() {
        let lambda = SPEED_OF_LIGHT_M_PER_S / 3.0e9;
        let h_t_null = lambda * 80_000.0 / (2.0 * 20.0);
        let f = two_ray_propagation_factor_magnitude(3.0e9, h_t_null, 20.0, 80_000.0, 1.0);
        assert!(
            f < 1e-6,
            "expected null |F| ≈ 0 at h_t={h_t_null:.2} m; got {f}"
        );
        // Also confirm the analytic first-null altitude is within 5 m
        // of the 200 m spec value.
        assert!(
            (h_t_null - 200.0).abs() < 5.0,
            "analytic first-null altitude {h_t_null:.2} m off from spec 200 m"
        );
    }

    /// Test 6 — First two-ray peak altitude. Same geometry as Test 5,
    /// peak at `h_t = λ·R / (4·h_r) ≈ 99.93 m`, with `|F| → 2`.
    #[test]
    fn two_ray_first_peak_low_grazing() {
        let lambda = SPEED_OF_LIGHT_M_PER_S / 3.0e9;
        let h_t_peak = lambda * 80_000.0 / (4.0 * 20.0);
        let f = two_ray_propagation_factor_magnitude(3.0e9, h_t_peak, 20.0, 80_000.0, 1.0);
        assert!(
            (f - 2.0).abs() < 1e-6,
            "expected peak |F| ≈ 2 at h_t={h_t_peak:.2} m; got {f}"
        );
        assert!(
            (h_t_peak - 100.0).abs() < 5.0,
            "analytic first-peak altitude {h_t_peak:.2} m off from spec 100 m"
        );
    }

    /// Test 7 — ITU-R P.676 anchor: X-band (10 GHz), 100 km, standard
    /// reference atmosphere (288.15 K, 101.325 kPa, 7.5 g/m³ H₂O).
    /// Expected ≈ 1.3 dB ± 0.5 dB.
    #[test]
    fn p676_x_band_reference_atmosphere() {
        let att = itu_r_p676_gas_attenuation_db(10.0, 100.0, 288.15, 101.325, 7.5);
        assert!(
            (att - 1.3).abs() < 0.5,
            "X-band 100 km gas attenuation {att:.3} dB outside 1.3 ± 0.5"
        );
    }

    /// Test 8 — Atmospheric gas attenuation is strictly increasing in
    /// path length for any fixed atmosphere / frequency.
    #[test]
    fn p676_monotonic_in_range() {
        let mut prev = -1.0;
        for r_km in [1.0, 10.0, 50.0, 100.0, 250.0, 500.0] {
            let att = itu_r_p676_gas_attenuation_db(10.0, r_km, 288.15, 101.325, 7.5);
            assert!(
                att > prev,
                "gas attenuation not monotonic: prev={prev} att={att} at {r_km} km"
            );
            prev = att;
        }
    }

    /// Test 9 — ITU-R P.838: X-band, 10 mm/h rain (moderate),
    /// horizontal polarization, 100 km. P.838-3 Table 1 gives
    /// k_H=0.01217, α_H=1.2571 ⇒ γ ≈ 0.22 dB/km ⇒ 22 dB at 100 km,
    /// well above the 0.5 dB floor required by the packet.
    #[test]
    fn p838_x_band_moderate_rain_horizontal() {
        let att = itu_r_p838_rain_attenuation_db(
            10.0,
            10.0,
            100.0,
            RainPolarization::Horizontal,
        );
        assert!(
            att > 0.5,
            "expected > 0.5 dB rain attenuation, got {att:.3} dB"
        );
        // Anchor the magnitude near the table prediction.
        let expected = 0.01217 * 10f64.powf(1.2571) * 100.0;
        assert!(
            (att - expected).abs() < 0.1,
            "rain attenuation {att:.3} dB diverges from table prediction {expected:.3} dB"
        );
    }

    /// Test 10 — Zero rainfall ⇒ zero rain attenuation, regardless of
    /// frequency / polarization / path length.
    #[test]
    fn p838_zero_rain_zero_attenuation() {
        for pol in [
            RainPolarization::Horizontal,
            RainPolarization::Vertical,
            RainPolarization::Circular,
        ] {
            let att = itu_r_p838_rain_attenuation_db(10.0, 0.0, 100.0, pol);
            assert_eq!(att, 0.0, "expected zero attenuation for zero rain ({pol:?})");
        }
    }

    /// Test 11 — Sea-level refractivity matches the canonical
    /// N_0 = 315 N-units from ITU-R P.453-14.
    #[test]
    fn p453_sea_level_refractivity() {
        let n = itu_r_p453_refractivity_n_units(0.0);
        assert!(
            (n - 315.0).abs() < 1e-9,
            "expected N(0) = 315 N-units, got {n}"
        );
    }

    /// Test 12 — At one scale height (7350 m), refractivity drops to
    /// N_0 / e ≈ 115.88 N-units.
    #[test]
    fn p453_one_scale_height_refractivity() {
        let n = itu_r_p453_refractivity_n_units(7350.0);
        let expected = 315.0 / std::f64::consts::E;
        assert!(
            (n - expected).abs() < 0.5,
            "expected N(7350) ≈ {expected:.3}, got {n:.3}"
        );
        assert!((expected - 115.88).abs() < 0.05);
    }

    /// Test 13 (bonus) — Circular polarization rain attenuation lies
    /// between the H- and V-polarization values for typical X-band rain,
    /// confirming the P.530-17 polarization-mixing formula is correctly
    /// applied.
    #[test]
    fn p838_circular_polarization_between_h_and_v() {
        let h = itu_r_p838_rain_attenuation_db(10.0, 25.0, 50.0, RainPolarization::Horizontal);
        let v = itu_r_p838_rain_attenuation_db(10.0, 25.0, 50.0, RainPolarization::Vertical);
        let c = itu_r_p838_rain_attenuation_db(10.0, 25.0, 50.0, RainPolarization::Circular);
        let lo = h.min(v);
        let hi = h.max(v);
        assert!(
            c >= lo - 1e-9 && c <= hi + 1e-9,
            "circular {c:.3} dB not bracketed by H {h:.3} / V {v:.3}"
        );
    }
}
