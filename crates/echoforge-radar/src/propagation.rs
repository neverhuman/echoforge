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
    // Strict inequality: a target on the geometric horizon is non-LOS.
    min_target_altitude_for_los_m(antenna_height_m, target_range_m, k_factor) < target_altitude_m
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
    let delta_phi =
        4.0 * std::f64::consts::PI * antenna_height_m * target_altitude_m / (lambda * range_m);
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
    propagation_rain::gas_attenuation_db(
        freq_ghz,
        range_km,
        temperature_k,
        pressure_kpa,
        water_vapor_g_per_m3,
    )
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
    propagation_rain::rain_attenuation_db(freq_ghz, rain_rate_mm_per_hr, range_km, polarization)
}

pub(crate) fn interpolate_linear(x: f64, anchors: &[(f64, f64)]) -> f64 {
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
    propagation_rain::refractivity_n_units(altitude_m)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
#[path = "propagation_tests.rs"]
mod tests;

#[path = "propagation_rain.rs"]
mod propagation_rain;
