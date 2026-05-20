//! ITU-R P.453 modified refractivity (M-units) and linear refractivity
//! profile primitives.
//!
//! Atmospheric refractivity is conventionally expressed in two related
//! units. The plain *refractivity* `N(h)` (N-units) is defined as
//!
//! ```text
//! N(h) = (n(h) - 1) × 10⁶
//! ```
//!
//! where `n(h)` is the radio refractive index at altitude `h`. The
//! *modified refractivity* `M(h)` (M-units) folds in the curvature of the
//! Earth so that horizontal rays in a hypothetical flat-Earth atmosphere
//! correspond to constant-`M` surfaces:
//!
//! ```text
//! M(h) = N(h) + 0.157 · h     (h in metres)
//! ```
//!
//! The coefficient `0.157 m⁻¹` comes from `10⁶ / R_E` with the mean Earth
//! radius `R_E ≈ 6 371 km`. A negative `dM/dh` gradient signals a duct:
//! the ray bends faster than the Earth curves and trapping becomes
//! possible. This is the canonical "M-unit" working frame used by every
//! ducting-aware propagation tool (AREPS, PETOOL, the Skolnik §2.10
//! anomalous-propagation worked examples).
//!
//! This module exposes three small functions:
//!
//! 1. [`modified_refractivity_m_units`] — the algebraic conversion
//!    `M = N + 0.157·h` given a known `N` value.
//! 2. [`refractivity_profile`] — a linear `N(h)` profile parameterised by
//!    surface refractivity and a per-kilometre gradient. The linear form
//!    is the standard first-order approximation used in ITU-R P.453 for
//!    the lower troposphere and matches the layered profiles consumed by
//!    [`super::duct_detection`] and [`super::ducting_horizon`].
//! 3. [`modified_refractivity_profile`] — composes the two so a caller
//!    can ask for `M(h)` directly from `(N₀, dN/dh, h)`.
//!
//! # References
//!
//! - ITU-R Recommendation P.453-14 (2019), *The radio refractive index:
//!   its formula and refractivity data*, §1 and §2 — definitions of
//!   `N(h)` and `M(h)`, surface refractivity `N₀`, reference gradient
//!   `dN/dh ≈ -39 N/km` for the standard atmosphere.
//! - Skolnik, *Introduction to Radar Systems*, 3rd ed. (2001), §2.10
//!   eq. (2.40)–(2.42) — modified refractivity formulation and the
//!   `0.157 m⁻¹` Earth-curvature coefficient.
//! - Hitney & Vieth (1990), *Statistical Assessment of Evaporation Duct
//!   Propagation*, IEEE Trans. Ant. Prop. 38(6) — operational M-unit
//!   profiles for marine surface and evaporation ducts.

/// Coefficient `10⁶ / R_E` (per metre) converting the Earth-curvature
/// term to N-units of refractivity. `R_E ≈ 6 371 km` ⇒ `0.157 m⁻¹`.
pub const EARTH_CURVATURE_COEFFICIENT_PER_M: f64 = 0.157;

/// Convert a refractivity value `N` (N-units) at altitude `height_m` to
/// modified refractivity `M` (M-units) per ITU-R P.453-14 §2:
///
/// ```text
/// M(h) = N(h) + 0.157 · h     (h in metres)
/// ```
///
/// Constant-`M` surfaces correspond to horizontal rays in the equivalent
/// flat-Earth picture, so `dM/dh < 0` signals trapping (a duct), and
/// `dM/dh = 0` is the threshold for total trapping of a horizontal ray.
pub fn modified_refractivity_m_units(n_units: f64, height_m: f64) -> f64 {
    n_units + EARTH_CURVATURE_COEFFICIENT_PER_M * height_m
}

/// Linear refractivity profile `N(h) = N₀ + (dN/dh)·h` evaluated at
/// `height_m`, with the gradient supplied in N-units per kilometre.
///
/// The linear form is the first-order Taylor approximation of the
/// exponential P.453 reference atmosphere about the surface and is the
/// idealisation under which the duct-detection rules
/// (`gradient < -157 N/km` ⇒ trapping; `gradient ∈ [-157, -40]` ⇒
/// super-refractive; `gradient ≥ -40` ⇒ standard) are stated. It is
/// adequate for the lower troposphere where ducting work happens
/// (typical layers a few hundred metres thick).
pub fn refractivity_profile(n_at_msl: f64, gradient_n_per_km: f64, height_m: f64) -> f64 {
    let height_km = height_m / 1000.0;
    n_at_msl + gradient_n_per_km * height_km
}

/// Modified refractivity `M(h)` from the same `(N₀, dN/dh, h)` triple
/// as [`refractivity_profile`]. Convenience composition equivalent to
/// `modified_refractivity_m_units(refractivity_profile(...), h)`.
pub fn modified_refractivity_profile(
    n_at_msl: f64,
    gradient_n_per_km: f64,
    height_m: f64,
) -> f64 {
    let n_h = refractivity_profile(n_at_msl, gradient_n_per_km, height_m);
    modified_refractivity_m_units(n_h, height_m)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// At the surface, `M(0) = N(0)` because the curvature term vanishes.
    #[test]
    fn modified_refractivity_at_surface_equals_n() {
        let m = modified_refractivity_m_units(315.0, 0.0);
        assert!(
            (m - 315.0).abs() < 1e-9,
            "M(0) should equal N(0)=315, got {m}"
        );
    }

    /// Spec gate: at h = 1000 m with N(1000) = 260, M = 260 + 157 = 417
    /// (±1 M-unit tolerance).
    #[test]
    fn modified_refractivity_at_1km_standard_atmosphere() {
        let m = modified_refractivity_m_units(260.0, 1000.0);
        assert!(
            (m - 417.0).abs() < 1.0,
            "M(1000) for N=260 should be 417 ± 1, got {m}"
        );
    }

    /// Linear `N(h)` profile evaluated at 0 m returns the surface value.
    #[test]
    fn refractivity_profile_at_surface_returns_n0() {
        let n = refractivity_profile(315.0, -40.0, 0.0);
        assert!((n - 315.0).abs() < 1e-9, "expected N(0) = 315, got {n}");
    }

    /// Standard atmosphere linear profile (N₀ = 300, gradient = -40
    /// N/km) at 1 km yields N(1000) = 300 + (-40) = 260, matching the
    /// gate value used in the M-unit test above.
    #[test]
    fn refractivity_profile_linear_one_km() {
        let n = refractivity_profile(300.0, -40.0, 1000.0);
        assert!((n - 260.0).abs() < 1e-9, "expected N(1000) = 260, got {n}");
    }

    /// Composed profile reproduces the M = 417 result directly from
    /// `(N₀=300, dN/dh=-40, h=1000 m)`.
    #[test]
    fn modified_refractivity_profile_composition() {
        let m = modified_refractivity_profile(300.0, -40.0, 1000.0);
        assert!(
            (m - 417.0).abs() < 1.0,
            "M(1000) from composed profile should be 417 ± 1, got {m}"
        );
    }

    /// Trapping threshold: a layer with `dN/dh = -157 N/km` has
    /// `dM/dh = 0` because the Earth-curvature coefficient `0.157 m⁻¹`
    /// in M-units corresponds to exactly `+157 N/km`. Verify by
    /// evaluating M at h = 0 and h = 100 m and confirming the
    /// difference is within 0.5 M-unit of zero.
    #[test]
    fn trapping_threshold_flat_m_profile() {
        let m0 = modified_refractivity_profile(315.0, -157.0, 0.0);
        let m100 = modified_refractivity_profile(315.0, -157.0, 100.0);
        assert!(
            (m100 - m0).abs() < 0.5,
            "dN/dh = -157 N/km should yield flat M-profile; got ΔM = {}",
            m100 - m0
        );
    }

    /// Strong trapping gradient (dN/dh = -300 N/km) yields a *decreasing*
    /// M-profile, the operational duct signature.
    #[test]
    fn strong_trapping_decreasing_m_profile() {
        let m0 = modified_refractivity_profile(330.0, -300.0, 0.0);
        let m100 = modified_refractivity_profile(330.0, -300.0, 100.0);
        assert!(
            m100 < m0,
            "expected dM/dh < 0 under strong trapping; got M(0)={m0}, M(100)={m100}"
        );
    }
}
