//! Bent-ray horizon extension under atmospheric ducting.
//!
//! Under the standard 4/3-Earth model the radar horizon is
//!
//! ```text
//! D ≈ 4.12 · (√h_r + √h_t)   km     (h_r, h_t in metres)
//! ```
//!
//! (Skolnik §2.10 eq. (2.40); the leading coefficient `4.12 ≈ √(2·k·R_E)
//! / 1000` for `k = 4/3` and `R_E = 6 371 km`.) Ducting layers extend the
//! horizon by trapping rays inside or near the layer. The amount of
//! extension depends on the duct type, depth, geometry and (for
//! evaporation ducts) frequency:
//!
//! - **Surface duct**: typical extensions 30–300 % at S/X-band. Naval
//!   Postgraduate School / Hitney experiments and Lincoln Lab worked
//!   examples bound the multiplier near `1 + h_d / 100` saturated at
//!   ~3.5×, where `h_d` is the duct top (m). A 50 m surface duct
//!   commonly extends 100 km of nominal horizon to ~150 km.
//! - **Elevated duct**: above-duct targets are *blocked* (trapping
//!   layer acts as a one-sided mirror; a surface radar sees through
//!   the layer poorly). For below-duct geometry, weak extension
//!   (factor 1.1–1.3) per Patterson (AREPS).
//! - **Evaporation duct**: frequency-dependent. At S-band (~3 GHz) the
//!   duct is too thin (5–40 m) compared with the trapped half-mode
//!   wavelength so the boost is essentially nil. At X-band (~10 GHz)
//!   boost climbs to 10–30 %; at Ku-band (~16 GHz) the duct is many
//!   wavelengths deep and trapping is efficient — 30–80 % extension.
//!   The frequency-dependence is the diagnostic signature: a sensor
//!   without it cannot be claiming to model evaporation ducts.
//!
//! This module exposes:
//!
//! - [`standard_horizon_range_m`] — self-contained 4/3-Earth horizon
//!   (matches the Skolnik §2.10 formula; the canonical Lincoln Lab
//!   worked example at h_r = h_t = 100 m yields 82.4 km).
//! - [`ducting_horizon_range_m`] — ducting-aware horizon. Falls back to
//!   the standard horizon when no duct is present.
//! - [`horizon_extension_pct`] — convenience returning the percentage
//!   extension over the standard horizon.
//! - [`AboveDuctVisibility`] — flag used by the elevated-duct branch to
//!   signal that a target *above* the trapping layer is geometrically
//!   blocked even though the formal horizon range is finite.
//! - [`ducting_horizon_with_blocking`] — combined return value that
//!   pairs the horizon range with the above-duct-blocked flag for the
//!   elevated case.
//!
//! # References
//!
//! - Skolnik, *Introduction to Radar Systems*, 3rd ed. (2001), §2.10 —
//!   standard horizon formula and anomalous-propagation extensions.
//! - Hitney, *Refractive Effects from VHF to EHF, Part B* (NPS lecture
//!   notes 1992) — surface- and evaporation-duct propagation factor
//!   curves vs frequency and duct height.
//! - Hitney & Vieth (1990), *Statistical Assessment of Evaporation Duct
//!   Propagation*, IEEE Trans. Ant. Prop. 38(6) — frequency-dependence
//!   of evaporation-duct extension, S-band null vs Ku-band saturation.
//! - Patterson et al. (1994), *AREPS: Advanced Refractive Effects
//!   Prediction System* (NRaD TD-2648) — operational duct-extension
//!   look-up tables used by the U.S. Navy.
//! - MIT Lincoln Laboratory, *Introduction to Radar Systems* short
//!   course, ch. 4 worked examples — `D = 4.12 · (√h_r + √h_t)` km
//!   anchor, h_r = h_t = 100 m ⇒ D ≈ 82 km.

use super::duct_detection::DuctRegime;
use crate::propagation::interpolate_linear;

/// Skolnik §2.10 leading coefficient (km) for the 4/3-Earth horizon
/// formula `D = SKOLNIK_HORIZON_COEFFICIENT_KM · (√h_r + √h_t)`.
/// Numerically `√(2 · (4/3) · 6 371 km) ≈ 4.124` — rounded to `4.12`
/// in the textbook.
pub const SKOLNIK_HORIZON_COEFFICIENT_KM: f64 = 4.12;

/// Maximum surface-duct multiplier on the standard horizon range,
/// after the linear-in-depth scaling saturates. Per Hitney (1992) and
/// Lincoln Lab worked examples, multi-hundred-percent extensions are
/// possible but rarely exceed 3.5× even in the most aggressive
/// trapping conditions.
pub const SURFACE_DUCT_MAX_MULTIPLIER: f64 = 3.5;

/// Visibility flag for the elevated-duct case. The horizon formula
/// still returns a finite range, but for an *above-duct* geometry the
/// surface radar is functionally blocked from seeing the target.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AboveDuctVisibility {
    /// Target sits within the same geometry as the radar (below the
    /// duct base, or in the standard / surface-duct / evaporation-duct
    /// cases that don't generate a trapping layer aloft). The reported
    /// horizon is meaningful.
    Visible,

    /// Target is geometrically above an elevated trapping layer that
    /// blocks the surface radar's line-of-sight. The reported range is
    /// a *formal* horizon — callers should mark targets above the duct
    /// as non-detectable regardless of range.
    AboveDuctBlocked,
}

/// Combined return value for [`ducting_horizon_with_blocking`].
#[derive(Debug, Clone, Copy)]
pub struct DuctingHorizonResult {
    /// Horizon range in metres (extended by ducting where applicable).
    pub range_m: f64,
    /// Above-duct visibility flag. See [`AboveDuctVisibility`].
    pub visibility: AboveDuctVisibility,
}

/// Standard 4/3-Earth horizon range (m) between a radar of height
/// `radar_height_m` and a target of height `target_height_m`.
///
/// Implements the Skolnik §2.10 / MIT Lincoln Lab ch. 4 formula
///
/// ```text
/// D = 4.12 · (√h_r + √h_t)   km
/// ```
///
/// Returns `0.0` for non-positive antenna or target heights.
pub fn standard_horizon_range_m(radar_height_m: f64, target_height_m: f64) -> f64 {
    let h_r = radar_height_m.max(0.0);
    let h_t = target_height_m.max(0.0);
    let d_km = SKOLNIK_HORIZON_COEFFICIENT_KM * (h_r.sqrt() + h_t.sqrt());
    d_km * 1000.0
}

/// Ducting-aware horizon range (m) for the supplied regime.
///
/// - `DuctRegime::None` → standard 4/3-Earth horizon.
/// - `DuctRegime::Surface { height_m }` → `(1 + h/100)` multiplier on
///   the standard horizon, clamped to [`SURFACE_DUCT_MAX_MULTIPLIER`].
/// - `DuctRegime::Elevated { .. }` → 1.2× multiplier (typical
///   below-duct boost per AREPS); the above-duct blocking flag is
///   surfaced by [`ducting_horizon_with_blocking`].
/// - `DuctRegime::Evaporation { height_m }` → frequency-dependent
///   multiplier, see [`evaporation_duct_multiplier`].
///
/// For elevated ducts the function does *not* signal above-duct
/// blocking; that requires the explicit
/// [`ducting_horizon_with_blocking`] wrapper. Use this entry point
/// when the consumer only cares about the (possibly extended) range.
pub fn ducting_horizon_range_m(
    radar_height_m: f64,
    target_height_m: f64,
    duct: DuctRegime,
    frequency_ghz: f64,
) -> f64 {
    let standard = standard_horizon_range_m(radar_height_m, target_height_m);
    let multiplier = match duct {
        DuctRegime::None => 1.0,
        DuctRegime::Surface { height_m } => {
            let m = 1.0 + (height_m / 100.0);
            m.clamp(1.0, SURFACE_DUCT_MAX_MULTIPLIER)
        }
        DuctRegime::Elevated { .. } => {
            // Below-duct boost per AREPS / Patterson 1994 fig 4.6
            // (typical 1.1-1.3 for elevated layers a few hundred m
            // thick). Pick the mid-band 1.2 by default.
            1.2
        }
        DuctRegime::Evaporation { height_m } => evaporation_duct_multiplier(height_m, frequency_ghz),
    };
    standard * multiplier
}

/// Convenience: percentage horizon extension over the standard
/// 4/3-Earth horizon for the supplied regime.
///
/// `(D_duct - D_std) / D_std · 100`. Returns 0 % for `DuctRegime::None`.
pub fn horizon_extension_pct(
    radar_height_m: f64,
    target_height_m: f64,
    duct: DuctRegime,
    frequency_ghz: f64,
) -> f64 {
    let standard = standard_horizon_range_m(radar_height_m, target_height_m);
    if standard <= 0.0 {
        return 0.0;
    }
    let ducted = ducting_horizon_range_m(radar_height_m, target_height_m, duct, frequency_ghz);
    100.0 * (ducted - standard) / standard
}

/// Ducting-aware horizon plus an above-duct visibility flag. Use this
/// entry point when the target geometry might place it above an
/// elevated trapping layer; callers can treat
/// `AboveDuctVisibility::AboveDuctBlocked` as a non-detection regardless
/// of the reported range.
pub fn ducting_horizon_with_blocking(
    radar_height_m: f64,
    target_height_m: f64,
    duct: DuctRegime,
    frequency_ghz: f64,
) -> DuctingHorizonResult {
    let range_m = ducting_horizon_range_m(radar_height_m, target_height_m, duct, frequency_ghz);
    let visibility = match duct {
        DuctRegime::Elevated { base_m, top_m } => {
            // Target above the duct top is blocked. Target between
            // base and top is itself trapped (visible by trapping).
            // Target below the duct base is visible by surface
            // line-of-sight + below-duct boost.
            if target_height_m > top_m {
                AboveDuctVisibility::AboveDuctBlocked
            } else {
                let _ = base_m; // suppressed reserved-binding warning
                AboveDuctVisibility::Visible
            }
        }
        _ => AboveDuctVisibility::Visible,
    };
    DuctingHorizonResult { range_m, visibility }
}

/// Frequency-dependent evaporation-duct multiplier.
///
/// Trapping efficiency in an evaporation duct scales with the ratio of
/// duct height to wavelength `h_d / λ`. Below ~3 GHz over a 5-40 m
/// duct, that ratio is small (the trapped mode doesn't fit) and the
/// boost is essentially nil. At X-band (~10 GHz) and especially
/// Ku-band (~16 GHz) the ratio is large and the boost saturates the
/// duct's natural envelope.
///
/// Anchors (multiplier, mid-of-band):
///
/// | f (GHz) | mid-boost mult | spec phrasing      |
/// |---------|----------------|--------------------|
/// | 3       | 1.00           | marginal           |
/// | 10      | 1.15           | 10-30 % marginal   |
/// | 16      | 1.50           | 30-80 % substantial|
/// | 24      | 1.70           | substantial        |
///
/// Multiplier scales mildly with duct height inside the published
/// 5-40 m envelope; thicker ducts (closer to 40 m) get a small
/// further boost. Below the L-band (≤ 1 GHz) the boost stays at unity
/// (curve clamped). Above 40 GHz absorption dominates regardless.
fn evaporation_duct_multiplier(duct_height_m: f64, frequency_ghz: f64) -> f64 {
    if duct_height_m <= 0.0 {
        return 1.0;
    }
    const ANCHORS: &[(f64, f64)] = &[
        (1.0, 1.00),
        (3.0, 1.00),
        (10.0, 1.15),
        (16.0, 1.50),
        (24.0, 1.70),
        (40.0, 1.70),
    ];
    let f = frequency_ghz.clamp(ANCHORS[0].0, ANCHORS[ANCHORS.len() - 1].0);
    let base = interpolate_linear(f, ANCHORS);

    // Mild duct-height adjustment: a 5 m duct gets the base multiplier;
    // a 40 m duct gets up to +15 %. Clamp to [5, 40] m envelope. The
    // height-scale boost only applies once the wavelength fits inside
    // the duct (cm-wave, ≥ 5 GHz); below that the trapped half-mode is
    // wider than the duct depth and the boost is genuinely nil
    // regardless of height (S-band ~0 % boost is the diagnostic
    // signature per Hitney & Vieth 1990 fig 7).
    let h = duct_height_m.clamp(5.0, 40.0);
    let height_scale = if f >= 5.0 {
        1.0 + 0.15 * ((h - 5.0) / 35.0)
    } else {
        1.0
    };
    base * height_scale
}


// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
#[path = "ducting_horizon_tests.rs"]
mod tests;
