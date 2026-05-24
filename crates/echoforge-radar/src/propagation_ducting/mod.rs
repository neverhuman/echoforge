//! Atmospheric ducting via bent-ray ITU-R P.453 modified refractivity.
//!
//! Wave 15 of the radar-expert credibility roadmap lifts the static
//! P.453 refractivity calculation in [`crate::propagation`] to a
//! dynamic, regime-aware ducting model. The static reference
//! atmosphere is fine when the lower troposphere is "standard"
//! (`dN/dh ≈ -39 N/km`), but operational radar performance over coasts,
//! warm seas, and strong-inversion land surfaces is dominated by
//! anomalous propagation: surface ducts, evaporation ducts, and
//! elevated trapping layers can extend horizon range by 30 % to 300 %,
//! or *prevent* above-duct detection entirely.
//!
//! The module is organised as three small, self-contained primitives:
//!
//! - [`modified_refractivity`] — the modified-refractivity unit
//!   conversion `M(h) = N(h) + 0.157·h` and the linear-profile helpers
//!   used throughout the rest of the module.
//! - [`duct_detection`] — gradient/humidity → [`DuctRegime`] classifier
//!   per the ITU-R P.453, Skolnik §2.10 and Hitney 1992 rule-set.
//! - [`ducting_horizon`] — bent-ray horizon extension under each
//!   regime. Re-implements the standard 4/3-Earth Skolnik horizon for
//!   self-containment so downstream link-budget code can call this
//!   single module without reaching into [`crate::propagation`].
//!
//! # Composition
//!
//! Typical use:
//!
//! ```text
//! let regime = duct_detection::classify_duct_regime(grad, k, rh);
//! let range  = ducting_horizon::ducting_horizon_range_m(h_r, h_t, regime, f_ghz);
//! ```
//!
//! For the elevated-duct case the
//! [`ducting_horizon::ducting_horizon_with_blocking`] entry point pairs
//! the horizon range with an above-duct visibility flag so detection
//! pipelines can mark targets above a trapping layer as non-detectable.
//!
//! # References
//!
//! - ITU-R Recommendation P.453-14 (2019), §§1–4 — refractivity,
//!   modified refractivity, gradient regimes.
//! - Skolnik, *Introduction to Radar Systems*, 3rd ed. (2001), §2.10 —
//!   anomalous propagation, ducting horizon extensions.
//! - Hitney, *Refractive Effects from VHF to EHF, Part B* (NPS lecture
//!   notes 1992) — duct-depth and propagation-factor tabulations.
//! - Hitney & Vieth (1990), *Statistical Assessment of Evaporation
//!   Duct Propagation*, IEEE Trans. Ant. Prop. 38(6).
//! - Patterson et al. (1994), *Advanced Refractive Effects Prediction
//!   System (AREPS)*, NRaD TD-2648.
//! - MIT Lincoln Laboratory, *Introduction to Radar Systems* short
//!   course, ch. 4 worked examples.

pub mod duct_detection;
pub mod ducting_horizon;
pub mod modified_refractivity;

pub use duct_detection::{
    classify_duct_regime, DuctRegime, EVAPORATION_DUCT_MAX_RH, EVAPORATION_DUCT_MIN_RH,
    STANDARD_GRADIENT_THRESHOLD_N_PER_KM, SURFACE_DUCT_MIN_K_FACTOR,
    TRAPPING_GRADIENT_THRESHOLD_N_PER_KM,
};
pub use ducting_horizon::{
    ducting_horizon_range_m, ducting_horizon_with_blocking, horizon_extension_pct,
    standard_horizon_range_m, AboveDuctVisibility, DuctingHorizonResult,
    SKOLNIK_HORIZON_COEFFICIENT_KM, SURFACE_DUCT_MAX_MULTIPLIER,
};
pub use modified_refractivity::{
    modified_refractivity_m_units, modified_refractivity_profile, refractivity_profile,
    EARTH_CURVATURE_COEFFICIENT_PER_M,
};
