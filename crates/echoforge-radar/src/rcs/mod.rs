//! Aspect / elevation / frequency / polarization RCS lookup.
//!
//! Closes correctness gate **C3** of the EchoForge correctness-first plan:
//! the existing `TakeoffProfile::rcs_scalar` is a single scalar per target,
//! which makes every target look identical from every aspect — a known
//! label-leak failure mode. Real radar cross section varies by 20+ dB with
//! aspect angle, depends on frequency, polarization, and elevation, and
//! fluctuates pulse-to-pulse / scan-to-scan per the Swerling models.
//!
//! This module provides an [`Rcs`] container that owns one or more
//! [`RcsLookup`] tables and exposes two evaluators:
//!
//! - [`Rcs::evaluate_static`] — deterministic median RCS (in dBsm) by
//!   bilinear interpolation over `(aspect, elevation)` for the nearest
//!   `(target_class, freq_ghz, polarization)` table. Useful for tests
//!   and plotting.
//! - [`Rcs::evaluate`] — same lookup, then overlays a Swerling-model
//!   fluctuation seeded deterministically by `(seed, pulse_index)`.
//!
//! ## Lookup rules
//!
//! 1. **Target class match**: exact string match on `target_class`. If
//!    multiple tables exist for the same class, the frequency / polarization
//!    rules below choose among them.
//! 2. **Frequency match**: among tables that match the requested class
//!    and (preferred) polarization, the table whose `frequency_ghz` is
//!    closest in **log-frequency** to the requested value wins.
//!    Log-frequency is the right metric for radar bands because they
//!    are roughly geometrically spaced (e.g. L 1 GHz, S 3 GHz, X 10 GHz,
//!    Ka 35 GHz, W 94 GHz).
//! 3. **Polarization match**: exact match preferred. If no exact match
//!    exists for the chosen class+frequency, the value is the
//!    polarization-average of every table that matches class and
//!    log-nearest frequency. This mirrors the standard "co-pol average"
//!    recovery used when only a single polarization measurement is on hand.
//! 4. **Aspect wrap**: aspect is reduced modulo 360 before lookup; aspect
//!    359.9 and aspect -0.1 both land near aspect 0.
//! 5. **Out-of-grid clamp**: elevation (and any aspect outside the grid
//!    after wrapping) is clamped to the nearest grid edge — the module
//!    refuses to extrapolate beyond the measured / cited region.
//!
//! ## Fluctuation models
//!
//! Per Skolnik *Introduction to Radar Systems* (3rd ed., ch. 2) the
//! Swerling cases are:
//!
//! - **Swerling 0** — non-fluctuating (returns the deterministic median).
//! - **Swerling 1** — Rayleigh amplitude, scan-to-scan correlated. The
//!   seed is mixed with a derived scan index (`pulse_index /
//!   SCAN_SIZE`) so every pulse inside the same scan returns the same
//!   value.
//! - **Swerling 2** — Rayleigh amplitude, pulse-to-pulse decorrelated.
//!   Seed is mixed with the pulse index directly.
//! - **Swerling 3** — chi-squared 4-DOF, scan-to-scan correlated.
//! - **Swerling 4** — chi-squared 4-DOF, pulse-to-pulse decorrelated.
//!
//! The same `(seed, pulse_index)` always returns the same RCS, so episodes
//! remain bit-reproducible — Lane B (reproducibility) is not regressed.
//!
//! ## Strict-open posture
//!
//! The bundled tables under [`Rcs::seeded_public_proxy_v1`] are
//! **public-proxy reference tables based on published aggregate
//! measurements; not measured equivalents; do not claim platform-specific
//! signature truth.** Each table cites the source it was patterned after
//! verbatim, and downstream validation reports MUST gate any platform-
//! specific claim on the validation-tier ladder.
//!
//! ## What this module does NOT do
//!
//! - It does not integrate with `synthesize_takeoff_episode` (that wiring
//!   is Lane A, the radar-equation lane, in a separate packet).
//! - It does not invent measurements. New tables added by callers must
//!   cite a public source.
//! - It does not extrapolate beyond the grid; out-of-band requests clamp
//!   to the nearest measured edge.

use serde::{Deserialize, Serialize};

mod lookup_tables;
mod prng;
#[cfg(test)]
mod tests;

/// Polarization channels recognised by [`RcsLookup`] tables.
///
/// `Co` and `Cross` are the polarization-agnostic helpers: a table tagged
/// `Co` is treated as the co-polar average (mean of `Hh`+`Vv`) and a
/// `Cross` table as the cross-polar average (mean of `Hv`+`Vh`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Polarization {
    Hh,
    Hv,
    Vh,
    Vv,
    Co,
    Cross,
}

/// Swerling fluctuation models, per Skolnik *Introduction to Radar
/// Systems* 3rd ed., ch. 2.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SwerlingModel {
    /// Non-fluctuating; [`Rcs::evaluate`] returns the deterministic median.
    Swerling0,
    /// Rayleigh, scan-to-scan correlated.
    Swerling1,
    /// Rayleigh, pulse-to-pulse decorrelated.
    Swerling2,
    /// Chi-squared 4-DOF, scan-to-scan correlated.
    Swerling3,
    /// Chi-squared 4-DOF, pulse-to-pulse decorrelated.
    Swerling4,
}

/// Number of pulses considered one "scan" for Swerling 1 / Swerling 3.
///
/// Callers that integrate a different number of pulses per scan can
/// re-create the table with their own scan size; this constant is the
/// default the fluctuation overlay uses internally.
pub const SWERLING_DEFAULT_SCAN_SIZE: usize = 32;

/// Sorted azimuth / elevation grid (degrees) for an [`RcsLookup`].
///
/// Both vectors must be sorted strictly ascending. Azimuth values are
/// interpreted modulo 360 at query time.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AspectGrid {
    pub azimuth_deg: Vec<f64>,
    pub elevation_deg: Vec<f64>,
}

impl AspectGrid {
    /// Returns true when both axes are non-empty and strictly ascending.
    pub fn is_valid(&self) -> bool {
        if self.azimuth_deg.is_empty() || self.elevation_deg.is_empty() {
            return false;
        }
        self.azimuth_deg.windows(2).all(|w| w[0] < w[1])
            && self.elevation_deg.windows(2).all(|w| w[0] < w[1])
    }
}

/// A single (target_class, frequency, polarization) RCS table.
///
/// `rcs_dbsm` is stored row-major over `[azimuth × elevation]`, i.e.
/// `rcs_dbsm[az_index * n_elev + el_index]`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RcsLookup {
    pub target_class: String,
    pub frequency_ghz: f64,
    pub polarization: Polarization,
    pub aspect_grid: AspectGrid,
    pub rcs_dbsm: Vec<f64>,
    pub fluctuation: SwerlingModel,
    pub citation: String,
    pub citation_url: Option<String>,
}

impl RcsLookup {
    /// Returns the expected element count for [`Self::rcs_dbsm`].
    pub fn expected_len(&self) -> usize {
        self.aspect_grid.azimuth_deg.len() * self.aspect_grid.elevation_deg.len()
    }

    /// Returns true when `rcs_dbsm.len()` matches the grid and the grid
    /// itself is valid.
    pub fn is_valid(&self) -> bool {
        self.aspect_grid.is_valid() && self.rcs_dbsm.len() == self.expected_len()
    }

    /// Bilinearly interpolates the table at the requested aspect /
    /// elevation. Aspect is wrapped to `[0, 360)`. Elevation is clamped
    /// to the grid edges (no extrapolation).
    fn interpolate_dbsm(&self, aspect_deg: f64, elevation_deg: f64) -> f64 {
        debug_assert!(self.is_valid(), "RcsLookup grid invalid for interpolation");

        let az = prng::wrap_deg_360(aspect_deg);
        let n_az = self.aspect_grid.azimuth_deg.len();
        let n_el = self.aspect_grid.elevation_deg.len();

        let (az_lo, az_hi, az_frac) = prng::wrapping_bracket(&self.aspect_grid.azimuth_deg, az);
        let (el_lo, el_hi, el_frac) =
            prng::clamping_bracket(&self.aspect_grid.elevation_deg, elevation_deg);

        let idx = |az_i: usize, el_i: usize| -> usize { az_i * n_el + el_i };

        let v00 = self.rcs_dbsm[idx(az_lo, el_lo)];
        let v01 = self.rcs_dbsm[idx(az_lo, el_hi)];
        let v10 = self.rcs_dbsm[idx(az_hi, el_lo)];
        let v11 = self.rcs_dbsm[idx(az_hi, el_hi)];
        let _ = n_az; // silence reserved-binding warning when debug_assert is off

        let v_lo = v00 * (1.0 - el_frac) + v01 * el_frac;
        let v_hi = v10 * (1.0 - el_frac) + v11 * el_frac;
        v_lo * (1.0 - az_frac) + v_hi * az_frac
    }
}

/// Owns one or more [`RcsLookup`] tables and provides the lookup /
/// fluctuation interface.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Rcs {
    pub tables: Vec<RcsLookup>,
}

impl Rcs {
    /// Constructs an empty `Rcs` with no tables.
    pub fn empty() -> Self {
        Self { tables: Vec::new() }
    }

    /// Adds a lookup table. The `citation` field on each entry must reference
    /// a published source; the vendor-scrub lane enforces this at CI time.
    /// Rerun: `just vendor-scrub` — CI citation check; replayable locally.
    // jankurai:allow HLT-027-HUMAN-REVIEW-EVIDENCE-GAP vendor-scrub CI lane is the replayable proof; rerun: just vendor-scrub
    pub fn add_table(&mut self, table: RcsLookup) {
        self.tables.push(table);
    }

    /// Deterministic median RCS in dBsm.
    ///
    /// Performs the (target_class, log-nearest frequency,
    /// polarization-or-average) lookup described in the module docs and
    /// bilinearly interpolates the chosen table over (aspect, elevation).
    /// Returns `f64::NEG_INFINITY` when no table matches the requested
    /// target class.
    pub fn evaluate_static(
        &self,
        target_class: &str,
        aspect_deg: f64,
        elevation_deg: f64,
        freq_ghz: f64,
        pol: Polarization,
    ) -> f64 {
        match self.select(target_class, freq_ghz, pol) {
            Selection::None => f64::NEG_INFINITY,
            Selection::Exact(idx) => self.tables[idx].interpolate_dbsm(aspect_deg, elevation_deg),
            Selection::Averaged(indices) => {
                let mut acc = 0.0f64;
                for idx in &indices {
                    acc += self.tables[*idx].interpolate_dbsm(aspect_deg, elevation_deg);
                }
                acc / indices.len() as f64
            }
        }
    }

    /// RCS with a Swerling fluctuation overlay applied.
    ///
    /// The fluctuation model is taken from the chosen table's
    /// [`RcsLookup::fluctuation`] field; the deterministic median is
    /// converted to linear (m²), the fluctuation factor is sampled
    /// from a deterministic RNG seeded by `(seed, pulse_index)`, and
    /// the result is returned in dBsm. The same `(target_class,
    /// aspect, elevation, freq, pol, seed, pulse_index)` always
    /// returns the same value.
    #[allow(clippy::too_many_arguments)]
    pub fn evaluate(
        &self,
        target_class: &str,
        aspect_deg: f64,
        elevation_deg: f64,
        freq_ghz: f64,
        pol: Polarization,
        seed: u64,
        pulse_index: usize,
    ) -> f64 {
        let median_dbsm =
            self.evaluate_static(target_class, aspect_deg, elevation_deg, freq_ghz, pol);
        if !median_dbsm.is_finite() {
            return median_dbsm;
        }
        let sel = self.select(target_class, freq_ghz, pol);
        let model = if let Selection::Exact(idx) = sel {
            self.tables[idx].fluctuation
        } else if let Selection::Averaged(ref indices) = sel {
            self.tables[indices[0]].fluctuation
        } else {
            return median_dbsm;
        };

        prng::apply_swerling(median_dbsm, model, seed, pulse_index)
    }

    /// Returns a populated `Rcs` with public-proxy reference tables
    /// (small fixed-wing UAS, large bird, quadrotor) built from
    /// published aggregate measurements; not measured equivalents; do
    /// not claim platform-specific signature truth. Each table carries
    /// a verbatim citation of the source it was patterned after.
    pub fn seeded_public_proxy_v1() -> Self {
        let mut rcs = Rcs::empty();
        rcs.add_table(lookup_tables::small_fixed_wing_uas_x_band_vv());
        rcs.add_table(lookup_tables::single_large_bird_x_band_hh());
        rcs.add_table(lookup_tables::quadrotor_x_band_vv());
        rcs
    }

    fn select(&self, target_class: &str, freq_ghz: f64, pol: Polarization) -> Selection {
        if self.tables.is_empty() {
            return Selection::None;
        }

        let class_matches: Vec<usize> = self
            .tables
            .iter()
            .enumerate()
            .filter(|(_, t)| t.target_class == target_class)
            .map(|(i, _)| i)
            .collect();
        if class_matches.is_empty() {
            return Selection::None;
        }

        let log_target = prng::log10_safe(freq_ghz);
        let mut best_log_dist = f64::INFINITY;
        for idx in &class_matches {
            let d = (prng::log10_safe(self.tables[*idx].frequency_ghz) - log_target).abs();
            if d < best_log_dist {
                best_log_dist = d;
            }
        }
        let tol = 1e-9;
        let freq_matches: Vec<usize> = class_matches
            .into_iter()
            .filter(|idx| {
                let d =
                    (prng::log10_safe(self.tables[*idx].frequency_ghz) - log_target).abs();
                d <= best_log_dist + tol
            })
            .collect();

        if let Some(idx) = freq_matches
            .iter()
            .copied()
            .find(|i| self.tables[*i].polarization == pol)
        {
            return Selection::Exact(idx);
        }

        Selection::Averaged(freq_matches)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Selection {
    None,
    Exact(usize),
    Averaged(Vec<usize>),
}
