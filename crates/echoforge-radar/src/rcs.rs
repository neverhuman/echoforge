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
//!    fallback used when only a single polarization measurement is on hand.
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

use std::f64::consts::PI;

use serde::{Deserialize, Serialize};

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

        let az = wrap_deg_360(aspect_deg);
        let n_az = self.aspect_grid.azimuth_deg.len();
        let n_el = self.aspect_grid.elevation_deg.len();

        let (az_lo, az_hi, az_frac) = wrapping_bracket(&self.aspect_grid.azimuth_deg, az);
        let (el_lo, el_hi, el_frac) =
            clamping_bracket(&self.aspect_grid.elevation_deg, elevation_deg);

        let idx = |az_i: usize, el_i: usize| -> usize { az_i * n_el + el_i };

        let v00 = self.rcs_dbsm[idx(az_lo, el_lo)];
        let v01 = self.rcs_dbsm[idx(az_lo, el_hi)];
        let v10 = self.rcs_dbsm[idx(az_hi, el_lo)];
        let v11 = self.rcs_dbsm[idx(az_hi, el_hi)];
        let _ = n_az; // silence unused when debug_assert is off

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
        Self {
            tables: Vec::new(),
        }
    }

    /// Adds a table. Callers should populate `citation` truthfully; the
    /// strict-open posture forbids fabricated references.
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
        let median_dbsm = self.evaluate_static(target_class, aspect_deg, elevation_deg, freq_ghz, pol);
        if !median_dbsm.is_finite() {
            return median_dbsm;
        }
        let model = match self.select(target_class, freq_ghz, pol) {
            Selection::None => return median_dbsm,
            Selection::Exact(idx) => self.tables[idx].fluctuation,
            Selection::Averaged(indices) => self.tables[indices[0]].fluctuation,
        };

        apply_swerling(median_dbsm, model, seed, pulse_index)
    }

    /// Returns a populated `Rcs` with public-proxy reference tables
    /// (small fixed-wing UAS, large bird, quadrotor) built from
    /// published aggregate measurements; not measured equivalents; do
    /// not claim platform-specific signature truth. Each table carries
    /// a verbatim citation of the source it was patterned after.
    pub fn seeded_public_proxy_v1() -> Self {
        let mut rcs = Rcs::empty();
        rcs.add_table(small_fixed_wing_uas_x_band_vv());
        rcs.add_table(single_large_bird_x_band_hh());
        rcs.add_table(quadrotor_x_band_vv());
        rcs
    }

    /// Internal helper. Picks the best table (or set of tables) for
    /// the requested key triple.
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

        // Pick the log-nearest frequency among class matches.
        let log_target = log10_safe(freq_ghz);
        let mut best_log_dist = f64::INFINITY;
        for idx in &class_matches {
            let d = (log10_safe(self.tables[*idx].frequency_ghz) - log_target).abs();
            if d < best_log_dist {
                best_log_dist = d;
            }
        }
        // Tolerance is generous so multiple-pol tables at the same
        // frequency all qualify for the averaging fallback below.
        let tol = 1e-9;
        let freq_matches: Vec<usize> = class_matches
            .into_iter()
            .filter(|idx| {
                let d = (log10_safe(self.tables[*idx].frequency_ghz) - log_target).abs();
                d <= best_log_dist + tol
            })
            .collect();

        // Prefer exact polarization match.
        if let Some(idx) = freq_matches
            .iter()
            .copied()
            .find(|i| self.tables[*i].polarization == pol)
        {
            return Selection::Exact(idx);
        }

        // No exact polarization match — average everything left.
        Selection::Averaged(freq_matches)
    }
}

/// Result of [`Rcs::select`].
#[derive(Debug, Clone, PartialEq, Eq)]
enum Selection {
    None,
    Exact(usize),
    Averaged(Vec<usize>),
}

// ---------------------------------------------------------------------------
// Deterministic RNG (mirrors the `SplitMix64` already used by sim.rs so we
// do not add a new dependency).
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct SplitMix64 {
    state: u64,
    cached_normal: Option<f64>,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self {
            state: seed,
            cached_normal: None,
        }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }

    fn unit(&mut self) -> f64 {
        // Use the top 53 bits for a uniform [0,1) f64.
        let bits = self.next_u64() >> 11;
        (bits as f64) / ((1u64 << 53) as f64)
    }

    fn normal(&mut self) -> f64 {
        if let Some(value) = self.cached_normal.take() {
            return value;
        }
        let u1 = self.unit().clamp(1e-12, 1.0 - 1e-12);
        let u2 = self.unit().clamp(1e-12, 1.0 - 1e-12);
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * PI * u2;
        let z0 = r * theta.cos();
        let z1 = r * theta.sin();
        self.cached_normal = Some(z1);
        z0
    }

    /// Sample from `Exponential(1)`. Used as the Rayleigh-power kernel
    /// for Swerling 1 / Swerling 2.
    fn exponential_unit(&mut self) -> f64 {
        let u = self.unit().clamp(1e-12, 1.0 - 1e-12);
        -u.ln()
    }

    /// Sample from `chi-squared(k=4) / 4`, which has mean 1 and is the
    /// Swerling 3 / 4 amplitude-squared kernel.
    fn chi_squared4_norm(&mut self) -> f64 {
        // chi-squared(4) = sum of 4 standard-normal squares; dividing by
        // 4 yields a mean-1 fluctuation factor.
        let mut acc = 0.0;
        for _ in 0..4 {
            let z = self.normal();
            acc += z * z;
        }
        acc / 4.0
    }
}

// ---------------------------------------------------------------------------
// Helpers.
// ---------------------------------------------------------------------------

fn wrap_deg_360(deg: f64) -> f64 {
    let mut v = deg % 360.0;
    if v < 0.0 {
        v += 360.0;
    }
    if v >= 360.0 {
        v -= 360.0;
    }
    v
}

fn log10_safe(freq_ghz: f64) -> f64 {
    if freq_ghz > 0.0 {
        freq_ghz.log10()
    } else {
        f64::NEG_INFINITY
    }
}

/// Bracket `value` on a strictly-ascending grid by linear search. Returns
/// `(lo_idx, hi_idx, frac)` where `frac ∈ [0,1]` is the position of
/// `value` between `grid[lo_idx]` and `grid[hi_idx]`. Out-of-range values
/// clamp to the nearest edge (no extrapolation).
fn clamping_bracket(grid: &[f64], value: f64) -> (usize, usize, f64) {
    debug_assert!(!grid.is_empty());
    if grid.len() == 1 {
        return (0, 0, 0.0);
    }
    if value <= grid[0] {
        return (0, 0, 0.0);
    }
    let last = grid.len() - 1;
    if value >= grid[last] {
        return (last, last, 0.0);
    }
    for i in 0..last {
        if value >= grid[i] && value <= grid[i + 1] {
            let span = grid[i + 1] - grid[i];
            let frac = if span > 0.0 { (value - grid[i]) / span } else { 0.0 };
            return (i, i + 1, frac);
        }
    }
    (last, last, 0.0)
}

/// Bracket `value` on an ascending azimuth grid with wrap-around. The
/// caller is responsible for wrapping `value` into `[0, 360)` first.
fn wrapping_bracket(grid: &[f64], value: f64) -> (usize, usize, f64) {
    debug_assert!(!grid.is_empty());
    let n = grid.len();
    if n == 1 {
        return (0, 0, 0.0);
    }
    let first = grid[0];
    let last = grid[n - 1];

    // Inside the grid — bracket normally.
    if value >= first && value <= last {
        for i in 0..(n - 1) {
            if value >= grid[i] && value <= grid[i + 1] {
                let span = grid[i + 1] - grid[i];
                let frac = if span > 0.0 { (value - grid[i]) / span } else { 0.0 };
                return (i, i + 1, frac);
            }
        }
        return (n - 1, n - 1, 0.0);
    }

    // Wrap segment: between grid[n-1] and grid[0] + 360.
    let span = (first + 360.0) - last;
    let v = if value < first { value + 360.0 } else { value };
    let frac = if span > 0.0 { (v - last) / span } else { 0.0 };
    (n - 1, 0, frac)
}

/// Apply a Swerling fluctuation factor to a median dBsm value and
/// return the fluctuated dBsm.
fn apply_swerling(median_dbsm: f64, model: SwerlingModel, seed: u64, pulse_index: usize) -> f64 {
    match model {
        SwerlingModel::Swerling0 => median_dbsm,
        SwerlingModel::Swerling1 => {
            let scan_idx = pulse_index / SWERLING_DEFAULT_SCAN_SIZE;
            let mut rng = SplitMix64::new(mix_seed(seed, scan_idx as u64, 0xA1));
            db_factor(median_dbsm, rng.exponential_unit())
        }
        SwerlingModel::Swerling2 => {
            let mut rng = SplitMix64::new(mix_seed(seed, pulse_index as u64, 0xA2));
            db_factor(median_dbsm, rng.exponential_unit())
        }
        SwerlingModel::Swerling3 => {
            let scan_idx = pulse_index / SWERLING_DEFAULT_SCAN_SIZE;
            let mut rng = SplitMix64::new(mix_seed(seed, scan_idx as u64, 0xA3));
            db_factor(median_dbsm, rng.chi_squared4_norm())
        }
        SwerlingModel::Swerling4 => {
            let mut rng = SplitMix64::new(mix_seed(seed, pulse_index as u64, 0xA4));
            db_factor(median_dbsm, rng.chi_squared4_norm())
        }
    }
}

fn db_factor(median_dbsm: f64, linear_factor: f64) -> f64 {
    // Convert median dBsm -> linear, scale, convert back. The factor is
    // already a unitless mean-1 fluctuation, so any factor of exactly 0
    // would push dBsm to -inf; clamp at a numerically safe floor.
    let f = linear_factor.max(1e-30);
    median_dbsm + 10.0 * f.log10()
}

fn mix_seed(seed: u64, counter: u64, salt: u64) -> u64 {
    // Cheap mixing that avoids a zero output for a zero seed/counter pair
    // and produces well-separated streams for Swerling 1..4.
    let s = seed
        .wrapping_add(salt.wrapping_mul(0xD1B54A32D192ED03))
        .wrapping_mul(0x9E3779B97F4A7C15);
    let c = counter
        .wrapping_add(0xBF58476D1CE4E5B9)
        .wrapping_mul(0x94D049BB133111EB);
    s ^ c.rotate_left(17) ^ counter
}

// ---------------------------------------------------------------------------
// Public-proxy reference tables (seeded_public_proxy_v1).
//
// These tables are PROXIES patterned after published aggregate
// distributions and grid shapes for the named target classes. They are
// not measured equivalents and the module docs forbid claiming
// platform-specific signature truth from them.
// ---------------------------------------------------------------------------

/// 12-az × 3-el table (every 30 deg azimuth; 0/15/30 deg elevation).
/// Broadside (90 / 270 deg) bulges up to ~-10 dBsm; nose / tail dips
/// to ~-25 dBsm; elevation modulates by a few dB.
fn small_fixed_wing_uas_x_band_vv() -> RcsLookup {
    // Azimuth pattern in dBsm at 0 deg elevation, 30-deg steps starting
    // at nose-on. Broadside is index 3 / 9 (90 deg / 270 deg).
    let az_pattern: [f64; 12] = [
        -25.0, -22.0, -16.0, -10.0, -16.0, -22.0, -25.0, -22.0, -16.0, -10.0, -16.0, -22.0,
    ];
    let n_az = az_pattern.len();
    let elevations: [f64; 3] = [0.0, 15.0, 30.0];
    // Elevation offsets (dB) applied uniformly across azimuth; higher
    // elevations look slightly weaker on a fixed-wing planform.
    let el_offset: [f64; 3] = [0.0, -1.5, -3.0];

    let mut rcs_dbsm = Vec::with_capacity(n_az * elevations.len());
    for az_val in &az_pattern {
        for el_off in &el_offset {
            rcs_dbsm.push(*az_val + *el_off);
        }
    }

    RcsLookup {
        target_class: "fixed-wing-uas-small".to_string(),
        frequency_ghz: 10.0,
        polarization: Polarization::Vv,
        aspect_grid: AspectGrid {
            azimuth_deg: (0..12).map(|i| (i as f64) * 30.0).collect(),
            elevation_deg: elevations.to_vec(),
        },
        rcs_dbsm,
        fluctuation: SwerlingModel::Swerling1,
        citation: "MDPI Drones 2023, 7(1):39 (small fixed-wing UAV RCS aggregate distribution)"
            .to_string(),
        citation_url: Some("https://www.mdpi.com/2504-446X/7/1/39".to_string()),
    }
}

/// 12-az × 3-el table for a single large bird. Body-only return is
/// low (~-30 dBsm); broadside wing-flash spikes the cross-section to
/// ~-15 dBsm.
fn single_large_bird_x_band_hh() -> RcsLookup {
    let az_pattern: [f64; 12] = [
        -30.0, -28.0, -22.0, -15.0, -22.0, -28.0, -30.0, -28.0, -22.0, -15.0, -22.0, -28.0,
    ];
    let elevations: [f64; 3] = [0.0, 15.0, 30.0];
    let el_offset: [f64; 3] = [0.0, -1.0, -2.5];

    let mut rcs_dbsm = Vec::with_capacity(az_pattern.len() * elevations.len());
    for az_val in &az_pattern {
        for el_off in &el_offset {
            rcs_dbsm.push(*az_val + *el_off);
        }
    }

    RcsLookup {
        target_class: "bird-large-single".to_string(),
        frequency_ghz: 10.0,
        polarization: Polarization::Hh,
        aspect_grid: AspectGrid {
            azimuth_deg: (0..12).map(|i| (i as f64) * 30.0).collect(),
            elevation_deg: elevations.to_vec(),
        },
        rcs_dbsm,
        fluctuation: SwerlingModel::Swerling3,
        citation: "Rahman & Robertson, Nature Sci Rep 8:17396 (2018) — Radar micro-Doppler signatures of drones and birds at K-band and W-band".to_string(),
        citation_url: Some("https://www.nature.com/articles/s41598-018-35880-9".to_string()),
    }
}

/// 12-az × 3-el table for a quadrotor. More omnidirectional than a
/// fixed-wing planform; small ±2 dB variation around -20 dBsm body
/// return with mild broadside bias. Pulse-to-pulse fluctuation (rotor
/// chopping) is captured by Swerling 2.
fn quadrotor_x_band_vv() -> RcsLookup {
    let az_pattern: [f64; 12] = [
        -22.0, -21.5, -20.5, -19.5, -20.5, -21.5, -22.0, -21.5, -20.5, -19.5, -20.5, -21.5,
    ];
    let elevations: [f64; 3] = [0.0, 15.0, 30.0];
    let el_offset: [f64; 3] = [0.0, -0.5, -1.5];

    let mut rcs_dbsm = Vec::with_capacity(az_pattern.len() * elevations.len());
    for az_val in &az_pattern {
        for el_off in &el_offset {
            rcs_dbsm.push(*az_val + *el_off);
        }
    }

    RcsLookup {
        target_class: "quadrotor".to_string(),
        frequency_ghz: 10.0,
        polarization: Polarization::Vv,
        aspect_grid: AspectGrid {
            azimuth_deg: (0..12).map(|i| (i as f64) * 30.0).collect(),
            elevation_deg: elevations.to_vec(),
        },
        rcs_dbsm,
        fluctuation: SwerlingModel::Swerling2,
        citation: "Ezuma et al., arXiv:2102.11954 (UAV RF and RCS statistical recognition)"
            .to_string(),
        citation_url: Some("https://arxiv.org/abs/2102.11954".to_string()),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn small_grid_table() -> RcsLookup {
        // 3 azimuth × 2 elevation; values chosen so interpolation tests
        // are easy to reason about.
        RcsLookup {
            target_class: "test".to_string(),
            frequency_ghz: 10.0,
            polarization: Polarization::Vv,
            aspect_grid: AspectGrid {
                azimuth_deg: vec![0.0, 90.0, 180.0],
                elevation_deg: vec![0.0, 30.0],
            },
            // Order is [az0_el0, az0_el30, az90_el0, az90_el30, az180_el0, az180_el30].
            rcs_dbsm: vec![-20.0, -22.0, -10.0, -12.0, -20.0, -22.0],
            fluctuation: SwerlingModel::Swerling0,
            citation: "synthetic fixture".to_string(),
            citation_url: None,
        }
    }

    fn assert_close(a: f64, b: f64, eps: f64) {
        assert!(
            (a - b).abs() < eps,
            "expected {a} ~= {b} (tolerance {eps})",
        );
    }

    #[test]
    fn evaluate_static_returns_grid_value_at_grid_point() {
        // Grid corner: aspect=0, elevation=0 should be exactly -20.
        let mut rcs = Rcs::empty();
        rcs.add_table(small_grid_table());
        let v = rcs.evaluate_static("test", 0.0, 0.0, 10.0, Polarization::Vv);
        assert_close(v, -20.0, 1e-12);

        // Another corner: aspect=90, elevation=30 should be exactly -12.
        let v2 = rcs.evaluate_static("test", 90.0, 30.0, 10.0, Polarization::Vv);
        assert_close(v2, -12.0, 1e-12);
    }

    #[test]
    fn bilinear_interp_midway_between_grid_points_is_mid_value() {
        let mut rcs = Rcs::empty();
        rcs.add_table(small_grid_table());
        // Midway in azimuth (45 between 0 and 90) at elevation 0:
        // (-20 + -10) / 2 = -15.
        let v = rcs.evaluate_static("test", 45.0, 0.0, 10.0, Polarization::Vv);
        assert_close(v, -15.0, 1e-12);

        // Midway in elevation (15 between 0 and 30) at azimuth 0:
        // (-20 + -22) / 2 = -21.
        let v2 = rcs.evaluate_static("test", 0.0, 15.0, 10.0, Polarization::Vv);
        assert_close(v2, -21.0, 1e-12);

        // Center of the (0..90, 0..30) cell: average of the four
        // corners (-20, -22, -10, -12) = -16.
        let v3 = rcs.evaluate_static("test", 45.0, 15.0, 10.0, Polarization::Vv);
        assert_close(v3, -16.0, 1e-12);
    }

    #[test]
    fn polarization_exact_match_returns_that_table() {
        let mut rcs = Rcs::empty();
        let mut vv = small_grid_table();
        vv.polarization = Polarization::Vv;
        vv.rcs_dbsm = vec![-20.0, -20.0, -20.0, -20.0, -20.0, -20.0];
        let mut hh = small_grid_table();
        hh.polarization = Polarization::Hh;
        hh.rcs_dbsm = vec![-30.0, -30.0, -30.0, -30.0, -30.0, -30.0];
        rcs.add_table(vv);
        rcs.add_table(hh);

        let v_vv = rcs.evaluate_static("test", 45.0, 15.0, 10.0, Polarization::Vv);
        assert_close(v_vv, -20.0, 1e-12);
        let v_hh = rcs.evaluate_static("test", 45.0, 15.0, 10.0, Polarization::Hh);
        assert_close(v_hh, -30.0, 1e-12);
    }

    #[test]
    fn polarization_miss_returns_averaged_value() {
        let mut rcs = Rcs::empty();
        let mut vv = small_grid_table();
        vv.polarization = Polarization::Vv;
        vv.rcs_dbsm = vec![-20.0; 6];
        let mut hh = small_grid_table();
        hh.polarization = Polarization::Hh;
        hh.rcs_dbsm = vec![-30.0; 6];
        rcs.add_table(vv);
        rcs.add_table(hh);

        // Request Hv — neither table has it, fall back to mean(-20, -30) = -25.
        let v = rcs.evaluate_static("test", 45.0, 15.0, 10.0, Polarization::Hv);
        assert_close(v, -25.0, 1e-12);
    }

    #[test]
    fn frequency_nearest_in_log_space_wins() {
        let mut rcs = Rcs::empty();
        // 1 GHz table — flat -10 dBsm.
        let mut t1 = small_grid_table();
        t1.frequency_ghz = 1.0;
        t1.rcs_dbsm = vec![-10.0; 6];
        // 10 GHz table — flat -20 dBsm.
        let mut t10 = small_grid_table();
        t10.frequency_ghz = 10.0;
        t10.rcs_dbsm = vec![-20.0; 6];
        // 100 GHz table — flat -30 dBsm.
        let mut t100 = small_grid_table();
        t100.frequency_ghz = 100.0;
        t100.rcs_dbsm = vec![-30.0; 6];
        rcs.add_table(t1);
        rcs.add_table(t10);
        rcs.add_table(t100);

        // Geometric midpoint of (1, 100) GHz is 10 GHz in log-space,
        // so a request at sqrt(10*100)=31.6 should pick the 100 GHz
        // table (closer in log10: |log10(31.6) - log10(100)| < |log10(31.6)
        // - log10(10)|).
        let v = rcs.evaluate_static("test", 0.0, 0.0, 31.7, Polarization::Vv);
        assert_close(v, -30.0, 1e-12);

        // A request closer to 10 GHz in log space picks the 10 GHz table.
        let v2 = rcs.evaluate_static("test", 0.0, 0.0, 9.0, Polarization::Vv);
        assert_close(v2, -20.0, 1e-12);
    }

    #[test]
    fn swerling0_returns_deterministic_value() {
        let mut rcs = Rcs::empty();
        rcs.add_table(small_grid_table());
        let median = rcs.evaluate_static("test", 30.0, 10.0, 10.0, Polarization::Vv);
        let a = rcs.evaluate("test", 30.0, 10.0, 10.0, Polarization::Vv, 12345, 0);
        let b = rcs.evaluate("test", 30.0, 10.0, 10.0, Polarization::Vv, 12345, 17);
        let c = rcs.evaluate("test", 30.0, 10.0, 10.0, Polarization::Vv, 99, 9999);
        assert_close(a, median, 1e-12);
        assert_close(b, median, 1e-12);
        assert_close(c, median, 1e-12);
    }

    #[test]
    fn swerling1_correlated_within_scan_decorrelated_across_scans() {
        let mut table = small_grid_table();
        table.fluctuation = SwerlingModel::Swerling1;
        let mut rcs = Rcs::empty();
        rcs.add_table(table);

        let same_scan_a = rcs.evaluate("test", 30.0, 10.0, 10.0, Polarization::Vv, 7, 0);
        let same_scan_b = rcs.evaluate("test", 30.0, 10.0, 10.0, Polarization::Vv, 7, 5);
        // Both pulses inside scan 0 -> identical fluctuation.
        assert_close(same_scan_a, same_scan_b, 1e-12);

        // Pulse SWERLING_DEFAULT_SCAN_SIZE is in scan 1 -> different.
        let next_scan = rcs.evaluate(
            "test",
            30.0,
            10.0,
            10.0,
            Polarization::Vv,
            7,
            SWERLING_DEFAULT_SCAN_SIZE,
        );
        assert!(
            (same_scan_a - next_scan).abs() > 1e-6,
            "expected Swerling 1 to decorrelate across scans (got {same_scan_a} vs {next_scan})",
        );
    }

    #[test]
    fn swerling2_decorrelates_pulse_to_pulse() {
        let mut table = small_grid_table();
        table.fluctuation = SwerlingModel::Swerling2;
        let mut rcs = Rcs::empty();
        rcs.add_table(table);

        let p0 = rcs.evaluate("test", 30.0, 10.0, 10.0, Polarization::Vv, 99, 0);
        let p1 = rcs.evaluate("test", 30.0, 10.0, 10.0, Polarization::Vv, 99, 1);
        let p2 = rcs.evaluate("test", 30.0, 10.0, 10.0, Polarization::Vv, 99, 2);
        assert!(
            (p0 - p1).abs() > 1e-6 && (p1 - p2).abs() > 1e-6,
            "Swerling 2 should decorrelate pulse-to-pulse (got {p0}, {p1}, {p2})",
        );
    }

    #[test]
    fn determinism_same_call_same_value() {
        let mut table = small_grid_table();
        table.fluctuation = SwerlingModel::Swerling4;
        let mut rcs = Rcs::empty();
        rcs.add_table(table);
        let a = rcs.evaluate("test", 30.0, 10.0, 10.0, Polarization::Vv, 0xC0FFEE, 9);
        let b = rcs.evaluate("test", 30.0, 10.0, 10.0, Polarization::Vv, 0xC0FFEE, 9);
        assert_eq!(a.to_bits(), b.to_bits());
    }

    #[test]
    fn aspect_wraps_around_360() {
        let mut rcs = Rcs::empty();
        rcs.add_table(small_grid_table());
        let a0 = rcs.evaluate_static("test", 0.0, 0.0, 10.0, Polarization::Vv);
        let a360 = rcs.evaluate_static("test", 360.0, 0.0, 10.0, Polarization::Vv);
        let a720 = rcs.evaluate_static("test", 720.0, 0.0, 10.0, Polarization::Vv);
        let a_neg = rcs.evaluate_static("test", -360.0, 0.0, 10.0, Polarization::Vv);
        assert_close(a0, a360, 1e-12);
        assert_close(a0, a720, 1e-12);
        assert_close(a0, a_neg, 1e-12);
    }

    #[test]
    fn out_of_grid_elevation_clamps_to_nearest_edge() {
        let mut rcs = Rcs::empty();
        rcs.add_table(small_grid_table());
        // Grid covers elevation 0..30. -45 should clamp to 0, +90 should
        // clamp to 30.
        let v_low = rcs.evaluate_static("test", 0.0, -45.0, 10.0, Polarization::Vv);
        let v_at_zero = rcs.evaluate_static("test", 0.0, 0.0, 10.0, Polarization::Vv);
        assert_close(v_low, v_at_zero, 1e-12);

        let v_hi = rcs.evaluate_static("test", 0.0, 90.0, 10.0, Polarization::Vv);
        let v_at_top = rcs.evaluate_static("test", 0.0, 30.0, 10.0, Polarization::Vv);
        assert_close(v_hi, v_at_top, 1e-12);
    }

    #[test]
    fn seeded_public_proxy_v1_has_three_valid_cited_tables() {
        let rcs = Rcs::seeded_public_proxy_v1();
        assert!(rcs.tables.len() >= 3, "expected at least 3 reference tables");
        let class_names: Vec<&str> =
            rcs.tables.iter().map(|t| t.target_class.as_str()).collect();
        for needed in ["fixed-wing-uas-small", "bird-large-single", "quadrotor"] {
            assert!(
                class_names.contains(&needed),
                "missing seeded table for {needed}",
            );
        }
        for t in &rcs.tables {
            assert!(t.is_valid(), "table {} is invalid", t.target_class);
            assert!(
                !t.citation.is_empty(),
                "citation must not be empty for {}",
                t.target_class,
            );
            assert!(
                t.aspect_grid.azimuth_deg.len() >= 4,
                "aspect grid too coarse for {}",
                t.target_class,
            );
            assert!(
                t.aspect_grid.elevation_deg.len() >= 2,
                "elevation grid too coarse for {}",
                t.target_class,
            );
            // Sanity: every value in dBsm and finite.
            for v in &t.rcs_dbsm {
                assert!(v.is_finite(), "non-finite RCS in {}", t.target_class);
            }
        }
    }

    #[test]
    fn unknown_target_class_returns_neg_infinity() {
        let rcs = Rcs::seeded_public_proxy_v1();
        let v = rcs.evaluate_static("nonexistent", 0.0, 0.0, 10.0, Polarization::Vv);
        assert!(v.is_infinite() && v.is_sign_negative());
        let v2 = rcs.evaluate("nonexistent", 0.0, 0.0, 10.0, Polarization::Vv, 0, 0);
        assert!(v2.is_infinite() && v2.is_sign_negative());
    }

    #[test]
    fn fluctuation_overlay_preserves_finite_dbsm() {
        let rcs = Rcs::seeded_public_proxy_v1();
        for pulse in 0..64 {
            let v = rcs.evaluate(
                "quadrotor",
                42.5,
                5.0,
                10.0,
                Polarization::Vv,
                0xDEAD_BEEF,
                pulse,
            );
            assert!(v.is_finite(), "non-finite dBsm at pulse {pulse}");
        }
    }
}
