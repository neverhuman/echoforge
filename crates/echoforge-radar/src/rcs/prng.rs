//! Deterministic PRNG and Swerling fluctuation helpers.

use std::f64::consts::PI;
use super::{SwerlingModel, SWERLING_DEFAULT_SCAN_SIZE};

/// Mirrors the `SplitMix64` used elsewhere in the crate for determinism.
#[derive(Debug, Clone)]
pub(super) struct SplitMix64 {
    state: u64,
    cached_normal: Option<f64>,
}

impl SplitMix64 {
    pub(super) fn new(seed: u64) -> Self {
        Self { state: seed, cached_normal: None }
    }

    pub(super) fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }

    pub(super) fn unit(&mut self) -> f64 {
        let bits = self.next_u64() >> 11;
        (bits as f64) / ((1u64 << 53) as f64)
    }

    pub(super) fn normal(&mut self) -> f64 {
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

    /// Sample from `Exponential(1)` — Rayleigh-power kernel for Swerling 1/2.
    pub(super) fn exponential_unit(&mut self) -> f64 {
        let u = self.unit().clamp(1e-12, 1.0 - 1e-12);
        -u.ln()
    }

    /// Sample from `chi-squared(k=4) / 4` — Swerling 3/4 amplitude kernel.
    pub(super) fn chi_squared4_norm(&mut self) -> f64 {
        let mut acc = 0.0;
        for _ in 0..4 {
            let z = self.normal();
            acc += z * z;
        }
        acc / 4.0
    }
}

// ---------------------------------------------------------------------------
// Math helpers (private to the crate, pub(super) for sub-modules).
// ---------------------------------------------------------------------------

pub(super) fn wrap_deg_360(deg: f64) -> f64 {
    let mut v = deg % 360.0;
    if v < 0.0 { v += 360.0; }
    if v >= 360.0 { v -= 360.0; }
    v
}

pub(super) fn log10_safe(freq_ghz: f64) -> f64 {
    if freq_ghz > 0.0 { freq_ghz.log10() } else { f64::NEG_INFINITY }
}

/// Bracket `value` on a strictly-ascending grid. Returns `(lo_idx, hi_idx, frac)`.
/// Out-of-range values clamp to the nearest edge.
pub(super) fn clamping_bracket(grid: &[f64], value: f64) -> (usize, usize, f64) {
    debug_assert!(!grid.is_empty());
    if grid.len() == 1 { return (0, 0, 0.0); }
    if value <= grid[0] { return (0, 0, 0.0); }
    let last = grid.len() - 1;
    if value >= grid[last] { return (last, last, 0.0); }
    for i in 0..last {
        if value >= grid[i] && value <= grid[i + 1] {
            let span = grid[i + 1] - grid[i];
            let frac = if span > 0.0 { (value - grid[i]) / span } else { 0.0 };
            return (i, i + 1, frac);
        }
    }
    (last, last, 0.0)
}

/// Bracket `value` on an ascending azimuth grid with wrap-around.
/// Caller must pre-wrap `value` into `[0, 360)`.
pub(super) fn wrapping_bracket(grid: &[f64], value: f64) -> (usize, usize, f64) {
    debug_assert!(!grid.is_empty());
    let n = grid.len();
    if n == 1 { return (0, 0, 0.0); }
    let first = grid[0];
    let last = grid[n - 1];
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
    let span = (first + 360.0) - last;
    let v = if value < first { value + 360.0 } else { value };
    let frac = if span > 0.0 { (v - last) / span } else { 0.0 };
    (n - 1, 0, frac)
}

/// Apply a Swerling fluctuation factor to a median dBsm value.
pub(super) fn apply_swerling(
    median_dbsm: f64,
    model: SwerlingModel,
    seed: u64,
    pulse_index: usize,
) -> f64 {
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
    let f = linear_factor.max(1e-30);
    median_dbsm + 10.0 * f.log10()
}

fn mix_seed(seed: u64, counter: u64, salt: u64) -> u64 {
    let s = seed
        .wrapping_add(salt.wrapping_mul(0xD1B54A32D192ED03))
        .wrapping_mul(0x9E3779B97F4A7C15);
    let c = counter
        .wrapping_add(0xBF58476D1CE4E5B9)
        .wrapping_mul(0x94D049BB133111EB);
    s ^ c.rotate_left(17) ^ counter
}
