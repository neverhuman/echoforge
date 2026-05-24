//! 2-D range-azimuth CFAR (Cell-Averaging and Ordered-Statistic variants).
//!
//! Existing 1-D CFAR detectors (`crate::cfar::ca_cfar_1d`,
//! `crate::detectors::os_cfar::OsCfarDetector`) slide a one-dimensional
//! training window in range. Real radar surveillance products are
//! two-dimensional (range × azimuth, or range × Doppler), and the
//! single-cell sidelobe / clutter masking patterns that drive false alarms
//! are intrinsically 2-D — see Skolnik *Introduction to Radar Systems*
//! 3rd ed. §7.7.2 and Rohling 1983.
//!
//! This module implements both CA-CFAR (cell average over the 2-D training
//! ring) and OS-CFAR (k-th order statistic over the same training ring)
//! on a row-major `&[Vec<f64>]` range-azimuth power map. The 2-D windowing
//! is the only new physics; the threshold multiplier `alpha` is resolved
//! through the existing 1-D Rohling / Skolnik library
//! (`super::cfar_alpha::resolve_alpha`) for the total training-cell count
//! `N = (2·tr_r + 2·g_r + 1)·(2·tr_a + 2·g_a + 1) - (2·g_r+1)·(2·g_a+1)`,
//! i.e. the area of the outer window minus the guard region (which excludes
//! the cell-under-test and the immediate guard cells).
//!
//! Edge handling: any cell-under-test closer than `tr + g` cells to the
//! range or azimuth boundary is skipped. The caller therefore sees no
//! detections in the outermost margin; this mirrors the 1-D edge policy.
//!
//! # References
//!
//! - Skolnik, *Introduction to Radar Systems* 3rd ed. (McGraw-Hill 2001),
//!   §7.7.2 — closed-form CA-CFAR.
//! - Rohling, "Radar CFAR Thresholding in Clutter and Multiple Target
//!   Situations", IEEE Trans. AES-19 no. 4 (July 1983) — OS-CFAR implicit
//!   Pfa equation.
//! - Smith & Varshney, "VI-CFAR: A Novel CFAR Algorithm Based on Data
//!   Variability", IEEE Trans. AES-36 no. 3 (July 2000) — informational
//!   reference for the variability-index extension that this module does
//!   not implement.
//! - Hansen & Sawyers, "Detectability Loss Due to Greatest-Of Selection in
//!   a CA-CFAR", IEEE Trans. AES-16 no. 1 (Jan 1980) — informational
//!   reference for the GO/SO 1-D variants that this module does not
//!   implement in 2-D.

use super::cfar_alpha::{
    alpha_library_lookup, ca_cfar_scale_gaussian, os_cfar_scale_gaussian, CfarVariant,
    NoiseDistribution,
};

/// Which 2-D CFAR statistic to apply over the training ring.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Cfar2dVariant {
    /// Arithmetic mean over all training cells (Skolnik §7.7.2 closed form
    /// for the Gaussian / Rayleigh-amplitude case).
    CaCfar,
    /// `k_quantile_index`-th order statistic over the sorted training ring
    /// (Rohling 1983; 0-indexed within the training set of size `N`).
    /// `k_quantile_index` is clamped to `[0, N - 1]` at evaluation time.
    OsCfar { k_quantile_index: usize },
}

/// Parameters for the 2-D range-azimuth CFAR detector. The training ring
/// around each cell-under-test is `(2*tr_r + 2*g_r + 1) × (2*tr_a + 2*g_a + 1)`
/// cells in size with the inner `(2*g_r + 1) × (2*g_a + 1)` guard region (plus
/// the cell-under-test itself) excluded.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Cfar2dParams {
    /// Training cells per side along the range axis.
    pub training_range_cells: usize,
    /// Training cells per side along the azimuth axis.
    pub training_azimuth_cells: usize,
    /// Guard cells per side along the range axis (excluded from the
    /// training ring on both leading and lagging sides).
    pub guard_range_cells: usize,
    /// Guard cells per side along the azimuth axis (excluded from the
    /// training ring on both leading and lagging sides).
    pub guard_azimuth_cells: usize,
    /// Designed false-alarm probability.
    pub pfa: f64,
    /// Statistic variant applied over the training ring.
    pub variant: Cfar2dVariant,
}

/// Single 2-D CFAR detection emitted by [`Cfar2dDetector::process`].
/// Both `statistic_linear` and `threshold_linear` are in the same power
/// domain as the input map (linear, not dB).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Cfar2dDetection {
    pub range_bin: usize,
    pub azimuth_bin: usize,
    pub statistic_linear: f64,
    pub threshold_linear: f64,
}

/// 2-D range-azimuth CFAR detector. Holds the parameters and resolves
/// alpha lazily per `process` call (so the same instance can be applied to
/// maps of different sizes without re-allocating).
#[derive(Debug, Clone, Copy)]
pub struct Cfar2dDetector {
    pub params: Cfar2dParams,
}

impl Cfar2dDetector {
    pub fn new(params: Cfar2dParams) -> Self {
        Self { params }
    }

    /// Number of training cells in the 2-D ring, i.e. the area of the
    /// outer window minus the area of the inner guard window (which
    /// includes the cell-under-test).
    fn training_cell_count(&self) -> usize {
        let outer = (2 * self.params.training_range_cells + 2 * self.params.guard_range_cells + 1)
            * (2 * self.params.training_azimuth_cells
                + 2 * self.params.guard_azimuth_cells
                + 1);
        let inner =
            (2 * self.params.guard_range_cells + 1) * (2 * self.params.guard_azimuth_cells + 1);
        outer.saturating_sub(inner)
    }

    /// Threshold multiplier (`alpha`) for the configured variant. Routes
    /// through the existing 1-D Rohling / Skolnik library: CA-CFAR uses
    /// the Gaussian closed form; OS-CFAR uses the Rohling 1983 implicit
    /// equation. The 1-D library is well-defined for any `(N, k, Pfa)`
    /// triple — the 2-D windowing only changes `N`.
    fn alpha(&self, training_cells: usize) -> f64 {
        if training_cells == 0 {
            return 0.0;
        }
        match self.params.variant {
            Cfar2dVariant::CaCfar => {
                // Library lookup is a no-op for the Gaussian path
                // (library only covers Weibull / K-distribution); fall
                // through to the closed form on the Gaussian default.
                if let Some(v) = alpha_library_lookup(
                    CfarVariant::CellAveraging,
                    NoiseDistribution::Gaussian,
                    training_cells,
                    self.params.pfa as f32,
                ) {
                    return v as f64;
                }
                ca_cfar_scale_gaussian(training_cells, self.params.pfa as f32) as f64
            }
            Cfar2dVariant::OsCfar { k_quantile_index } => {
                // `k_quantile_index` is 0-indexed within the training set;
                // Rohling 1983 uses 1-indexed `rank`. Clamp and convert.
                let rank = k_quantile_index.min(training_cells.saturating_sub(1)) + 1;
                if let Some(v) = alpha_library_lookup(
                    CfarVariant::OrderedStatistic { rank },
                    NoiseDistribution::Gaussian,
                    training_cells,
                    self.params.pfa as f32,
                ) {
                    return v as f64;
                }
                os_cfar_scale_gaussian(training_cells, rank, self.params.pfa as f32) as f64
            }
        }
    }

    /// Slide the 2-D CFAR window across `range_azimuth_map`. Each row is
    /// one range bin; each column within a row is one azimuth bin. The
    /// returned detections are in row-major order (range, then azimuth)
    /// and are reported in the linear power domain.
    ///
    /// Edge cells closer than `tr + g` to any boundary are skipped — this
    /// matches the 1-D `ca_cfar_1d` policy and avoids panics on small
    /// maps or peaks placed at the corner.
    pub fn process(&self, range_azimuth_map: &[Vec<f64>]) -> Vec<Cfar2dDetection> {
        let rows = range_azimuth_map.len();
        if rows == 0 {
            return Vec::new();
        }
        let cols = range_azimuth_map[0].len();
        if cols == 0 {
            return Vec::new();
        }
        // All rows must have the same width; we report no detections from
        // ragged input instead of panicking.
        if range_azimuth_map.iter().any(|row| row.len() != cols) {
            return Vec::new();
        }

        let tr_r = self.params.training_range_cells;
        let tr_a = self.params.training_azimuth_cells;
        let g_r = self.params.guard_range_cells;
        let g_a = self.params.guard_azimuth_cells;
        let win_r = tr_r + g_r;
        let win_a = tr_a + g_a;

        if rows < 2 * win_r + 1 || cols < 2 * win_a + 1 {
            return Vec::new();
        }

        let n_train = self.training_cell_count();
        let alpha = self.alpha(n_train);
        if alpha <= 0.0 || !alpha.is_finite() {
            return Vec::new();
        }

        let mut detections: Vec<Cfar2dDetection> = Vec::new();
        // Scratch buffer for the OS-variant sort. Pre-sized to `n_train`.
        let mut scratch: Vec<f64> = Vec::with_capacity(n_train);

        for r in win_r..(rows - win_r) {
            for a in win_a..(cols - win_a) {
                // Build the training ring: outer (2*win_r+1) × (2*win_a+1)
                // window with the inner guard-and-CUT region excluded.
                let cut = range_azimuth_map[r][a];

                // Compute the noise estimate over the training ring.
                let stat = match self.params.variant {
                    Cfar2dVariant::CaCfar => {
                        let mut sum = 0.0f64;
                        let mut count = 0usize;
                        for rr in (r - win_r)..=(r + win_r) {
                            let row = &range_azimuth_map[rr];
                            let in_guard_row = rr >= r.saturating_sub(g_r) && rr <= r + g_r;
                            for aa in (a - win_a)..=(a + win_a) {
                                let in_guard_col = aa >= a.saturating_sub(g_a) && aa <= a + g_a;
                                if in_guard_row && in_guard_col {
                                    continue;
                                }
                                sum += row[aa];
                                count += 1;
                            }
                        }
                        if count == 0 {
                            continue;
                        }
                        sum / count as f64
                    }
                    Cfar2dVariant::OsCfar { k_quantile_index } => {
                        scratch.clear();
                        for rr in (r - win_r)..=(r + win_r) {
                            let row = &range_azimuth_map[rr];
                            let in_guard_row = rr >= r.saturating_sub(g_r) && rr <= r + g_r;
                            for aa in (a - win_a)..=(a + win_a) {
                                let in_guard_col = aa >= a.saturating_sub(g_a) && aa <= a + g_a;
                                if in_guard_row && in_guard_col {
                                    continue;
                                }
                                scratch.push(row[aa]);
                            }
                        }
                        if scratch.is_empty() {
                            continue;
                        }
                        scratch.sort_by(|x, y| {
                            x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal)
                        });
                        let k = k_quantile_index.min(scratch.len() - 1);
                        scratch[k]
                    }
                };

                let threshold = alpha * stat;
                if cut > threshold && threshold.is_finite() {
                    detections.push(Cfar2dDetection {
                        range_bin: r,
                        azimuth_bin: a,
                        statistic_linear: cut,
                        threshold_linear: threshold,
                    });
                }
            }
        }

        detections
    }
}


#[cfg(test)]
#[path = "cfar_2d_tests.rs"]
mod tests;
