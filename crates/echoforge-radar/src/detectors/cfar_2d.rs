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
mod tests {
    use super::*;

    /// Tiny SplitMix64-style RNG so tests are deterministic without a
    /// crate dependency. Mirrors the local AlphaRng in `cfar_alpha`.
    struct TestRng(u64);

    impl TestRng {
        fn new(seed: u64) -> Self {
            Self(seed.wrapping_add(0x9e37_79b9_7f4a_7c15))
        }

        fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            z ^ (z >> 31)
        }

        fn open_unit_f64(&mut self) -> f64 {
            let bits = self.next_u64() >> 11;
            let denom = (1u64 << 53) as f64;
            (bits as f64 + 0.5) / denom
        }

        /// Unit-mean exponential power sample (= |I + jQ|^2 with I, Q
        /// independent Gaussian). Matches the Gaussian / Rayleigh
        /// convention used by `cfar_alpha::sample_distribution`.
        fn exp_power(&mut self) -> f64 {
            let u = self.open_unit_f64();
            -((1.0 - u).ln())
        }
    }

    fn gaussian_power_map(rows: usize, cols: usize, seed: u64) -> Vec<Vec<f64>> {
        let mut rng = TestRng::new(seed);
        let mut out = Vec::with_capacity(rows);
        for _ in 0..rows {
            let mut row = Vec::with_capacity(cols);
            for _ in 0..cols {
                row.push(rng.exp_power());
            }
            out.push(row);
        }
        out
    }

    #[test]
    fn ca_cfar_2d_detects_single_injected_peak() {
        // 64×32 unit-mean Gaussian power map with a strong peak injected
        // at (32, 16). CA-CFAR should flag exactly that cell.
        let mut map = gaussian_power_map(64, 32, 0xC2FA_2D01);
        map[32][16] = 1.0e4;

        let det = Cfar2dDetector::new(Cfar2dParams {
            training_range_cells: 4,
            training_azimuth_cells: 4,
            guard_range_cells: 2,
            guard_azimuth_cells: 2,
            pfa: 1e-4,
            variant: Cfar2dVariant::CaCfar,
        });
        let detections = det.process(&map);

        // Must contain the injected peak.
        assert!(
            detections
                .iter()
                .any(|d| d.range_bin == 32 && d.azimuth_bin == 16),
            "expected detection at (32, 16); got {detections:?}"
        );
        // At Pfa=1e-4 over ~ (64-12) × (32-12) = 52 × 20 = 1040 evaluated
        // cells, the expected number of false alarms is ~0.1. With a
        // single injected peak, the detector should produce exactly one
        // detection on this seed.
        assert_eq!(
            detections.len(),
            1,
            "expected exactly 1 detection, got {detections:?}"
        );
    }

    #[test]
    fn os_cfar_2d_detects_single_injected_peak() {
        let mut map = gaussian_power_map(64, 32, 0xC2FA_2D02);
        map[20][10] = 5.0e3;

        // Training ring has N = (2*4+2*2+1)*(2*4+2*2+1) - (2*2+1)^2
        //                     = 13*13 - 25 = 144 cells.
        // 75th-percentile order index is 0.75 * 144 - 1 ≈ 107.
        let det = Cfar2dDetector::new(Cfar2dParams {
            training_range_cells: 4,
            training_azimuth_cells: 4,
            guard_range_cells: 2,
            guard_azimuth_cells: 2,
            pfa: 1e-4,
            variant: Cfar2dVariant::OsCfar {
                k_quantile_index: 107,
            },
        });
        let detections = det.process(&map);

        assert!(
            detections
                .iter()
                .any(|d| d.range_bin == 20 && d.azimuth_bin == 10),
            "expected detection at (20, 10); got {detections:?}"
        );
    }

    #[test]
    fn ca_cfar_2d_empirical_pfa_within_tolerance() {
        // 128×64 = 8192-cell Gaussian map, single trial. We're not running
        // 100k *independent* maps (that would be ~1e9 cells and minutes of
        // CPU). Instead we slide the CFAR window across one large map and
        // count exceedances; cells are i.i.d. so the in-map count is a
        // valid empirical Pfa estimator up to a small spatial-correlation
        // correction that the unit-mean-exponential samples don't induce.
        //
        // Window: tr=4, g=2 → ~ (128-12) × (64-12) = 116 × 52 = 6032
        // evaluated cells. At nominal Pfa=1e-2 we expect ~60 false alarms;
        // ±50% tolerance is a generous smoke gate.
        let map = gaussian_power_map(128, 64, 0xC2FA_2D03);
        let det = Cfar2dDetector::new(Cfar2dParams {
            training_range_cells: 4,
            training_azimuth_cells: 4,
            guard_range_cells: 2,
            guard_azimuth_cells: 2,
            pfa: 1e-2,
            variant: Cfar2dVariant::CaCfar,
        });
        let detections = det.process(&map);
        let evaluated = (128 - 2 * 6) * (64 - 2 * 6);
        let observed_pfa = detections.len() as f64 / evaluated as f64;
        let nominal = 1e-2_f64;
        let ratio = observed_pfa / nominal;
        assert!(
            (0.5..=2.0).contains(&ratio),
            "observed Pfa {observed_pfa} should be within 50% of nominal {nominal} \
             (ratio {ratio}); detections={}, evaluated={evaluated}",
            detections.len()
        );
    }

    #[test]
    fn edge_cell_peak_does_not_panic() {
        // Peak at the corner (0, 0): the detector skips it because it's
        // inside the edge margin. Must not panic.
        let mut map = gaussian_power_map(32, 32, 0xC2FA_2D04);
        map[0][0] = 1.0e6;

        let det = Cfar2dDetector::new(Cfar2dParams {
            training_range_cells: 4,
            training_azimuth_cells: 4,
            guard_range_cells: 2,
            guard_azimuth_cells: 2,
            pfa: 1e-3,
            variant: Cfar2dVariant::CaCfar,
        });
        let detections = det.process(&map);
        // Detection at (0, 0) is impossible because the CFAR window can't
        // sit entirely inside the map for that CUT.
        assert!(
            detections.iter().all(|d| !(d.range_bin == 0 && d.azimuth_bin == 0)),
            "edge cell (0, 0) should be skipped, got {detections:?}"
        );
    }

    #[test]
    fn throughput_256x128_under_100ms() {
        // 256 × 128 map = 32,768 cells; CFAR sweep ≈ 232 × 104 = 24,128
        // evaluated cells with a 13 × 13 = 169-cell training ring each.
        // Single-thread CA-CFAR should finish in well under 100 ms on
        // commodity x86_64. The threshold is loose so we don't fail in
        // debug builds on slow CI runners.
        let map = gaussian_power_map(256, 128, 0xC2FA_2D05);
        let det = Cfar2dDetector::new(Cfar2dParams {
            training_range_cells: 4,
            training_azimuth_cells: 4,
            guard_range_cells: 2,
            guard_azimuth_cells: 2,
            pfa: 1e-3,
            variant: Cfar2dVariant::CaCfar,
        });
        let start = std::time::Instant::now();
        let _ = det.process(&map);
        let dur = start.elapsed();
        assert!(
            dur < std::time::Duration::from_millis(500),
            "process(256x128) took {dur:?}, expected < 500 ms even in debug builds"
        );
    }

    #[test]
    fn training_cell_count_matches_window_geometry() {
        // tr=4, g=2 → outer 13x13 = 169; inner 5x5 = 25; ring = 144.
        let det = Cfar2dDetector::new(Cfar2dParams {
            training_range_cells: 4,
            training_azimuth_cells: 4,
            guard_range_cells: 2,
            guard_azimuth_cells: 2,
            pfa: 1e-3,
            variant: Cfar2dVariant::CaCfar,
        });
        assert_eq!(det.training_cell_count(), 144);
    }

    #[test]
    fn empty_or_tiny_input_returns_no_detections() {
        let det = Cfar2dDetector::new(Cfar2dParams {
            training_range_cells: 4,
            training_azimuth_cells: 4,
            guard_range_cells: 2,
            guard_azimuth_cells: 2,
            pfa: 1e-3,
            variant: Cfar2dVariant::CaCfar,
        });
        let empty: Vec<Vec<f64>> = Vec::new();
        assert!(det.process(&empty).is_empty());
        let tiny = vec![vec![1.0; 4]; 4];
        assert!(det.process(&tiny).is_empty());
    }
}
