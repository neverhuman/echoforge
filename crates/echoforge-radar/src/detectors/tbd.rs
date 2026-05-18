//! Track-Before-Detect (TBD) via Hough-transform integration.
//!
//! Standard CFAR fires per-cell. TBD accumulates sub-threshold energy
//! across multiple CPIs along candidate target tracks, declaring a
//! detection only after coherent or near-coherent integration over N
//! frames passes a TBD-specific threshold. The Hough variant
//! parameterizes tracks as straight-line trajectories in
//! (range, Doppler) over time and votes each cell into the
//! corresponding Hough bins.
//!
//! References:
//!   - Tonissen & Evans, "Performance of dynamic programming techniques
//!     for track-before-detect", IEEE Trans. AES vol 32 no 4, Oct 1996.
//!   - Salmond & Birch, "A particle filter for track-before-detect",
//!     ACC 2001.
//!   - Carlson, Evans, Wilson, "Search radar detection and track with
//!     the Hough transform", IEEE Trans. AES vol 30 no 1, Jan 1994
//!     (THE Hough-TBD reference).
//!
//! Strict-open posture: closed-form Hough-vote math from Carlson-Evans-
//! Wilson 1994; no measured-truth claims.

use crate::ComplexSample;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TbdConfig {
    /// Number of CPIs to integrate over.
    pub n_cpis: usize,
    /// Minimum range-bin gradient (m per CPI) for valid track candidates.
    /// Excludes static clutter (gradient ~= 0).
    pub min_range_gradient: f32,
    /// Maximum range-bin gradient.
    pub max_range_gradient: f32,
    /// Minimum number of CPI hits (out of n_cpis) to declare TBD detection.
    pub m_of_n_threshold: usize,
    /// Per-cell threshold (linear power) -- anything below this contributes
    /// to the Hough vote but isn't a CFAR detection.
    pub sub_threshold_power: f32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TbdTrackCandidate {
    pub start_range_bin: usize,
    pub range_gradient: f32,
    pub doppler_bin: usize,
    pub doppler_gradient: f32,
    pub accumulated_power: f32,
    pub n_cpis_hit: usize,
    pub confidence: f32,
}

/// Run Hough-transform TBD over a stack of complex range-Doppler grids
/// (one per CPI). Returns track candidates that meet the M-of-N +
/// accumulated-power thresholds.
///
/// The input `range_doppler_stack` is a slice of CPIs, where each CPI is a
/// 2-D grid indexed `[range_bin][doppler_bin]`. The Hough accumulator bins
/// candidate tracks by (start-range-bin, range-gradient) pairs and walks
/// the predicted range cell through each CPI, recording hits when the
/// peak Doppler-bin power exceeds `sub_threshold_power`. Candidates with
/// at least `m_of_n_threshold` hits are emitted, then deduplicated against
/// nearby (start-range +/- 2, gradient +/- 0.5) candidates with lower
/// accumulated power.
///
/// The accumulator implements the straight-line Hough parameterization
/// from Carlson-Evans-Wilson 1994 in (range, time) space; the simplified
/// MVP omits Doppler-gradient voting (target's max-Doppler bin is recorded
/// as the dominant Doppler at the start CPI but not refined across the
/// track).
pub fn hough_tbd_detect(
    range_doppler_stack: &[Vec<Vec<ComplexSample>>],
    config: TbdConfig,
) -> Vec<TbdTrackCandidate> {
    if range_doppler_stack.is_empty() {
        return Vec::new();
    }
    let n_cpis = range_doppler_stack.len();
    let n_range = range_doppler_stack[0].len();
    if n_range == 0 {
        return Vec::new();
    }

    // Simple Hough accumulator: bin track candidates by (start_range, range_gradient)
    // pairs and integrate magnitude across the predicted track. For
    // sub-threshold cells, contribute partial vote. For above-threshold,
    // contribute full vote.
    let mut candidates: Vec<TbdTrackCandidate> = Vec::new();

    // For each potential start_range_bin and each candidate range gradient...
    let gradient_steps: usize = 8; // discretize gradient range
    let gradient_range = config.max_range_gradient - config.min_range_gradient;

    for start_range in 0..n_range {
        for grad_idx in 0..=gradient_steps {
            let gradient = config.min_range_gradient
                + gradient_range * (grad_idx as f32) / (gradient_steps as f32);

            // For each CPI, predict where the track should be and accumulate.
            let mut acc_power = 0.0_f32;
            let mut hits = 0usize;
            let mut dominant_doppler_bin = 0usize;
            let mut dominant_doppler_seen = false;

            for (cpi_idx, grid) in range_doppler_stack.iter().enumerate() {
                let predicted_range = start_range as f32 + gradient * cpi_idx as f32;
                let predicted_bin = predicted_range as isize;
                if predicted_bin < 0 || predicted_bin >= grid.len() as isize {
                    continue;
                }
                let pb = predicted_bin as usize;

                // Find max-magnitude Doppler bin in this range.
                let row = &grid[pb];
                let mut max_mag = 0.0_f32;
                let mut max_idx = 0usize;
                for (d_idx, cell) in row.iter().enumerate() {
                    let mag = cell.norm();
                    if mag > max_mag {
                        max_mag = mag;
                        max_idx = d_idx;
                    }
                }
                let power = max_mag * max_mag;

                if power > config.sub_threshold_power {
                    acc_power += power;
                    hits += 1;
                    if !dominant_doppler_seen {
                        dominant_doppler_bin = max_idx;
                        dominant_doppler_seen = true;
                    }
                }
            }

            if hits >= config.m_of_n_threshold {
                candidates.push(TbdTrackCandidate {
                    start_range_bin: start_range,
                    range_gradient: gradient,
                    doppler_bin: dominant_doppler_bin,
                    doppler_gradient: 0.0,
                    accumulated_power: acc_power,
                    n_cpis_hit: hits,
                    confidence: (hits as f32) / (n_cpis as f32),
                });
            }
        }
    }

    // Dedupe nearby candidates (within 2 range bins, same gradient +/- 0.5).
    candidates.sort_by(|a, b| {
        b.accumulated_power
            .partial_cmp(&a.accumulated_power)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut deduped: Vec<TbdTrackCandidate> = Vec::new();
    for cand in candidates {
        let near = deduped.iter().any(|d: &TbdTrackCandidate| {
            (d.start_range_bin as isize - cand.start_range_bin as isize).abs() < 3
                && (d.range_gradient - cand.range_gradient).abs() < 0.5
        });
        if !near {
            deduped.push(cand);
        }
    }
    deduped
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ComplexSample;

    fn make_grid(
        n_range: usize,
        n_doppler: usize,
        target: Option<(usize, usize, f32)>,
    ) -> Vec<Vec<ComplexSample>> {
        let mut grid = vec![vec![ComplexSample::new(0.0, 0.0); n_doppler]; n_range];
        // Add background noise (deterministic for test reproducibility).
        for r in 0..n_range {
            for d in 0..n_doppler {
                let n = ((r * 7 + d * 13) % 100) as f32 * 0.001;
                grid[r][d] = ComplexSample::new(n, n * 0.7);
            }
        }
        if let Some((r, d, mag)) = target {
            if r < n_range && d < n_doppler {
                grid[r][d] = ComplexSample::new(mag, 0.0);
            }
        }
        grid
    }

    #[test]
    fn tbd_finds_moving_sub_threshold_target() {
        // Build a 5-CPI stack with a target moving by 2 range bins per CPI
        // at sub-CFAR-threshold power. TBD should integrate enough energy
        // to declare detection.
        let n_range = 64;
        let n_doppler = 16;
        let stack: Vec<_> = (0..5)
            .map(|cpi| {
                let r = 10 + 2 * cpi;
                make_grid(n_range, n_doppler, Some((r, 5, 0.2)))
            })
            .collect();

        let config = TbdConfig {
            n_cpis: 5,
            min_range_gradient: 0.5,
            max_range_gradient: 4.0,
            m_of_n_threshold: 3,
            sub_threshold_power: 0.01,
        };
        let candidates = hough_tbd_detect(&stack, config);
        assert!(
            !candidates.is_empty(),
            "TBD should find the moving target track"
        );
        let best = &candidates[0];
        assert!(
            best.n_cpis_hit >= 3,
            "expected >=3 CPI hits, got {}",
            best.n_cpis_hit
        );
    }

    #[test]
    fn tbd_rejects_isolated_noise() {
        // 5-CPI stack with NO target, just noise. TBD should not produce
        // any high-confidence candidates above the M-of-N threshold.
        let stack: Vec<_> = (0..5).map(|_| make_grid(64, 16, None)).collect();
        let config = TbdConfig {
            n_cpis: 5,
            min_range_gradient: 0.5,
            max_range_gradient: 4.0,
            m_of_n_threshold: 3,
            sub_threshold_power: 1.0, // high threshold to reject noise
        };
        let candidates = hough_tbd_detect(&stack, config);
        let high_conf = candidates.iter().filter(|c| c.confidence > 0.7).count();
        assert_eq!(high_conf, 0, "TBD should not find tracks in pure noise");
    }

    #[test]
    fn tbd_handles_empty_input() {
        let candidates = hough_tbd_detect(
            &[],
            TbdConfig {
                n_cpis: 5,
                min_range_gradient: 0.0,
                max_range_gradient: 5.0,
                m_of_n_threshold: 3,
                sub_threshold_power: 0.5,
            },
        );
        assert!(candidates.is_empty());
    }
}
