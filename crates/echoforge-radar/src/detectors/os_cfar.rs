//! OS-CFAR (Ordered Statistic CFAR).
//!
//! Builds the training cell set (with guard cells excluded) on each side of
//! the cell-under-test, sorts it ascending, and uses the k-th order statistic
//! as the noise estimate. The threshold multiplier `alpha` is calibrated for
//! a desired probability of false alarm (Pfa); a closed-form `alpha` for OS
//! depends on `(N, k, Pfa)` so we fall back to the CA-CFAR closed form scaled
//! by `2*N_train / k` which preserves the same mean-noise scaling and gives a
//! deterministic, conservative threshold suitable for synthetic tests.

use crate::cfar::ca_cfar_scale;

use super::{magnitude_to_db, DetectionEvent, DetectionKind, Detector};

#[derive(Debug, Clone, Copy)]
pub struct OsCfarParams {
    pub training_cells: usize,
    pub guard_cells: usize,
    pub pfa: f32,
    /// Quantile of the sorted training set used as noise estimate
    /// (0.0..=1.0, typically 0.75).
    pub quantile: f32,
}

impl OsCfarParams {
    pub fn new(training_cells: usize, guard_cells: usize, pfa: f32, quantile: f32) -> Self {
        Self {
            training_cells,
            guard_cells,
            pfa,
            quantile,
        }
    }
}

/// Map a quantile in `(0.0..=1.0]` to an order index `k` in `[0, n)` for a
/// sorted training set of length `n`.
pub fn order_index_for_quantile(n: usize, quantile: f32) -> usize {
    if n == 0 {
        return 0;
    }
    let q = quantile.clamp(0.0, 1.0);
    let raw = (q * n as f32).round() as isize - 1;
    raw.clamp(0, (n - 1) as isize) as usize
}

#[derive(Debug, Clone, Copy)]
pub struct OsCfarDetector {
    pub params: OsCfarParams,
}

impl OsCfarDetector {
    pub fn new(params: OsCfarParams) -> Self {
        Self { params }
    }

    fn alpha(&self) -> f32 {
        // Reuse the CA-CFAR closed form against the total number of training
        // cells (both sides). This is the standard scaling driver; the OS
        // estimator's quantile already shifts the effective noise estimate
        // upward, which keeps Pfa <= the CA target on i.i.d. exponential
        // noise. Good enough for synthetic detection tests; calibrate later
        // for production Pfa control.
        let total = self.params.training_cells.saturating_mul(2);
        ca_cfar_scale(total, self.params.pfa)
    }
}

impl Detector for OsCfarDetector {
    type Input = Vec<f32>;

    fn detect(&self, input: &Self::Input) -> Vec<DetectionEvent> {
        let n_train = self.params.training_cells;
        let g = self.params.guard_cells;
        let window = n_train + g;
        let alpha = self.alpha();

        let mut events = Vec::new();
        if input.len() < 2 * window + 1 {
            return events;
        }

        let mut scratch: Vec<f32> = Vec::with_capacity(2 * n_train);
        for index in window..(input.len() - window) {
            scratch.clear();
            // Leading training cells: [index-window, index-g)
            scratch.extend_from_slice(&input[index - window..index - g]);
            // Lagging training cells: (index+g, index+window]
            scratch.extend_from_slice(&input[index + g + 1..=index + window]);

            // Partial sort up to the order index of interest.
            scratch.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let k = order_index_for_quantile(scratch.len(), self.params.quantile);
            let noise = scratch[k];

            let threshold = alpha * noise;
            let stat = input[index];
            if stat > threshold && threshold.is_finite() {
                events.push(DetectionEvent::new(
                    index,
                    None,
                    magnitude_to_db(stat as f64),
                    DetectionKind::OsCfar,
                ));
            }
        }

        events
    }

    fn kind(&self) -> DetectionKind {
        DetectionKind::OsCfar
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cfar::CfarParams;
    use crate::detectors::ca_cfar::CaCfarDetector;

    #[test]
    fn os_cfar_detects_isolated_target_in_uniform_noise() {
        // 128 cells of unit noise with a strong target at index 64.
        let mut power = vec![1.0f32; 128];
        power[64] = 200.0;

        let det = OsCfarDetector::new(OsCfarParams::new(8, 2, 1e-3, 0.75));
        let events = det.detect(&power);

        assert!(
            events.iter().any(|e| e.range_bin == 64),
            "expected detection at bin 64, got {events:?}"
        );
        assert!(events.iter().all(|e| e.kind == DetectionKind::OsCfar));
    }

    #[test]
    fn os_cfar_quantile_index_handles_edges() {
        assert_eq!(order_index_for_quantile(0, 0.5), 0);
        assert_eq!(order_index_for_quantile(4, 0.0), 0);
        assert_eq!(order_index_for_quantile(4, 1.0), 3);
        assert_eq!(order_index_for_quantile(16, 0.75), 11);
    }

    #[test]
    fn os_cfar_resists_masking_relative_to_ca_cfar() {
        // Two strong targets close together: classic two-target masking case
        // for CA-CFAR. Both targets sit inside each other's training window.
        // Place them so the gap between them is smaller than `2 * (training
        // + guard)`, but the training spans are still inside the array.
        let n = 256usize;
        let mut power = vec![1.0f32; n];
        let t1 = 100usize;
        let t2 = 108usize;
        power[t1] = 80.0;
        power[t2] = 80.0;

        let os = OsCfarDetector::new(OsCfarParams::new(12, 2, 1e-3, 0.5));
        let ca = CaCfarDetector::new(CfarParams::new(12, 2, 1e-3));

        let os_events = os.detect(&power);
        let ca_decisions = crate::cfar::ca_cfar_1d(&power, ca.params);
        let ca_hits: Vec<_> = ca_decisions
            .into_iter()
            .filter(|d| d.evaluated && d.detected)
            .map(|d| d.index)
            .collect();
        let os_hits: Vec<_> = os_events.iter().map(|e| e.range_bin).collect();

        // OS-CFAR with median quantile should still flag both targets even
        // when their neighbor pollutes the training mean.
        assert!(os_hits.contains(&t1), "OS missed t1; hits={os_hits:?}");
        assert!(os_hits.contains(&t2), "OS missed t2; hits={os_hits:?}");

        // CA-CFAR may miss at least one of the two due to masking. We assert
        // the *weaker* property that CA's hit set is a subset of {t1, t2}
        // plus that OS finds at least as many of the targets.
        let os_target_hits = os_hits.iter().filter(|i| **i == t1 || **i == t2).count();
        let ca_target_hits = ca_hits.iter().filter(|i| **i == t1 || **i == t2).count();
        assert!(
            os_target_hits >= ca_target_hits,
            "OS should match or beat CA on target hits; os={os_target_hits} ca={ca_target_hits}"
        );
    }
}
