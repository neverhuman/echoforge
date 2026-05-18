//! SO-CFAR (Smallest-Of CFAR).
//!
//! Threshold is `alpha * min(mean_lead, mean_lag)`. This makes the detector
//! robust against masking when another target sits in one of the training
//! windows: the contaminated side has a high mean, but the clean side keeps
//! the noise estimate low so the CUT can still cross threshold.

use crate::cfar::{ca_cfar_scale, CfarParams};

use super::{magnitude_to_db, DetectionEvent, DetectionKind, Detector};

#[derive(Debug, Clone, Copy)]
pub struct SoCfarDetector {
    pub params: CfarParams,
}

impl SoCfarDetector {
    pub fn new(params: CfarParams) -> Self {
        Self { params }
    }
}

impl Detector for SoCfarDetector {
    type Input = Vec<f32>;

    fn detect(&self, input: &Self::Input) -> Vec<DetectionEvent> {
        let n_train = self.params.training_cells;
        let g = self.params.guard_cells;
        let window = n_train + g;
        let alpha = ca_cfar_scale(n_train, self.params.pfa);
        let mut events = Vec::new();
        if input.len() < 2 * window + 1 || n_train == 0 {
            return events;
        }

        for index in window..(input.len() - window) {
            let lead = &input[index - window..index - g];
            let lag = &input[index + g + 1..=index + window];
            let mean_lead = lead.iter().sum::<f32>() / lead.len() as f32;
            let mean_lag = lag.iter().sum::<f32>() / lag.len() as f32;
            let noise = mean_lead.min(mean_lag);
            let threshold = alpha * noise;
            let stat = input[index];
            if stat > threshold && threshold.is_finite() {
                events.push(DetectionEvent::new(
                    index,
                    None,
                    magnitude_to_db(stat as f64),
                    DetectionKind::SoCfar,
                ));
            }
        }
        events
    }

    fn kind(&self) -> DetectionKind {
        DetectionKind::SoCfar
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cfar::ca_cfar_1d;
    use crate::detectors::ca_cfar::CaCfarDetector;

    #[test]
    fn so_cfar_handles_two_target_masking_better_than_ca_cfar() {
        // Two strong targets close enough that each lands in the other's
        // training window. CA-CFAR's mean is dragged up by the polluter and
        // may miss one of the two. SO-CFAR uses the cleaner side and should
        // still detect both.
        let n = 256usize;
        let t1 = 100usize;
        let t2 = 108usize;
        let mut power = vec![1.0f32; n];
        power[t1] = 80.0;
        power[t2] = 80.0;

        let params = CfarParams::new(12, 2, 1e-3);
        let so = SoCfarDetector::new(params).detect(&power);
        let so_hits: Vec<_> = so.iter().map(|e| e.range_bin).collect();
        assert!(so_hits.contains(&t1), "SO missed t1; hits={so_hits:?}");
        assert!(so_hits.contains(&t2), "SO missed t2; hits={so_hits:?}");

        let _ca_unused = CaCfarDetector::new(params);
        let ca_decisions = ca_cfar_1d(&power, params);
        let ca_hits: Vec<_> = ca_decisions
            .into_iter()
            .filter(|d| d.evaluated && d.detected)
            .map(|d| d.index)
            .collect();
        let so_target_hits = so_hits.iter().filter(|i| **i == t1 || **i == t2).count();
        let ca_target_hits = ca_hits.iter().filter(|i| **i == t1 || **i == t2).count();
        assert!(
            so_target_hits >= ca_target_hits,
            "SO should match or beat CA on target hits; so={so_target_hits} ca={ca_target_hits}"
        );
    }
}
