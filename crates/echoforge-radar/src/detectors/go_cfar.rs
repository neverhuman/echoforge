//! GO-CFAR (Greatest-Of CFAR).
//!
//! Compute the mean of the leading and lagging training windows separately
//! and threshold against `alpha * max(mean_lead, mean_lag)`. This makes the
//! detector robust at clutter edges where one side of the CUT has a much
//! higher noise floor.

use crate::cfar::{ca_cfar_scale, CfarParams};

use super::{magnitude_to_db, DetectionEvent, DetectionKind, Detector};

#[derive(Debug, Clone, Copy)]
pub struct GoCfarDetector {
    pub params: CfarParams,
}

impl GoCfarDetector {
    pub fn new(params: CfarParams) -> Self {
        Self { params }
    }
}

impl Detector for GoCfarDetector {
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
            let noise = mean_lead.max(mean_lag);
            let threshold = alpha * noise;
            let stat = input[index];
            if stat > threshold && threshold.is_finite() {
                events.push(DetectionEvent::new(
                    index,
                    None,
                    magnitude_to_db(stat as f64),
                    DetectionKind::GoCfar,
                ));
            }
        }
        events
    }

    fn kind(&self) -> DetectionKind {
        DetectionKind::GoCfar
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cfar::ca_cfar_1d;

    fn clutter_edge_power(
        n: usize,
        edge: usize,
        low: f32,
        high: f32,
        target: usize,
        peak: f32,
    ) -> Vec<f32> {
        let mut p = Vec::with_capacity(n);
        for i in 0..n {
            p.push(if i < edge { low } else { high });
        }
        p[target] = peak;
        p
    }

    #[test]
    fn go_cfar_does_not_falsely_detect_on_clutter_edge() {
        // Clutter edge halfway through; small jump in noise floor with NO
        // real target inserted. CA-CFAR averages across the edge and will
        // tend to over-detect; GO-CFAR takes the larger side and should
        // suppress those spurious detections.
        let n = 200usize;
        let edge = 100usize;
        let params = CfarParams::new(8, 2, 1e-3);
        // No injected target: just the clutter step.
        let mut p = vec![1.0f32; n];
        for i in edge..n {
            p[i] = 6.0;
        }

        let go = GoCfarDetector::new(params).detect(&p);
        let ca_decisions = ca_cfar_1d(&p, params);
        let ca_hits: usize = ca_decisions
            .iter()
            .filter(|d| d.evaluated && d.detected)
            .count();
        let go_hits: usize = go.len();

        assert!(
            go_hits <= ca_hits,
            "GO-CFAR should suppress edge false alarms (go={go_hits}, ca={ca_hits})"
        );
    }

    #[test]
    fn go_cfar_still_detects_target_atop_clutter_edge() {
        // Target sits firmly on the high side; both detectors should see it.
        let n = 200usize;
        let edge = 100usize;
        let target = 150usize;
        let p = clutter_edge_power(n, edge, 1.0, 6.0, target, 400.0);
        let params = CfarParams::new(8, 2, 1e-3);
        let go = GoCfarDetector::new(params).detect(&p);
        assert!(go.iter().any(|e| e.range_bin == target));
    }
}
