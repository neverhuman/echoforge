use crate::backend::ArrayBackend;
use crate::cfar::CfarDecision;
use crate::chain::RadarChain;
use crate::ComplexSample;

#[derive(Debug, Clone, PartialEq)]
pub struct TrackingTrack {
    pub track_id: usize,
    pub detection_index: usize,
    pub statistic: f32,
    pub threshold: f32,
    pub confidence: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TrackingFusionReport {
    pub backend_name: &'static str,
    pub evaluated_cells: usize,
    pub detections: usize,
    pub tracks: Vec<TrackingTrack>,
}

#[derive(Debug, Default, Clone, Copy)]
pub struct TrackingFusionAdapter;

impl TrackingFusionAdapter {
    pub fn fuse_decisions(
        &self,
        backend_name: &'static str,
        decisions: &[CfarDecision],
    ) -> TrackingFusionReport {
        let mut evaluated_cells = 0usize;
        let mut tracks = Vec::new();

        for decision in decisions.iter().filter(|decision| decision.evaluated) {
            evaluated_cells += 1;

            if !decision.detected {
                continue;
            }

            let confidence = if decision.threshold.is_finite() && decision.threshold > 0.0 {
                decision.statistic / decision.threshold
            } else {
                0.0
            };

            tracks.push(TrackingTrack {
                track_id: tracks.len(),
                detection_index: decision.index,
                statistic: decision.statistic,
                threshold: decision.threshold,
                confidence,
            });
        }

        TrackingFusionReport {
            backend_name,
            evaluated_cells,
            detections: tracks.len(),
            tracks,
        }
    }

    pub fn fuse<B: ArrayBackend>(
        &self,
        chain: &RadarChain<B>,
        received: &[ComplexSample],
        reference: &[ComplexSample],
    ) -> TrackingFusionReport {
        let output = chain.detect(received, reference);
        self.fuse_decisions(chain.backend_name(), &output.cfar)
    }
}
