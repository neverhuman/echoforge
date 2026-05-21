//! DetectorGraph runtime: chain detectors that consume a `RangeDoppler`,
//! merge their `DetectionEvent` outputs, deduplicate by
//! `(range_bin, doppler_bin)` and aggregate magnitudes.

use std::collections::HashMap;

use crate::detectors::{
    blob_rd::BlobRdDetector, ca_cfar::CaCfarDetector, go_cfar::GoCfarDetector,
    micro_doppler_feature::MicroDopplerDetector, os_cfar::OsCfarDetector, so_cfar::SoCfarDetector,
    DetectionEvent, DetectionKind, Detector, RangeDoppler,
};

/// Adapter that lifts a 1-D detector to operate on a `RangeDoppler` by
/// reducing each range row to a single power value (either max or mean over
/// doppler bins). The reduced 1-D vector is then handed to the wrapped
/// detector. Resulting events retain `doppler_bin = None`.
pub struct RangeProjectedDetector<D: Detector<Input = Vec<f32>>> {
    pub inner: D,
    pub reducer: RowReducer,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RowReducer {
    Max,
    Mean,
}

impl<D: Detector<Input = Vec<f32>>> RangeProjectedDetector<D> {
    pub fn new(inner: D, reducer: RowReducer) -> Self {
        Self { inner, reducer }
    }

    fn reduce(&self, rd: &RangeDoppler) -> Vec<f32> {
        let mut out = Vec::with_capacity(rd.range_bins);
        for r in 0..rd.range_bins {
            let row = &rd.data[r * rd.doppler_bins..(r + 1) * rd.doppler_bins];
            let v = match self.reducer {
                RowReducer::Max => row.iter().copied().fold(f32::MIN, f32::max),
                RowReducer::Mean => row.iter().sum::<f32>() / rd.doppler_bins.max(1) as f32,
            };
            out.push(v);
        }
        out
    }
}

impl<D: Detector<Input = Vec<f32>>> Detector for RangeProjectedDetector<D> {
    type Input = RangeDoppler;

    fn detect(&self, rd: &Self::Input) -> Vec<DetectionEvent> {
        let projected = self.reduce(rd);
        self.inner.detect(&projected)
    }

    fn kind(&self) -> DetectionKind {
        self.inner.kind()
    }
}

/// Convenience aliases used by the runtime to box detectors.
pub type BoxedRdDetector = Box<dyn Detector<Input = RangeDoppler>>;

/// Aggregate output of the detector graph: deduplicated events plus per-kind
/// counts before deduplication (so callers can see how many raw events each
/// detector produced).
#[derive(Debug, Clone, PartialEq)]
pub struct FusedDetections {
    pub events: Vec<DetectionEvent>,
    pub per_detector: HashMap<DetectionKind, usize>,
}

/// Runtime that owns a chain of detectors and runs them sequentially against
/// a `RangeDoppler` input.
#[derive(Default)]
pub struct DetectorGraphRuntime {
    pub nodes: Vec<BoxedRdDetector>,
}

impl DetectorGraphRuntime {
    pub fn new(nodes: Vec<BoxedRdDetector>) -> Self {
        Self { nodes }
    }

    pub fn push(&mut self, node: BoxedRdDetector) {
        self.nodes.push(node);
    }

    /// Convenience builder: wrap a CA-CFAR detector with a row-max projector.
    pub fn add_ca_cfar(&mut self, det: CaCfarDetector) {
        self.nodes
            .push(Box::new(RangeProjectedDetector::new(det, RowReducer::Max)));
    }
    pub fn add_os_cfar(&mut self, det: OsCfarDetector) {
        self.nodes
            .push(Box::new(RangeProjectedDetector::new(det, RowReducer::Max)));
    }
    pub fn add_go_cfar(&mut self, det: GoCfarDetector) {
        self.nodes
            .push(Box::new(RangeProjectedDetector::new(det, RowReducer::Max)));
    }
    pub fn add_so_cfar(&mut self, det: SoCfarDetector) {
        self.nodes
            .push(Box::new(RangeProjectedDetector::new(det, RowReducer::Max)));
    }
    pub fn add_blob(&mut self, det: BlobRdDetector) {
        self.nodes.push(Box::new(det));
    }
    pub fn add_micro_doppler(&mut self, det: MicroDopplerDetector) {
        self.nodes.push(Box::new(det));
    }

    pub fn run(&self, rd: &RangeDoppler) -> FusedDetections {
        let mut per_detector: HashMap<DetectionKind, usize> = HashMap::new();
        let mut keyed: HashMap<(usize, Option<usize>), DetectionEvent> = HashMap::new();

        for node in &self.nodes {
            let kind = node.kind();
            let events = node.detect(rd);
            *per_detector.entry(kind).or_insert(0) += events.len();
            for ev in events {
                let key = (ev.range_bin, ev.doppler_bin);
                keyed
                    .entry(key)
                    .and_modify(|existing| {
                        // Aggregate magnitude by taking the maximum (dB).
                        if ev.magnitude_db > existing.magnitude_db {
                            existing.magnitude_db = ev.magnitude_db;
                        }
                    })
                    .or_insert(ev);
            }
        }

        let mut events: Vec<DetectionEvent> = keyed.into_values().collect();
        events.sort_by(|a, b| {
            a.range_bin
                .cmp(&b.range_bin)
                .then_with(|| a.doppler_bin.cmp(&b.doppler_bin))
        });
        FusedDetections {
            events,
            per_detector,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cfar::CfarParams;
    use crate::detectors::os_cfar::OsCfarParams;

    fn rd_with_target_row(
        range_bins: usize,
        doppler_bins: usize,
        target_row: usize,
    ) -> RangeDoppler {
        let mut rd = RangeDoppler::new(
            range_bins,
            doppler_bins,
            vec![1.0f32; range_bins * doppler_bins],
        );
        for d in 0..doppler_bins {
            rd.set(target_row, d, 50.0);
        }
        rd
    }

    #[test]
    fn graph_chains_ca_and_os_cfar_and_dedups() {
        let rd = rd_with_target_row(64, 8, 32);
        let mut graph = DetectorGraphRuntime::default();
        graph.add_ca_cfar(CaCfarDetector::new(CfarParams::new(8, 2, 1e-3)));
        graph.add_os_cfar(OsCfarDetector::new(OsCfarParams::new(8, 2, 1e-3, 0.75)));

        let fused = graph.run(&rd);
        let ca = fused
            .per_detector
            .get(&DetectionKind::CaCfar)
            .copied()
            .unwrap_or(0);
        let os = fused
            .per_detector
            .get(&DetectionKind::OsCfar)
            .copied()
            .unwrap_or(0);

        // Both should detect the strong target row.
        assert!(ca >= 1, "CA should fire at least once, got {ca}");
        assert!(os >= 1, "OS should fire at least once, got {os}");

        // Dedup: the fused event count must be >= max(individual counts)
        // and <= sum (no double counting per (range,doppler) key).
        assert!(fused.events.len() >= ca.max(os));
        assert!(fused.events.len() <= ca + os);

        // Target row should appear in deduped output.
        assert!(fused.events.iter().any(|e| e.range_bin == 32));
    }

    #[test]
    fn graph_dedups_identical_events() {
        // Two CA-CFAR detectors with identical params produce identical
        // events; the fused set must collapse them.
        let rd = rd_with_target_row(64, 4, 20);
        let mut graph = DetectorGraphRuntime::default();
        graph.add_ca_cfar(CaCfarDetector::new(CfarParams::new(8, 2, 1e-3)));
        graph.add_ca_cfar(CaCfarDetector::new(CfarParams::new(8, 2, 1e-3)));
        let fused = graph.run(&rd);
        // per-detector count is summed across both nodes (same kind).
        let ca_total = fused
            .per_detector
            .get(&DetectionKind::CaCfar)
            .copied()
            .unwrap_or(0);
        assert!(ca_total >= 2, "expected both CA-CFAR nodes to fire");
        let target_events = fused.events.iter().filter(|e| e.range_bin == 20).count();
        assert_eq!(target_events, 1, "duplicates should collapse to one event");
    }
}
