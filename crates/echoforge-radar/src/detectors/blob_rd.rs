//! Range-Doppler blob detector.
//!
//! Threshold the magnitude grid, then group connected components (4-way
//! adjacency) of above-threshold cells. Emit one event per cluster with the
//! magnitude-weighted centroid as `(range_bin, doppler_bin)` and the peak
//! magnitude as the event magnitude (in dB).

use std::collections::VecDeque;

use super::{magnitude_to_db, DetectionEvent, DetectionKind, Detector, RangeDoppler};

#[derive(Debug, Clone, Copy)]
pub struct BlobRdParams {
    /// Absolute magnitude threshold. Cells with `magnitude > threshold`
    /// participate in clustering.
    pub threshold: f32,
    /// Minimum cluster size (in cells) for a detection to be emitted.
    pub min_cluster_size: usize,
}

impl BlobRdParams {
    pub fn new(threshold: f32, min_cluster_size: usize) -> Self {
        Self {
            threshold,
            min_cluster_size,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct BlobRdDetector {
    pub params: BlobRdParams,
}

impl BlobRdDetector {
    pub fn new(params: BlobRdParams) -> Self {
        Self { params }
    }
}

impl Detector for BlobRdDetector {
    type Input = RangeDoppler;

    fn detect(&self, rd: &Self::Input) -> Vec<DetectionEvent> {
        let r = rd.range_bins;
        let d = rd.doppler_bins;
        if r == 0 || d == 0 {
            return Vec::new();
        }
        let mut visited = vec![false; r * d];
        let mut events = Vec::new();

        for r0 in 0..r {
            for d0 in 0..d {
                let idx = r0 * d + d0;
                if visited[idx] {
                    continue;
                }
                let v = rd.data[idx];
                if v <= self.params.threshold {
                    visited[idx] = true;
                    continue;
                }

                // BFS over above-threshold neighbors.
                let mut queue = VecDeque::new();
                queue.push_back((r0, d0));
                visited[idx] = true;

                let mut count = 0usize;
                let mut weight_sum = 0.0f64;
                let mut range_acc = 0.0f64;
                let mut doppler_acc = 0.0f64;
                let mut peak = f32::MIN;
                let mut peak_rc = (r0, d0);

                while let Some((rr, dd)) = queue.pop_front() {
                    let cell_idx = rr * d + dd;
                    let val = rd.data[cell_idx];
                    let w = val as f64;
                    weight_sum += w;
                    range_acc += rr as f64 * w;
                    doppler_acc += dd as f64 * w;
                    count += 1;
                    if val > peak {
                        peak = val;
                        peak_rc = (rr, dd);
                    }

                    let neighbors: [(isize, isize); 4] = [(-1, 0), (1, 0), (0, -1), (0, 1)];
                    for (dr, dd_off) in neighbors {
                        let nr = rr as isize + dr;
                        let nd = dd as isize + dd_off;
                        if nr < 0 || nd < 0 || nr >= r as isize || nd >= d as isize {
                            continue;
                        }
                        let n_idx = nr as usize * d + nd as usize;
                        if visited[n_idx] {
                            continue;
                        }
                        if rd.data[n_idx] > self.params.threshold {
                            visited[n_idx] = true;
                            queue.push_back((nr as usize, nd as usize));
                        } else {
                            visited[n_idx] = true;
                        }
                    }
                }

                if count >= self.params.min_cluster_size && weight_sum > 0.0 {
                    let centroid_r = (range_acc / weight_sum).round() as usize;
                    let centroid_d = (doppler_acc / weight_sum).round() as usize;
                    // Clamp centroid in case rounding drifts outside grid.
                    let cr = centroid_r.min(r - 1);
                    let cd = centroid_d.min(d - 1);
                    let _ = peak_rc; // peak position retained for future use
                    events.push(DetectionEvent::new(
                        cr,
                        Some(cd),
                        magnitude_to_db(peak as f64),
                        DetectionKind::Blob,
                    ));
                }
            }
        }
        events
    }

    fn kind(&self) -> DetectionKind {
        DetectionKind::Blob
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blob_detector_finds_single_3x3_blob_centroid() {
        let r = 16usize;
        let d = 16usize;
        let mut rd = RangeDoppler::zeros(r, d);
        // 3x3 blob centered at (7, 7) with values >> threshold.
        for rr in 6..=8 {
            for dd in 6..=8 {
                rd.set(rr, dd, 10.0);
            }
        }
        let det = BlobRdDetector::new(BlobRdParams::new(1.0, 1));
        let events = det.detect(&rd);
        assert_eq!(events.len(), 1, "expected one blob, got {events:?}");
        let e = &events[0];
        assert_eq!(e.range_bin, 7);
        assert_eq!(e.doppler_bin, Some(7));
        assert_eq!(e.kind, DetectionKind::Blob);
    }

    #[test]
    fn blob_detector_respects_min_cluster_size() {
        let r = 8usize;
        let d = 8usize;
        let mut rd = RangeDoppler::zeros(r, d);
        rd.set(3, 3, 5.0); // single isolated cell
        let det = BlobRdDetector::new(BlobRdParams::new(1.0, 4));
        let events = det.detect(&rd);
        assert!(events.is_empty(), "single cell should be filtered");
    }

    #[test]
    fn blob_detector_handles_two_separate_blobs() {
        let r = 16usize;
        let d = 16usize;
        let mut rd = RangeDoppler::zeros(r, d);
        for rr in 2..=3 {
            for dd in 2..=3 {
                rd.set(rr, dd, 8.0);
            }
        }
        for rr in 11..=12 {
            for dd in 11..=12 {
                rd.set(rr, dd, 8.0);
            }
        }
        let det = BlobRdDetector::new(BlobRdParams::new(1.0, 1));
        let events = det.detect(&rd);
        assert_eq!(events.len(), 2, "expected two blobs, got {events:?}");
    }
}
