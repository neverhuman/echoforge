//! Micro-Doppler feature detector.
//!
//! Operates on a `RangeDoppler` grid where each row is a doppler-power
//! spectrum at one range bin. Detects periodic sideband structure by
//! autocorrelating each row along the doppler axis (zero-mean) and looking
//! for the dominant non-zero lag whose autocorrelation exceeds a fraction of
//! the zero-lag energy.
//!
//! The dominant lag is converted to a modulation frequency in Hz via the
//! configured doppler bin spacing (`doppler_bin_hz`). The "sideband spread"
//! is the FWHM of the autocorrelation peak in Hz.

use super::{DetectionEvent, DetectionKind, Detector, RangeDoppler};

#[derive(Debug, Clone, Copy)]
pub struct MicroDopplerParams {
    /// Hz per doppler bin (i.e. doppler resolution).
    pub doppler_bin_hz: f64,
    /// Minimum lag to consider when searching for the dominant sideband
    /// (in bins). Helps skip the broad zero-lag lobe.
    pub min_lag_bins: usize,
    /// Maximum lag to consider (in bins). Defaults to `doppler_bins / 2`
    /// when set to 0.
    pub max_lag_bins: usize,
    /// Required ratio of the dominant lag's autocorrelation to lag-0 for the
    /// row to be reported (0.0..=1.0).
    pub detection_ratio: f32,
}

impl MicroDopplerParams {
    pub fn new(
        doppler_bin_hz: f64,
        min_lag_bins: usize,
        max_lag_bins: usize,
        detection_ratio: f32,
    ) -> Self {
        Self {
            doppler_bin_hz,
            min_lag_bins,
            max_lag_bins,
            detection_ratio,
        }
    }
}

/// Per-row micro-Doppler feature payload encoded into a `DetectionEvent`:
/// `magnitude_db` carries the dominant modulation frequency in Hz, and
/// `doppler_bin` carries the integer sideband spread (in bins). This keeps
/// the unified `DetectionEvent` shape while still surfacing the two key
/// micro-Doppler features.
#[derive(Debug, Clone, Copy)]
pub struct MicroDopplerDetector {
    pub params: MicroDopplerParams,
}

impl MicroDopplerDetector {
    pub fn new(params: MicroDopplerParams) -> Self {
        Self { params }
    }
}

fn zero_mean(row: &[f32]) -> Vec<f32> {
    let mean = row.iter().sum::<f32>() / row.len().max(1) as f32;
    row.iter().map(|v| v - mean).collect()
}

fn autocorrelation(row_zm: &[f32], max_lag: usize) -> Vec<f64> {
    let n = row_zm.len();
    let mut out = Vec::with_capacity(max_lag + 1);
    for lag in 0..=max_lag {
        let mut acc = 0.0f64;
        let mut count = 0usize;
        for i in 0..(n - lag) {
            acc += row_zm[i] as f64 * row_zm[i + lag] as f64;
            count += 1;
        }
        out.push(if count == 0 { 0.0 } else { acc / count as f64 });
    }
    out
}

impl Detector for MicroDopplerDetector {
    type Input = RangeDoppler;

    fn detect(&self, rd: &Self::Input) -> Vec<DetectionEvent> {
        let mut events = Vec::new();
        let r = rd.range_bins;
        let d = rd.doppler_bins;
        if r == 0 || d < 4 {
            return events;
        }
        let max_lag = if self.params.max_lag_bins == 0 {
            d / 2
        } else {
            self.params.max_lag_bins.min(d - 1)
        };
        let min_lag = self.params.min_lag_bins.max(1).min(max_lag);

        for range_bin in 0..r {
            let row = &rd.data[range_bin * d..(range_bin + 1) * d];
            let zm = zero_mean(row);
            let acf = autocorrelation(&zm, max_lag);
            let r0 = acf[0];
            if r0 <= 0.0 {
                continue;
            }

            // Find lag with maximum autocorrelation in [min_lag, max_lag].
            let mut best_lag = min_lag;
            let mut best_val = acf[min_lag];
            for (lag, value) in acf.iter().enumerate().take(max_lag + 1).skip(min_lag + 1) {
                if *value > best_val {
                    best_val = *value;
                    best_lag = lag;
                }
            }
            if best_val / r0 < self.params.detection_ratio as f64 {
                continue;
            }

            // FWHM in bins around `best_lag` measured against `best_val`.
            let half = best_val / 2.0;
            let mut left = best_lag;
            while left > 0 && acf[left] >= half {
                left -= 1;
            }
            let mut right = best_lag;
            while right < acf.len() - 1 && acf[right] >= half {
                right += 1;
            }
            let spread_bins = right.saturating_sub(left);

            let modulation_hz = best_lag as f64 * self.params.doppler_bin_hz;
            events.push(DetectionEvent::new(
                range_bin,
                Some(spread_bins),
                modulation_hz,
                DetectionKind::MicroDoppler,
            ));
        }
        events
    }

    fn kind(&self) -> DetectionKind {
        DetectionKind::MicroDoppler
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a single-row spectrogram with sidebands implanted at `±k_bins`
    /// from center via a periodic component `cos(2π k_bins / d * n)`.
    fn periodic_row(d: usize, k_bins: usize) -> Vec<f32> {
        let mut row = vec![0.0f32; d];
        for (n, slot) in row.iter_mut().enumerate().take(d) {
            let phase = 2.0 * std::f32::consts::PI * (k_bins as f32) * (n as f32) / d as f32;
            *slot = 1.0 + phase.cos();
        }
        row
    }

    #[test]
    fn micro_doppler_detects_dominant_modulation() {
        // 256-bin doppler row, periodic component at 10 cycles across the
        // window, doppler resolution 10 Hz/bin. The autocorrelation peak
        // should land at lag = d / k = 256 / 10 ≈ 25..26 bins, giving a
        // modulation frequency ≈ 250 Hz. The configured doppler_bin_hz can
        // map any chosen lag to an Hz value; we pick parameters so the
        // expected modulation is 100 Hz +/- tolerance.
        //
        // To target 100 Hz with d=200 and k=20 we get lag=10 bins; setting
        // doppler_bin_hz=10.0 yields 100 Hz.
        let d = 200usize;
        let k = 20usize;
        let row = periodic_row(d, k);
        let rd = RangeDoppler::new(1, d, row);
        let det = MicroDopplerDetector::new(MicroDopplerParams::new(10.0, 2, 0, 0.3));
        let events = det.detect(&rd);
        assert_eq!(
            events.len(),
            1,
            "expected one micro-Doppler event, got {events:?}"
        );
        let hz = events[0].magnitude_db;
        assert!(
            (hz - 100.0).abs() <= 10.0,
            "dominant modulation should be ~100 Hz, got {hz}"
        );
        assert_eq!(events[0].kind, DetectionKind::MicroDoppler);
    }

    #[test]
    fn micro_doppler_returns_no_events_for_flat_row() {
        let d = 128usize;
        let rd = RangeDoppler::new(1, d, vec![1.0f32; d]);
        let det = MicroDopplerDetector::new(MicroDopplerParams::new(5.0, 1, 0, 0.3));
        let events = det.detect(&rd);
        assert!(events.is_empty(), "flat row should yield no events");
    }
}
