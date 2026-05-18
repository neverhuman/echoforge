//! Tier 3 — CRUISE phase detector. Stable propulsion + flight.
//!
//! Public-proxy speeds per the Wave-A
//! `shahed-public-proxy-flight-envelope-v2` dossier
//! (`object-packs/public-proxy-v1/physics_dossier.md`):
//! piston cluster 40–60 m/s (max 55–60 per dossier); jet variant
//! (Shahed-238) 100–150 m/s. Blade-pass frequency 150–220 Hz for
//! B=2 × cruise RPM 4500–6500 / 60. Reject outside these clusters or
//! in the ambiguous gap (60, 100) m/s.
//!
//! Detection rule:
//!   1. Speed-cluster classify (piston / jet / ambiguous / out-of-class).
//!      If rejected, return `detected: false`.
//!   2. MTD bin selection: identify the dominant Doppler bin via
//!      `mti_mtd::mtd_chain` output (caller supplies the post-MTD power
//!      vector for the current range gate; the cruise detector takes the
//!      argmax bin).
//!   3. OS-CFAR threshold via `cfar_alpha::resolve_alpha` on the
//!      selected bin: cell-under-test vs noise estimate × alpha.
//!   4. Micro-Doppler confirmation (increases confidence): check for a
//!      blade-pass line in the [150, 220] Hz band ± 15% for piston, or
//!      compressor-like spectral signature for jet.
//!   5. M-of-N=5-of-7 over a trailing 7-CPI window, gated by a simple
//!      constant-velocity Kalman residual.
//!
//! Strict-open posture: per-tier Pd/Pfa published here are *public-proxy
//! expected* and do NOT claim platform-specific signature truth.

use crate::detectors::cfar_alpha::{resolve_alpha, CfarVariant, NoiseDistribution};

use super::kinematic_gate::{
    cruise_kinematic_gate_jet, cruise_kinematic_gate_piston, KinematicGate, KinematicObservation,
};
use super::speed_classifier::{PropulsionClass, SpeedClassifier};

/// Outcome of a single-CPI Tier 3 evaluation.
#[derive(Debug, Clone, PartialEq)]
pub struct CruiseDecision {
    pub detected: bool,
    pub propulsion_class: PropulsionClass,
    pub mof_n_ratio: f32,
    pub kalman_consistency: f32,
    pub cfar_passed: bool,
    pub micro_doppler_confirmed: bool,
    pub dominant_doppler_bin: Option<usize>,
    pub note: &'static str,
}

impl CruiseDecision {
    pub fn empty() -> Self {
        Self {
            detected: false,
            propulsion_class: PropulsionClass::None,
            mof_n_ratio: 0.0,
            kalman_consistency: 0.0,
            cfar_passed: false,
            micro_doppler_confirmed: false,
            dominant_doppler_bin: None,
            note: "",
        }
    }
}

/// Configuration for the Tier 3 cruise detector. The OS-CFAR parameters
/// (training cells, guard cells, Pfa, quantile rank) are passed through
/// to `cfar_alpha::resolve_alpha`.
#[derive(Debug, Clone, Copy)]
pub struct CruiseTierConfig {
    /// OS-CFAR training cells per side.
    pub cfar_training_cells: usize,
    /// OS-CFAR guard cells per side.
    pub cfar_guard_cells: usize,
    /// Desired probability of false alarm.
    pub cfar_pfa: f32,
    /// OS-CFAR rank index k (1-indexed within the combined window).
    pub cfar_rank: usize,
    /// Kalman speed-gate width (m/s).
    pub kalman_speed_gate_mps: f64,
    /// M-of-N denominator (window length).
    pub mof_n_window: usize,
    /// M-of-N threshold.
    pub mof_n_threshold: usize,
}

impl Default for CruiseTierConfig {
    fn default() -> Self {
        // Wave-1 OS-CFAR defaults: 16 train cells/side, 4 guard cells,
        // Pfa=1e-4, rank=12 (75th-percentile rank for total N=32).
        Self {
            cfar_training_cells: 16,
            cfar_guard_cells: 4,
            cfar_pfa: 1.0e-4,
            cfar_rank: 24,
            kalman_speed_gate_mps: 5.0,
            mof_n_window: 7,
            mof_n_threshold: 5,
        }
    }
}

/// Tier 3 cruise detector. Holds both the piston and jet gates and the
/// speed classifier; cruise CPIs run against whichever gate matches the
/// classifier's verdict.
#[derive(Debug, Clone, Copy)]
pub struct CruiseTierDetector {
    pub config: CruiseTierConfig,
    pub piston_gate: KinematicGate,
    pub jet_gate: KinematicGate,
    pub classifier: SpeedClassifier,
}

impl CruiseTierDetector {
    pub fn new(config: CruiseTierConfig) -> Self {
        Self {
            config,
            piston_gate: cruise_kinematic_gate_piston(),
            jet_gate: cruise_kinematic_gate_jet(),
            classifier: SpeedClassifier::shahed_class_default(),
        }
    }

    pub fn with_default() -> Self {
        Self::new(CruiseTierConfig::default())
    }

    /// Resolve the OS-CFAR alpha for the configured parameters. Uses
    /// the Gaussian/Rayleigh closed form (Skolnik §7.7.2, Rohling 1983);
    /// downstream Lane G_b ClutterRegime plumbing can swap in K /
    /// Weibull / log-normal as it lands.
    pub fn cfar_alpha(&self) -> f32 {
        let total = self.config.cfar_training_cells.saturating_mul(2);
        resolve_alpha(
            CfarVariant::OrderedStatistic {
                rank: self.config.cfar_rank.min(total.max(1)),
            },
            NoiseDistribution::Gaussian,
            total,
            self.config.cfar_pfa,
        )
    }

    /// Compute the OS-CFAR threshold (linear) given a noise estimate
    /// equal to the configured rank of the sorted training cells.
    pub fn cfar_threshold(&self, noise_estimate: f32) -> f32 {
        self.cfar_alpha() * noise_estimate
    }

    /// Evaluate one CPI worth of kinematic observation plus an
    /// optional Doppler power spectrum (post-MTD, single range gate).
    ///
    /// Arguments:
    ///   * `obs` — kinematic sliding window.
    ///   * `mtd_power_spectrum` — optional post-MTD power vector for the
    ///     current range gate. When supplied the detector picks the
    ///     argmax bin, fits OS-CFAR around it, and gates on the
    ///     threshold.
    ///   * `doppler_bin_hz` — Hz/bin width of `mtd_power_spectrum`.
    ///     When `Some`, the detector inspects the [150–220] Hz blade-pass
    ///     band ± 15% for the micro-Doppler confirmation step.
    pub fn evaluate(
        &self,
        obs: &KinematicObservation,
        mtd_power_spectrum: Option<&[f32]>,
        doppler_bin_hz: Option<f64>,
    ) -> CruiseDecision {
        let mut out = CruiseDecision::empty();
        let speed = match obs.current_radial_speed_mps() {
            Some(v) => v.abs(),
            None => {
                out.note = "empty observation";
                return out;
            }
        };

        // (1) Speed-cluster classify.
        let class = self.classifier.classify(speed);
        out.propulsion_class = class;
        let gate = match class {
            PropulsionClass::Piston => self.piston_gate,
            PropulsionClass::Jet => self.jet_gate,
            _ => {
                out.note = match class {
                    PropulsionClass::Ambiguous => "speed in dossier ambiguity gap",
                    PropulsionClass::BirdLike => "speed below piston cluster",
                    PropulsionClass::AircraftLike => "speed above jet cluster",
                    _ => "speed unclassified",
                };
                return out;
            }
        };

        if !gate.accepts(obs) {
            out.note = "cruise gate not satisfied";
            return out;
        }

        // (2) + (3) MTD bin selection + OS-CFAR.
        let mut cfar_passed = true;
        let mut dominant_bin: Option<usize> = None;
        if let Some(spec) = mtd_power_spectrum {
            if !spec.is_empty() {
                let (peak_idx, &peak_val) = spec
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .unwrap();
                dominant_bin = Some(peak_idx);
                let noise_est = noise_estimate_os_cfar(
                    spec,
                    peak_idx,
                    self.config.cfar_training_cells,
                    self.config.cfar_guard_cells,
                    self.config.cfar_rank,
                );
                let threshold = self.cfar_threshold(noise_est);
                cfar_passed = peak_val >= threshold;
                if !cfar_passed {
                    out.cfar_passed = false;
                    out.dominant_doppler_bin = dominant_bin;
                    out.note = "OS-CFAR threshold not exceeded";
                    return out;
                }
            }
        }
        out.cfar_passed = cfar_passed;
        out.dominant_doppler_bin = dominant_bin;

        // (4) Micro-Doppler confirmation (optional).
        if let (Some(spec), Some(bin_hz)) = (mtd_power_spectrum, doppler_bin_hz) {
            out.micro_doppler_confirmed = check_blade_pass_line(spec, bin_hz, class);
        }

        // (5) M-of-N=5-of-7 over Kalman residuals.
        let win = self.config.mof_n_window;
        let trailing = obs.trailing(win + 1);
        if trailing.len() < 2 {
            out.note = "insufficient history for M-of-N";
            return out;
        }
        let mut matches = 0usize;
        let mut considered = 0usize;
        let mut consistency_sum = 0.0f64;
        for pair in trailing.windows(2) {
            let residual = (pair[1].radial_speed_mps - pair[0].radial_speed_mps).abs();
            let normalized = (residual / self.config.kalman_speed_gate_mps).min(1.0);
            consistency_sum += 1.0 - normalized;
            considered += 1;
            if residual <= self.config.kalman_speed_gate_mps {
                matches += 1;
            }
        }
        out.mof_n_ratio = if considered == 0 {
            0.0
        } else {
            matches as f32 / considered as f32
        };
        out.kalman_consistency = if considered == 0 {
            0.0
        } else {
            (consistency_sum / considered as f64) as f32
        };
        out.detected = matches >= self.config.mof_n_threshold;
        out.note = if out.detected {
            "cruise detected"
        } else {
            "M-of-N below threshold"
        };
        out
    }
}

/// Standalone OS-CFAR noise estimate: take training cells on each side of
/// the cell-under-test (excluding guard cells), sort ascending, pick the
/// 1-indexed rank-th element. Returns 0.0 when the window cannot be filled.
fn noise_estimate_os_cfar(
    spec: &[f32],
    cut_bin: usize,
    training_cells: usize,
    guard_cells: usize,
    rank_1_indexed: usize,
) -> f32 {
    let n = spec.len();
    let window = training_cells + guard_cells;
    if cut_bin < window || cut_bin + window >= n {
        // Insufficient training window on either side; collect what is
        // available and pad with the value at the closest edge so the
        // sort still has a sensible noise estimate (matches OS-CFAR's
        // edge-handling convention).
        let mut samples: Vec<f32> = Vec::with_capacity(2 * training_cells);
        for i in 0..training_cells {
            let lo = cut_bin.saturating_sub(guard_cells + 1 + i);
            samples.push(spec[lo]);
        }
        for i in 0..training_cells {
            let hi = (cut_bin + guard_cells + 1 + i).min(n - 1);
            samples.push(spec[hi]);
        }
        samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let total = samples.len();
        if total == 0 {
            return 0.0;
        }
        let idx = (rank_1_indexed.saturating_sub(1)).min(total - 1);
        return samples[idx];
    }
    let mut samples: Vec<f32> = Vec::with_capacity(2 * training_cells);
    for i in 0..training_cells {
        samples.push(spec[cut_bin - guard_cells - 1 - i]);
        samples.push(spec[cut_bin + guard_cells + 1 + i]);
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let total = samples.len();
    let idx = (rank_1_indexed.saturating_sub(1)).min(total - 1);
    samples[idx]
}

/// Heuristic micro-Doppler check: for piston targets look for any bin
/// in the blade-pass band [150, 220] Hz ± 15% whose power exceeds twice
/// the spectrum median (a cheap signal-vs-floor proxy). For jet targets
/// the dossier does not pin a specific blade-pass band, so the function
/// looks for an above-floor line *anywhere outside the body-Doppler
/// region*, which is what jet compressor signatures look like in the
/// public literature.
fn check_blade_pass_line(spec: &[f32], doppler_bin_hz: f64, class: PropulsionClass) -> bool {
    if spec.is_empty() || doppler_bin_hz <= 0.0 {
        return false;
    }
    let n = spec.len();
    // Compute median as a stable floor estimator (the simple mean is
    // contaminated by any bright body-Doppler peak).
    let mut sorted: Vec<f32> = spec.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median = sorted[n / 2].max(1e-12);

    let (lo_hz, hi_hz) = match class {
        PropulsionClass::Piston => {
            // 150–220 Hz ± 15% => [127.5, 253.0] Hz.
            (150.0 * 0.85, 220.0 * 1.15)
        }
        PropulsionClass::Jet => {
            // Jet compressor lines typically sit at the high-end of the
            // useful spectrum; the dossier does not bound them tightly,
            // so use the upper half-band of the available spectrum as a
            // proxy for "compressor-like high-frequency content".
            let nyquist_hz = doppler_bin_hz * (n as f64) / 2.0;
            (nyquist_hz * 0.5, nyquist_hz)
        }
        _ => return false,
    };
    let lo_bin = (lo_hz / doppler_bin_hz).max(0.0) as usize;
    let hi_bin = ((hi_hz / doppler_bin_hz) as usize).min(n - 1);
    if lo_bin >= hi_bin {
        return false;
    }
    spec[lo_bin..=hi_bin].iter().any(|&v| v >= 2.0 * median)
}

#[cfg(test)]
mod tests {
    use super::super::kinematic_gate::KinematicSample;
    use super::*;

    fn cruise_window(samples: Vec<(f64, f64, f64)>) -> KinematicObservation {
        KinematicObservation::new(
            samples
                .into_iter()
                .map(|(t, v, h)| KinematicSample::new(t, v, h))
                .collect(),
            12_000.0,
            20.0,
        )
    }

    fn steady_cruise_window(speed: f64, alt: f64, n: usize) -> KinematicObservation {
        let samples: Vec<(f64, f64, f64)> = (0..n)
            .map(|k| (k as f64, speed, alt))
            .collect();
        cruise_window(samples)
    }

    #[test]
    fn cruise_detector_accepts_piston_50mps_steady() {
        let detector = CruiseTierDetector::with_default();
        let obs = steady_cruise_window(50.0, 800.0, 8);
        let dec = detector.evaluate(&obs, None, None);
        assert_eq!(dec.propulsion_class, PropulsionClass::Piston);
        assert!(dec.detected, "steady piston cruise must detect; {}", dec.note);
    }

    #[test]
    fn cruise_detector_accepts_jet_120mps_steady() {
        let detector = CruiseTierDetector::with_default();
        let obs = steady_cruise_window(120.0, 800.0, 8);
        let dec = detector.evaluate(&obs, None, None);
        assert_eq!(dec.propulsion_class, PropulsionClass::Jet);
        assert!(dec.detected, "steady jet cruise must detect; {}", dec.note);
    }

    #[test]
    fn cruise_detector_rejects_ambiguous_75mps() {
        let detector = CruiseTierDetector::with_default();
        let obs = steady_cruise_window(75.0, 800.0, 8);
        let dec = detector.evaluate(&obs, None, None);
        assert_eq!(dec.propulsion_class, PropulsionClass::Ambiguous);
        assert!(!dec.detected, "ambiguous cluster must NOT detect");
    }

    #[test]
    fn cruise_detector_rejects_bird_15mps() {
        let detector = CruiseTierDetector::with_default();
        let obs = steady_cruise_window(15.0, 800.0, 8);
        let dec = detector.evaluate(&obs, None, None);
        assert_eq!(dec.propulsion_class, PropulsionClass::BirdLike);
        assert!(!dec.detected);
    }

    #[test]
    fn cruise_detector_os_cfar_threshold_check_blocks_noise_floor() {
        // Provide a flat-noise spectrum; OS-CFAR threshold must reject
        // (peak ~= noise estimate × ~1 << alpha).
        let detector = CruiseTierDetector::with_default();
        let obs = steady_cruise_window(50.0, 800.0, 8);
        let spec = vec![1.0f32; 128];
        let dec = detector.evaluate(&obs, Some(&spec), Some(5.0));
        assert!(!dec.detected, "flat noise must not pass OS-CFAR");
        assert!(!dec.cfar_passed);
    }

    #[test]
    fn cruise_detector_os_cfar_threshold_passes_strong_peak() {
        // Spike one bin to 1000 × the floor; CFAR must accept.
        let detector = CruiseTierDetector::with_default();
        let obs = steady_cruise_window(50.0, 800.0, 8);
        let mut spec = vec![1.0f32; 128];
        spec[64] = 1000.0;
        let dec = detector.evaluate(&obs, Some(&spec), Some(5.0));
        assert!(dec.cfar_passed, "strong peak must clear OS-CFAR");
        assert_eq!(dec.dominant_doppler_bin, Some(64));
    }

    #[test]
    fn cruise_detector_blade_pass_micro_doppler_piston() {
        let detector = CruiseTierDetector::with_default();
        let obs = steady_cruise_window(50.0, 800.0, 8);
        // 256 bins × 1 Hz = 256 Hz Nyquist. Spike bin 180 (180 Hz)
        // which is squarely in [127.5, 253] Hz blade-pass window.
        let mut spec = vec![1.0f32; 256];
        spec[64] = 1000.0; // body Doppler peak (CFAR target)
        spec[180] = 50.0; // blade-pass line
        let dec = detector.evaluate(&obs, Some(&spec), Some(1.0));
        assert!(
            dec.micro_doppler_confirmed,
            "blade-pass at 180 Hz must be confirmed for piston cluster"
        );
    }
}
