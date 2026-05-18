//! Tier 2 — CLIMB-OUT phase detector. Post-boost piston engine takeover.
//!
//! Public-proxy timing per the Wave-A
//! `shahed-public-proxy-flight-envelope-v2` dossier
//! (`object-packs/public-proxy-v1/physics_dossier.md`): 3–30 s after
//! launch; airframe transitions from ~30 m/s to cruise (50–55 m/s) under
//! low piston thrust. Climb rate 2–4 m/s typical. First robust
//! opportunity for ground radar acquisition.
//!
//! Detection rule:
//!   1. Tier 2 kinematic gate (speed 25–60 m/s, accel 0.1–2.0 m/s²,
//!      altitude 30–1500 m AGL).
//!   2. MTI 2-pulse Doppler-notch check (reject if the target's body
//!      Doppler at the carrier falls inside the typical 2-pulse notch
//!      of ±30 Hz around DC).
//!   3. M-of-N=3-of-5 over a trailing 5-CPI window, gated by a simple
//!      constant-velocity Kalman residual on radial range and Doppler.
//!
//! The Kalman residual is a 1-D constant-velocity model:
//! `predicted_speed_k+1 = speed_k`, residual = |observed - predicted|.
//! The residual is normalized by a configurable speed-gate width
//! (default 5 m/s, which is generous against the dossier's 2–4 m/s
//! typical climb-rate scatter at S-band Doppler resolution).
//!
//! Strict-open posture: per-tier Pd/Pfa published by this detector are
//! *public-proxy expected* and do NOT claim platform-specific signature
//! truth.

use crate::propagation::SPEED_OF_LIGHT_M_PER_S;

use super::kinematic_gate::{climb_kinematic_gate, KinematicGate, KinematicObservation};

#[derive(Debug, Clone, PartialEq)]
pub struct ClimbDecision {
    pub detected: bool,
    pub mof_n_ratio: f32,
    pub mti_notch_rejected: bool,
    pub kalman_consistency: f32,
    pub note: &'static str,
}

impl ClimbDecision {
    pub fn empty() -> Self {
        Self {
            detected: false,
            mof_n_ratio: 0.0,
            mti_notch_rejected: false,
            kalman_consistency: 0.0,
            note: "",
        }
    }
}

/// Configuration for the Tier 2 climb-out detector.
#[derive(Debug, Clone, Copy)]
pub struct ClimbTierConfig {
    /// Carrier frequency (Hz) used to convert radial velocity to body
    /// Doppler for the MTI-notch check.
    pub carrier_freq_hz: f64,
    /// Two-sided MTI notch half-width (Hz). Typical 2-pulse MTI
    /// canceller has a notch of ~±30 Hz at S-band PRF; targets with
    /// |f_d| below this are rejected as zero-Doppler clutter look-alikes.
    pub mti_notch_half_width_hz: f64,
    /// Kalman speed-gate width (m/s) used to score residuals.
    pub kalman_speed_gate_mps: f64,
    /// M-of-N denominator.
    pub mof_n_window: usize,
    /// M-of-N threshold.
    pub mof_n_threshold: usize,
}

impl Default for ClimbTierConfig {
    fn default() -> Self {
        // S-band 3 GHz carrier, typical 30 Hz MTI notch, 5 m/s
        // Kalman gate. M-of-N=3-of-5.
        Self {
            carrier_freq_hz: 3.0e9,
            mti_notch_half_width_hz: 30.0,
            kalman_speed_gate_mps: 5.0,
            mof_n_window: 5,
            mof_n_threshold: 3,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ClimbOutTierDetector {
    pub config: ClimbTierConfig,
    pub gate: KinematicGate,
}

impl ClimbOutTierDetector {
    pub fn new(config: ClimbTierConfig) -> Self {
        Self {
            config,
            gate: climb_kinematic_gate(),
        }
    }

    pub fn with_default() -> Self {
        Self::new(ClimbTierConfig::default())
    }

    /// Convert a radial speed (m/s) into body Doppler at the configured
    /// carrier: `f_d = 2 * v_r * f_c / c`. Returns the unsigned
    /// magnitude so a closing/receding sign does not flip the notch test.
    fn body_doppler_hz(&self, radial_speed_mps: f64) -> f64 {
        (2.0 * radial_speed_mps.abs() * self.config.carrier_freq_hz / SPEED_OF_LIGHT_M_PER_S).abs()
    }

    /// Evaluate one CPI's worth of kinematic observation.
    pub fn evaluate(&self, obs: &KinematicObservation) -> ClimbDecision {
        let mut out = ClimbDecision::empty();
        let speed = match obs.current_radial_speed_mps() {
            Some(v) => v.abs(),
            None => {
                out.note = "empty observation";
                return out;
            }
        };

        // (1) Climb-out kinematic envelope.
        if !self.gate.accepts(obs) {
            out.note = "climb gate not satisfied";
            return out;
        }

        // (2) MTI 2-pulse notch check.
        let f_d = self.body_doppler_hz(speed);
        if f_d < self.config.mti_notch_half_width_hz {
            out.mti_notch_rejected = true;
            out.note = "target in 2-pulse MTI notch";
            return out;
        }

        // (3) M-of-N over a trailing 5-CPI window with constant-velocity
        // Kalman residual scoring. Each consecutive pair (s_k, s_{k+1})
        // produces a residual |s_{k+1} - s_k|; if that residual is
        // within the gate, the CPI counts as a match. The average
        // (1 - residual/gate) over considered pairs is the consistency
        // score.
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
            "climb-out detected"
        } else {
            "M-of-N below threshold"
        };
        out
    }
}

#[cfg(test)]
mod tests {
    use super::super::kinematic_gate::KinematicSample;
    use super::*;

    fn climb_window(samples: Vec<(f64, f64, f64)>) -> KinematicObservation {
        KinematicObservation::new(
            samples
                .into_iter()
                .map(|(t, v, h)| KinematicSample::new(t, v, h))
                .collect(),
            8_000.0,
            20.0,
        )
    }

    #[test]
    fn climb_detector_accepts_canonical_3_of_5() {
        // 6 samples → 5 deltas, each at gentle 0.5–1 m/s² accel and
        // 250 m altitude. All 5 deltas should fall within the 5 m/s
        // Kalman gate, satisfying 3-of-5.
        let detector = ClimbOutTierDetector::with_default();
        let obs = climb_window(vec![
            (0.0, 30.0, 250.0),
            (1.0, 31.0, 252.0),
            (2.0, 32.0, 254.0),
            (3.0, 33.0, 256.0),
            (4.0, 34.0, 258.0),
            (5.0, 35.0, 260.0),
        ]);
        let dec = detector.evaluate(&obs);
        assert!(dec.detected, "canonical climb-out must detect; {}", dec.note);
        assert!(!dec.mti_notch_rejected, "30+ m/s must clear MTI notch");
        assert!(dec.kalman_consistency > 0.5, "consistency = {}", dec.kalman_consistency);
    }

    #[test]
    fn climb_detector_rejects_in_mti_notch() {
        // Build an observation whose current speed maps to f_d < 30 Hz.
        // At 3 GHz, f_d = 2*v/c*f → v < 30 * c / (2 * 3e9) ≈ 1.5 m/s.
        // But our climb gate requires speed >= 25 m/s, so we must
        // construct a custom detector with a much wider MTI notch
        // (covering up to v=30 m/s → f_d=600 Hz at 3 GHz).
        let config = ClimbTierConfig {
            mti_notch_half_width_hz: 800.0,
            ..ClimbTierConfig::default()
        };
        let detector = ClimbOutTierDetector::new(config);
        let obs = climb_window(vec![
            (0.0, 29.0, 250.0),
            (1.0, 30.0, 252.0),
        ]);
        let dec = detector.evaluate(&obs);
        assert!(dec.mti_notch_rejected, "target inside notch must be rejected");
    }

    #[test]
    fn climb_detector_rejects_outside_gate() {
        // Speed 80 m/s is in the cruise-gap; climb gate must reject.
        let detector = ClimbOutTierDetector::with_default();
        let obs = climb_window(vec![
            (0.0, 79.0, 250.0),
            (1.0, 80.0, 250.0),
        ]);
        let dec = detector.evaluate(&obs);
        assert!(!dec.detected, "climb gate must reject 80 m/s");
    }
}
