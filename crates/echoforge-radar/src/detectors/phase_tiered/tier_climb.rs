//! Tier 2 — CLIMB-OUT phase detector. Post-boost piston engine takeover.
//!
//! Public-proxy timing per the Wave-A
//! `shahed-public-proxy-flight-envelope-v2` dossier
//! (`object-packs/public-proxy-v1/physics_dossier.md`): 3–30 s after
//! launch; airframe transitions from ~30 m/s to cruise (50–55 m/s) under
//! low piston thrust. Climb rate 2–4 m/s typical. First robust
//! opportunity for ground radar acquisition.
//!
//! Detection rule (radial-velocity branch):
//!   1. Tier 2 kinematic gate (speed 25–60 m/s, accel 0.1–2.0 m/s²,
//!      altitude 30–1500 m AGL).
//!   2. MTI 2-pulse Doppler-notch check (reject if the target's body
//!      Doppler at the carrier falls inside the typical 2-pulse notch
//!      of ±30 Hz around DC).
//!   3. M-of-N=3-of-5 over a trailing 5-CPI window, gated by a simple
//!      constant-velocity Kalman residual on radial range and Doppler.
//!
//! Detection rule (cross-flight branch — **Wave 4.5 H4**):
//!   When |v_radial| < `MTI_NOTCH_BODY_DOPPLER_HZ` (the body Doppler
//!   cutoff that maps the radial-velocity component into the 2-pulse MTI
//!   notch), the target is treated as flying perpendicular to the radar
//!   LOS (cross-flight geometry). The Skolnik §3.7 "MTI blind speeds"
//!   problem causes such targets to be silently rejected by the standard
//!   MTI gate even though they are otherwise observable. The compensating
//!   evidence pattern (per Skolnik §3.7 and Richards "Fundamentals of
//!   Radar Signal Processing" 2014 §5.4) is:
//!     a. Skip the MTI gate (the target is in the notch *by geometry*,
//!        not because it is clutter).
//!     b. Tighten the Kalman residual: require **4-of-5** within a
//!        tighter speed gate (default 3 m/s, vs 5 m/s in the radial
//!        branch). The cross-flight branch loses one degree of evidence
//!        (MTI gate) and must replace it with stricter kinematic
//!        consistency.
//!     c. Require **micro-Doppler confirmation**: a propeller blade-pass
//!        line in the [150, 220] Hz band ± 15% (matching the Tier 3
//!        piston cruise blade-pass spec) must be present in the supplied
//!        slow-time amplitude spectrum. Without micro-Doppler the
//!        cross-flight branch returns `detected: false` and notes the
//!        missing compensating evidence — never a silent miss.
//!
//! The Kalman residual is a 1-D constant-velocity model:
//! `predicted_speed_k+1 = speed_k`, residual = |observed - predicted|.
//! The residual is normalized by a configurable speed-gate width
//! (default 5 m/s radial / 3 m/s cross-flight, both generous against the
//! dossier's 2–4 m/s typical climb-rate scatter at S-band Doppler
//! resolution).
//!
//! Strict-open posture: per-tier Pd/Pfa published by this detector are
//! *public-proxy expected* and do NOT claim platform-specific signature
//! truth.

use crate::propagation::SPEED_OF_LIGHT_M_PER_S;

use super::kinematic_gate::{climb_kinematic_gate, KinematicGate, KinematicObservation};

/// Body-Doppler threshold below which the target is considered
/// "cross-flight" (perpendicular to radar LOS). At S-band 2.9 GHz, the
/// standard 2-pulse MTI canceller has a notch of ~30 Hz (Skolnik §3.7
/// "MTI blind speeds"). Cross-flight targets fall in the notch and need
/// compensating evidence (micro-Doppler + tighter Kalman) per Richards
/// "Fundamentals of Radar Signal Processing" 2014 §5.4. Expressed here
/// as a body-Doppler magnitude in Hz so it is independent of the carrier
/// frequency (the `ClimbOutTierDetector::body_doppler_hz` helper maps
/// `v_radial` → `f_d` for the configured carrier).
pub const MTI_NOTCH_BODY_DOPPLER_HZ: f64 = 30.0;

#[derive(Debug, Clone, PartialEq)]
pub struct ClimbDecision {
    pub detected: bool,
    pub mof_n_ratio: f32,
    pub mti_notch_rejected: bool,
    pub kalman_consistency: f32,
    /// **Wave 4.5 H4** — `true` when the cross-flight branch was taken
    /// (target |v_radial| body Doppler below `MTI_NOTCH_BODY_DOPPLER_HZ`,
    /// so the MTI gate was skipped and stricter Kalman + micro-Doppler
    /// confirmation were required). Published so reviewers see exactly
    /// which detection path produced the decision.
    pub cross_flight: bool,
    /// **Wave 4.5 H4** — `true` when the cross-flight branch confirmed a
    /// propeller blade-pass line in [150, 220] Hz ± 15% in the supplied
    /// slow-time amplitude spectrum. Always `false` for the radial-velocity
    /// branch (which does not require this evidence).
    pub micro_doppler_confirmed: bool,
    pub note: &'static str,
}

impl ClimbDecision {
    pub fn empty() -> Self {
        Self {
            detected: false,
            mof_n_ratio: 0.0,
            mti_notch_rejected: false,
            kalman_consistency: 0.0,
            cross_flight: false,
            micro_doppler_confirmed: false,
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
    /// Kalman speed-gate width (m/s) used to score residuals in the
    /// radial-velocity branch (standard 3-of-5 M-of-N).
    pub kalman_speed_gate_mps: f64,
    /// M-of-N denominator.
    pub mof_n_window: usize,
    /// M-of-N threshold (radial-velocity branch).
    pub mof_n_threshold: usize,
    /// **Wave 4.5 H4** — tighter Kalman speed-gate width (m/s) used in
    /// the cross-flight branch. The branch trades the MTI gate for
    /// stricter kinematic consistency, so the gate is narrower than the
    /// radial-velocity branch by default (3 m/s vs 5 m/s).
    pub cross_flight_kalman_speed_gate_mps: f64,
    /// **Wave 4.5 H4** — M-of-N threshold (cross-flight branch). Default
    /// 4-of-5 (vs 3-of-5 radial) to compensate for the skipped MTI gate.
    pub cross_flight_mof_n_threshold: usize,
}

impl Default for ClimbTierConfig {
    fn default() -> Self {
        // S-band 3 GHz carrier, typical 30 Hz MTI notch, 5 m/s
        // Kalman gate. M-of-N=3-of-5. Cross-flight branch: 3 m/s
        // tighter Kalman gate, 4-of-5 M-of-N.
        Self {
            carrier_freq_hz: 3.0e9,
            mti_notch_half_width_hz: 30.0,
            kalman_speed_gate_mps: 5.0,
            mof_n_window: 5,
            mof_n_threshold: 3,
            cross_flight_kalman_speed_gate_mps: 3.0,
            cross_flight_mof_n_threshold: 4,
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

    /// Evaluate one CPI's worth of kinematic observation (radial-velocity
    /// branch; no slow-time amplitude spectrum supplied so the
    /// cross-flight micro-Doppler test will always reject when taken).
    /// Equivalent to `evaluate_with_spectrum(obs, None, None)`.
    pub fn evaluate(&self, obs: &KinematicObservation) -> ClimbDecision {
        self.evaluate_with_spectrum(obs, None, None)
    }

    /// **Wave 4.5 H4** — evaluate one CPI with an optional slow-time
    /// amplitude spectrum and per-bin Hz width. When the target's body
    /// Doppler falls below `MTI_NOTCH_BODY_DOPPLER_HZ` the cross-flight
    /// branch is taken: the MTI gate is skipped and stricter Kalman +
    /// blade-pass micro-Doppler confirmation are required (Skolnik §3.7).
    ///
    /// Arguments:
    ///   * `obs` — kinematic sliding window.
    ///   * `slow_time_spectrum` — optional slow-time amplitude spectrum
    ///     for the current range gate (consumed by the cross-flight
    ///     branch to locate a propeller blade-pass line). When `None`
    ///     the cross-flight branch cannot confirm and will reject.
    ///   * `spectrum_bin_hz` — Hz/bin width of `slow_time_spectrum`.
    pub fn evaluate_with_spectrum(
        &self,
        obs: &KinematicObservation,
        slow_time_spectrum: Option<&[f32]>,
        spectrum_bin_hz: Option<f64>,
    ) -> ClimbDecision {
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

        // (2) Branch on body Doppler vs the cross-flight cutoff.
        let f_d = self.body_doppler_hz(speed);
        if f_d < MTI_NOTCH_BODY_DOPPLER_HZ {
            // ---- Cross-flight branch (Skolnik §3.7 MTI blind speed). ----
            // MTI gate skipped: target is in the notch by geometry, not because it
            // is clutter. Compensating evidence: stricter Kalman M-of-N + confirmed
            // propeller blade-pass line in the slow-time spectrum.
            out.cross_flight = true;
            let Some(kalman_ok) = self.apply_mofn(
                &mut out,
                obs,
                self.config.cross_flight_kalman_speed_gate_mps,
                self.config.cross_flight_mof_n_threshold,
                "cross-flight: insufficient history for M-of-N",
            ) else {
                return out;
            };
            out.micro_doppler_confirmed = match (slow_time_spectrum, spectrum_bin_hz) {
                (Some(spec), Some(bin_hz)) if bin_hz > 0.0 => {
                    check_blade_pass_line_piston(spec, bin_hz)
                }
                _ => false,
            };
            out.detected = kalman_ok && out.micro_doppler_confirmed;
            out.note = if out.detected {
                "cross-flight: detected (Kalman + micro-Doppler)"
            } else if !kalman_ok && !out.micro_doppler_confirmed {
                "cross-flight: Kalman M-of-N below threshold and no micro-Doppler line"
            } else if !kalman_ok {
                "cross-flight: Kalman M-of-N below threshold"
            } else {
                "cross-flight: no micro-Doppler blade-pass line"
            };
            return out;
        }

        // ---- Radial-velocity branch (existing behaviour). ----
        // (2a) MTI 2-pulse notch check.
        if f_d < self.config.mti_notch_half_width_hz {
            out.mti_notch_rejected = true;
            out.note = "target in 2-pulse MTI notch";
            return out;
        }

        // (3) M-of-N over a trailing 5-CPI window.
        let Some(radial_ok) = self.apply_mofn(
            &mut out,
            obs,
            self.config.kalman_speed_gate_mps,
            self.config.mof_n_threshold,
            "insufficient history for M-of-N",
        ) else {
            return out;
        };
        out.detected = radial_ok;
        out.note = if out.detected {
            "climb-out detected"
        } else {
            "M-of-N below threshold"
        };
        out
    }

    /// Apply the 1-D constant-velocity Kalman M-of-N gate and write
    /// `out.mof_n_ratio` / `out.kalman_consistency`. Returns `None` (with
    /// `out.note` set) when the observation window has fewer than 2 samples;
    /// returns `Some(ok)` indicating whether `matches >= threshold`.
    fn apply_mofn(
        &self,
        out: &mut ClimbDecision,
        obs: &KinematicObservation,
        gate_mps: f64,
        threshold: usize,
        short_window_note: &'static str,
    ) -> Option<bool> {
        let trailing = obs.trailing(self.config.mof_n_window + 1);
        if trailing.len() < 2 {
            out.note = short_window_note;
            return None;
        }
        let mut matches = 0usize;
        let mut considered = 0usize;
        let mut score_sum = 0.0f64;
        for pair in trailing.windows(2) {
            let residual = (pair[1].radial_speed_mps - pair[0].radial_speed_mps).abs();
            let normalized = (residual / gate_mps).min(1.0);
            score_sum += 1.0 - normalized;
            considered += 1;
            if residual <= gate_mps {
                matches += 1;
            }
        }
        out.mof_n_ratio = if considered == 0 { 0.0 } else { matches as f32 / considered as f32 };
        out.kalman_consistency = if considered == 0 { 0.0 } else { (score_sum / considered as f64) as f32 };
        Some(matches >= threshold)
    }

}

/// Cross-flight blade-pass confirmation: look for any bin in the
/// [150, 220] Hz ± 15% band whose power exceeds twice the spectrum
/// median (a stable signal-vs-floor proxy; the median is robust to a
/// bright body-Doppler peak that would contaminate the mean). Mirrors
/// `tier_cruise::check_blade_pass_line` for the piston class.
fn check_blade_pass_line_piston(spec: &[f32], doppler_bin_hz: f64) -> bool {
    if spec.is_empty() || doppler_bin_hz <= 0.0 {
        return false;
    }
    let n = spec.len();
    let mut sorted: Vec<f32> = spec.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median = sorted[n / 2].max(1e-12);

    let lo_hz = 150.0 * 0.85; // 127.5 Hz
    let hi_hz = 220.0 * 1.15; // 253.0 Hz
    let lo_bin = (lo_hz / doppler_bin_hz).max(0.0) as usize;
    let hi_bin = ((hi_hz / doppler_bin_hz) as usize).min(n - 1);
    if lo_bin >= hi_bin {
        return false;
    }
    spec[lo_bin..=hi_bin].iter().any(|&v| v >= 2.0 * median)
}

#[cfg(test)]
#[path = "tier_climb_tests.rs"]
mod tests;
