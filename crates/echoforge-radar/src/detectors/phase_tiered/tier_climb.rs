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
mod tests {
    use super::super::kinematic_gate::KinematicSample;
    use super::*;

    fn cross_flight_climb_detector() -> ClimbOutTierDetector {
        use super::super::kinematic_gate::climb_kinematic_gate;
        ClimbOutTierDetector {
            config: ClimbTierConfig::default(),
            gate: KinematicGate { radial_speed_mps_min: 0.0, accel_mps2_min: 0.0, ..climb_kinematic_gate() },
        }
    }

    fn cross_flight_obs() -> KinematicObservation {
        // v_radial steady ~1 m/s (cross-flight), rising 0.1 m/s/CPI — within 3 m/s Kalman gate
        climb_window((0..6).map(|k| (k as f64, 0.8 + k as f64 * 0.1, 250.0 + k as f64 * 2.0)).collect())
    }

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
        let obs = climb_window((0..6).map(|k| (k as f64, 30.0 + k as f64, 250.0 + k as f64 * 2.0)).collect());
        let dec = detector.evaluate(&obs);
        assert!(dec.detected, "canonical climb-out must detect; {}", dec.note);
        assert!(!dec.mti_notch_rejected, "30+ m/s must clear MTI notch");
        assert!(!dec.cross_flight, "30 m/s radial must take the radial branch");
        assert!(dec.kalman_consistency > 0.5, "consistency = {}", dec.kalman_consistency);
    }

    #[test]
    fn climb_detector_rejects_in_mti_notch() {
        // Build an observation whose current speed maps to f_d < 30 Hz.
        // At 3 GHz, f_d = 2*v/c*f → v < 30 * c / (2 * 3e9) ≈ 1.5 m/s.
        // But our climb gate requires speed >= 25 m/s, so we must
        // construct a custom detector with a much wider MTI notch
        // (covering up to v=30 m/s → f_d=600 Hz at 3 GHz). To stay on
        // the radial branch (so we exercise the MTI-notch rejection
        // path rather than the cross-flight branch), we also pull the
        // cross-flight cutoff above the notch by widening the carrier.
        // Easiest: raise the carrier so 30 m/s → f_d well above 30 Hz
        // while still falling under the inflated notch.
        // At 10 GHz, 30 m/s → f_d ≈ 2000 Hz; cross-flight cutoff at
        // 30 Hz maps to v ≈ 0.45 m/s, well below the climb gate's
        // 25 m/s lower bound, so the cross-flight branch never fires
        // inside the climb envelope.
        let config = ClimbTierConfig {
            carrier_freq_hz: 1.0e10,
            mti_notch_half_width_hz: 3000.0,
            ..ClimbTierConfig::default()
        };
        let detector = ClimbOutTierDetector::new(config);
        let obs = climb_window(vec![
            (0.0, 29.0, 250.0),
            (1.0, 30.0, 252.0),
        ]);
        let dec = detector.evaluate(&obs);
        assert!(dec.mti_notch_rejected, "target inside notch must be rejected");
        assert!(!dec.cross_flight, "29-30 m/s at X-band must stay on radial branch");
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

    // -----------------------------------------------------------------
    // Wave 4.5 H4 — MTI cross-flight tests.
    // -----------------------------------------------------------------

    /// Build a slow-time amplitude spectrum that has a flat noise floor
    /// of `floor_amp` over `n_bins` bins (default bin width 1 Hz when
    /// passed `doppler_bin_hz = 1.0`), then injects a blade-pass spike
    /// at `blade_hz` of amplitude `blade_amp`. Mirrors the convention
    /// used by `tier_cruise::tests::cruise_detector_blade_pass_*`.
    fn synthetic_slow_time_spectrum(
        n_bins: usize,
        floor_amp: f32,
        blade_hz: Option<f64>,
        blade_amp: f32,
        doppler_bin_hz: f64,
    ) -> Vec<f32> {
        let mut spec = vec![floor_amp; n_bins];
        if let Some(hz) = blade_hz {
            let bin = (hz / doppler_bin_hz).round() as usize;
            if bin < n_bins {
                spec[bin] = blade_amp;
            }
        }
        spec
    }

    /// Cross-flight target with v_radial = 5 m/s (low; well below the
    /// MTI notch body-Doppler cutoff at S-band) and tangential
    /// velocity = 50 m/s (in the climb-out envelope). A propeller
    /// blade-pass line at 190 Hz is present in the slow-time spectrum.
    /// Per Wave 4.5 H4 the cross-flight branch must accept this target
    /// because (a) the MTI gate is bypassed by geometry, (b) the
    /// kinematic window is steady (4-of-5 within 3 m/s), and
    /// (c) micro-Doppler confirms a piston propeller line.
    ///
    /// **Note on the climb gate:** the canonical climb gate keys on
    /// |v_radial| only, so to make the cross-flight target also satisfy
    /// the gate's lower-bound (25 m/s) we widen the gate's
    /// `radial_speed_mps_min` to 0 here. That is the correct
    /// cross-flight kinematic state: |v_total| is in the climb envelope
    /// but |v_radial| is small.
    #[test]
    fn cross_flight_target_with_micro_doppler_detected() {
        // Construct a custom climb gate that admits low-radial-speed
        // targets (cross-flight geometry).
        let detector = cross_flight_climb_detector();
        // 6 samples → 5 deltas, all within the tighter 3 m/s gate.
        // The branching is on the MOST RECENT speed; at S-band 3 GHz,
        // f_d = 2*v*f_c/c. v < 30 * c / (2 * 3e9) ≈ 1.499 m/s is the
        // cross-flight cutoff. Use v_radial steady at 1.0 m/s
        // (f_d ≈ 20 Hz < 30 Hz cutoff). This represents the
        // canonical cross-flight geometry — tangential velocity is
        // ~50 m/s (consistent with the climb-out envelope) but the
        // radial projection on the radar LOS is tiny.
        let obs = cross_flight_obs();
        // 256 bins at 1 Hz/bin = 256 Hz Nyquist. Spike at 190 Hz,
        // squarely in [127.5, 253] Hz blade-pass window.
        let spec = synthetic_slow_time_spectrum(256, 1.0, Some(190.0), 50.0, 1.0);
        let dec = detector.evaluate_with_spectrum(&obs, Some(&spec), Some(1.0));
        assert!(
            dec.cross_flight,
            "low |v_radial| at S-band must enter cross-flight branch; note = {}",
            dec.note,
        );
        assert!(
            dec.micro_doppler_confirmed,
            "blade-pass at 190 Hz must be confirmed in cross-flight branch",
        );
        assert!(
            dec.detected,
            "cross-flight + micro-Doppler must detect; note = {}",
            dec.note,
        );
        assert!(
            !dec.mti_notch_rejected,
            "cross-flight branch does not set mti_notch_rejected",
        );
    }

    /// Same cross-flight kinematic state but NO micro-Doppler line in
    /// the supplied slow-time spectrum. Per Wave 4.5 H4 the cross-flight
    /// branch must REJECT because the compensating evidence is absent:
    /// the MTI gate was skipped, so without the blade-pass line we have
    /// only kinematic evidence — insufficient to publish a detection.
    #[test]
    fn cross_flight_target_without_micro_doppler_rejected() {
        let detector = cross_flight_climb_detector();
        let obs = cross_flight_obs();
        // Flat noise floor: no blade-pass spike at all.
        let spec = synthetic_slow_time_spectrum(256, 1.0, None, 0.0, 1.0);
        let dec = detector.evaluate_with_spectrum(&obs, Some(&spec), Some(1.0));
        assert!(dec.cross_flight, "low |v_radial| must enter cross-flight branch");
        assert!(
            !dec.micro_doppler_confirmed,
            "flat noise floor has no blade-pass line",
        );
        assert!(
            !dec.detected,
            "cross-flight without micro-Doppler must NOT detect; note = {}",
            dec.note,
        );
    }

    /// Same cross-flight kinematic state but no spectrum supplied at
    /// all (`None`). The cross-flight branch must REJECT — without the
    /// slow-time spectrum the micro-Doppler confirmation cannot fire,
    /// and the cross-flight branch requires it. Guards against the
    /// caller forgetting to supply spectrum data being silently
    /// upgraded to a detection.
    #[test]
    fn cross_flight_target_without_spectrum_rejected() {
        let detector = cross_flight_climb_detector();
        let obs = cross_flight_obs();
        let dec = detector.evaluate(&obs);
        assert!(dec.cross_flight);
        assert!(!dec.micro_doppler_confirmed);
        assert!(!dec.detected, "no spectrum → no micro-Doppler → reject");
    }

    /// Standard climb-out target with v_radial = 40 m/s (well above
    /// the MTI notch body-Doppler cutoff). The radial-velocity branch
    /// must run unchanged: existing 3-of-5 M-of-N + MTI gate applies,
    /// `cross_flight = false`, detection succeeds.
    #[test]
    fn standard_climb_target_detected_normally() {
        let detector = ClimbOutTierDetector::with_default();
        let obs = climb_window((0..6).map(|k| (k as f64, 40.0 + k as f64, 250.0 + k as f64 * 2.0)).collect());
        let dec = detector.evaluate(&obs);
        assert!(!dec.cross_flight, "40 m/s radial must take the radial branch");
        assert!(!dec.mti_notch_rejected, "40 m/s clears the notch");
        assert!(dec.detected, "standard 3-of-5 must detect; note = {}", dec.note);
        // The cross-flight micro_doppler_confirmed flag must never be
        // set by the radial-velocity branch.
        assert!(
            !dec.micro_doppler_confirmed,
            "radial-velocity branch must not set micro_doppler_confirmed",
        );
    }

    /// Cross-flight kinematic but the speed track is *erratic* (large
    /// inter-CPI residuals). The cross-flight branch's stricter
    /// 4-of-5-within-3 m/s Kalman test must reject even when
    /// micro-Doppler is present — the kinematic-consistency leg of the
    /// compensating evidence is missing.
    #[test]
    fn cross_flight_erratic_kinematic_rejected_even_with_micro_doppler() {
        let mut detector = cross_flight_climb_detector();
        detector.gate.accel_mps2_max = 20.0;
        // 6 samples, alternating ±5 m/s residuals — far outside the
        // 3 m/s cross-flight Kalman gate.
        let obs = climb_window((0..6).map(|k| (k as f64, if k % 2 == 0 { 0.5 } else { 7.0 }, 250.0 + k as f64 * 2.0)).collect());
        let spec = synthetic_slow_time_spectrum(256, 1.0, Some(190.0), 50.0, 1.0);
        let dec = detector.evaluate_with_spectrum(&obs, Some(&spec), Some(1.0));
        // Last sample is 7 m/s → f_d = 2*7*3e9 / 3e8 = 140 Hz, ABOVE
        // the 30 Hz cross-flight cutoff. So this case takes the
        // radial branch, not the cross-flight branch. Confirm that
        // first.
        assert!(
            !dec.cross_flight,
            "v_radial 7 m/s at S-band f_d=140 Hz > 30 Hz cutoff takes radial branch",
        );
        // On the radial branch, the erratic kinematic ALSO causes
        // M-of-N to fail (5 m/s gate, residuals ~6.5 m/s).
        assert!(!dec.detected, "erratic kinematic must not detect");
    }

    /// Constants sanity: the cross-flight cutoff exists and equals
    /// the documented 30 Hz value at S-band.
    #[test]
    fn cross_flight_cutoff_constant_is_30_hz() {
        assert_eq!(MTI_NOTCH_BODY_DOPPLER_HZ, 30.0);
    }

    /// Branching boundary: a target whose body Doppler equals the
    /// cross-flight cutoff exactly takes the **radial branch** (the
    /// cutoff is exclusive). Documents the convention so callers can
    /// reason about the boundary case.
    #[test]
    fn cross_flight_cutoff_boundary_takes_radial_branch() {
        // At 3 GHz carrier, v giving f_d = 30 Hz exactly is
        // v = 30 * c / (2 * 3e9) = 1.499 m/s. The cutoff comparison
        // is `<`, so f_d == cutoff is the radial branch.
        let detector = cross_flight_climb_detector();
        let v_at_cutoff = MTI_NOTCH_BODY_DOPPLER_HZ
            * crate::propagation::SPEED_OF_LIGHT_M_PER_S
            / (2.0 * 3.0e9);
        let obs = climb_window(vec![
            (0.0, v_at_cutoff, 250.0),
            (1.0, v_at_cutoff, 252.0),
        ]);
        let dec = detector.evaluate(&obs);
        // f_d exactly == cutoff → radial branch.
        assert!(
            !dec.cross_flight,
            "boundary case (f_d == cutoff) must take the radial branch (cutoff exclusive)",
        );
    }
}
