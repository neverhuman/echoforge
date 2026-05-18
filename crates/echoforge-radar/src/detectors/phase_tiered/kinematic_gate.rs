//! Kinematic gate primitives shared by all three tiers.
//!
//! The phase-aware detector evaluates a sliding window of kinematic
//! observations (radial speed, altitude, time-stamped) and asks each
//! `KinematicGate` whether the current motion state lies within the
//! envelope of a given phase. Speed / acceleration / altitude bounds
//! are taken from the Wave-A `shahed-public-proxy-flight-envelope-v2`
//! dossier at `object-packs/public-proxy-v1/physics_dossier.md`
//! (cruise speed 50–55 m/s, min sustainable airspeed 35–40 m/s,
//! climb rate 2–4 m/s typical, low-ingress altitude 60–200 m AGL,
//! booster exit velocity 25–35 m/s in ~10–30 s to powered cruise,
//! body acceleration ~1–2g during boost).
//!
//! Strict-open posture: these envelopes describe an *open-source
//! public-proxy* of the Shahed-136 class and do NOT claim
//! platform-specific signature truth.

/// One sample in a kinematic observation window. `time_s` is a monotonic
/// wall-clock-like timestamp; `radial_speed_mps` is the line-of-sight
/// component (positive = closing in radar convention is irrelevant here
/// because the gates use magnitude); `altitude_agl_m` is the target's
/// height above local ground level.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KinematicSample {
    pub time_s: f64,
    pub radial_speed_mps: f64,
    pub altitude_agl_m: f64,
}

impl KinematicSample {
    pub fn new(time_s: f64, radial_speed_mps: f64, altitude_agl_m: f64) -> Self {
        Self {
            time_s,
            radial_speed_mps,
            altitude_agl_m,
        }
    }
}

/// A sliding window of kinematic samples plus the geometry needed to
/// run LOS-horizon checks (range from radar and the radar's own antenna
/// height AGL). The window is in time-order, oldest first.
#[derive(Debug, Clone)]
pub struct KinematicObservation {
    pub samples: Vec<KinematicSample>,
    pub range_m: f64,
    pub radar_altitude_agl_m: f64,
}

impl KinematicObservation {
    pub fn new(samples: Vec<KinematicSample>, range_m: f64, radar_altitude_agl_m: f64) -> Self {
        Self {
            samples,
            range_m,
            radar_altitude_agl_m,
        }
    }

    /// Most recent radial speed in m/s, or `None` if the window is empty.
    pub fn current_radial_speed_mps(&self) -> Option<f64> {
        self.samples.last().map(|s| s.radial_speed_mps)
    }

    /// Most recent altitude AGL in m, or `None` if the window is empty.
    pub fn current_altitude_agl_m(&self) -> Option<f64> {
        self.samples.last().map(|s| s.altitude_agl_m)
    }

    /// Finite-difference acceleration in m/s² between the last two
    /// samples, or `None` if fewer than two samples exist or the time
    /// delta is non-positive (degenerate or duplicate timestamps).
    pub fn current_acceleration_mps2(&self) -> Option<f64> {
        if self.samples.len() < 2 {
            return None;
        }
        let n = self.samples.len();
        let a = self.samples[n - 2];
        let b = self.samples[n - 1];
        let dt = b.time_s - a.time_s;
        if dt <= 0.0 {
            return None;
        }
        Some((b.radial_speed_mps - a.radial_speed_mps) / dt)
    }

    /// Finite-difference altitude rate (climb rate) in m/s between the
    /// last two samples, or `None` if fewer than two samples or
    /// non-positive time delta.
    pub fn altitude_rate_mps(&self) -> Option<f64> {
        if self.samples.len() < 2 {
            return None;
        }
        let n = self.samples.len();
        let a = self.samples[n - 2];
        let b = self.samples[n - 1];
        let dt = b.time_s - a.time_s;
        if dt <= 0.0 {
            return None;
        }
        Some((b.altitude_agl_m - a.altitude_agl_m) / dt)
    }

    /// Trailing-window slice helper: returns the last `n` samples
    /// (or all samples if fewer than `n` exist).
    pub fn trailing(&self, n: usize) -> &[KinematicSample] {
        let len = self.samples.len();
        let start = len.saturating_sub(n);
        &self.samples[start..]
    }
}

/// A single tier's kinematic acceptance envelope. Speed / acceleration
/// magnitude / altitude AGL gates are applied as a conjunction; a sample
/// must satisfy *all three* to be accepted.
///
/// The gate compares against `|speed|` and `|accel|` so the sign of the
/// radial component (closing vs receding) does not flip acceptance.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KinematicGate {
    pub radial_speed_mps_min: f64,
    pub radial_speed_mps_max: f64,
    pub accel_mps2_min: f64,
    pub accel_mps2_max: f64,
    pub altitude_agl_m_min: f64,
    pub altitude_agl_m_max: f64,
}

impl KinematicGate {
    /// Returns `true` iff the most-recent sample in `obs` satisfies the
    /// speed / acceleration / altitude envelope. If acceleration cannot
    /// be computed (single-sample window), the acceleration check is
    /// satisfied iff the gate's acceleration window includes 0
    /// (i.e. `accel_mps2_min <= 0`).
    pub fn accepts(&self, obs: &KinematicObservation) -> bool {
        let speed = match obs.current_radial_speed_mps() {
            Some(v) => v.abs(),
            None => return false,
        };
        let altitude = match obs.current_altitude_agl_m() {
            Some(v) => v,
            None => return false,
        };
        if speed < self.radial_speed_mps_min || speed > self.radial_speed_mps_max {
            return false;
        }
        if altitude < self.altitude_agl_m_min || altitude > self.altitude_agl_m_max {
            return false;
        }
        let accel = obs.current_acceleration_mps2();
        match accel {
            Some(a) => {
                let a = a.abs();
                a >= self.accel_mps2_min && a <= self.accel_mps2_max
            }
            None => self.accel_mps2_min <= 0.0,
        }
    }
}

/// **Tier 1 BOOST** kinematic envelope — rocket-assisted launch
/// transient. Public-proxy timing per
/// `shahed-public-proxy-flight-envelope-v2`: solid-propellant booster
/// burn typically 1–3 s, producing body acceleration of ~1–2g
/// (5–20 m/s²) as the airframe goes from rail-exit velocity (~9 m/s) to
/// release velocity (~25–35 m/s). Initial climb angle 30–45° above
/// horizontal places the airframe well below 200 m AGL during the burn.
/// Often below LOS for ground radar at long range; LOS honesty is
/// enforced separately by `BoostTierDetector::evaluate`.
pub fn boost_kinematic_gate() -> KinematicGate {
    KinematicGate {
        radial_speed_mps_min: 0.0,
        radial_speed_mps_max: 35.0,
        accel_mps2_min: 5.0,
        accel_mps2_max: 20.0,
        altitude_agl_m_min: 0.0,
        altitude_agl_m_max: 200.0,
    }
}

/// **Tier 2 CLIMB-OUT** kinematic envelope — post-boost piston engine
/// takeover. Public-proxy timing: 3–30 s after launch; airframe
/// transitions from ~30 m/s to cruise (50–55 m/s) under low piston
/// thrust. Climb rate 2–4 m/s typical. Acceleration band is the gentle
/// piston-driven build-up *after* the booster has separated, so the
/// upper bound is well below the boost band (which is 5–20 m/s²).
/// First robust opportunity for ground radar acquisition.
pub fn climb_kinematic_gate() -> KinematicGate {
    KinematicGate {
        radial_speed_mps_min: 25.0,
        radial_speed_mps_max: 60.0,
        accel_mps2_min: 0.1,
        accel_mps2_max: 2.0,
        altitude_agl_m_min: 30.0,
        altitude_agl_m_max: 1500.0,
    }
}

/// **Tier 3 CRUISE (piston cluster)** — stable propulsion + flight.
/// Public-proxy: piston cluster speed 40–60 m/s (max 55–60 per
/// dossier). Acceleration is bounded near zero because the airframe
/// is in steady cruise. Altitude spans the low-ingress band (60–200 m
/// AGL) through the dossier's max (~3000 m AGL).
pub fn cruise_kinematic_gate_piston() -> KinematicGate {
    KinematicGate {
        radial_speed_mps_min: 40.0,
        radial_speed_mps_max: 60.0,
        accel_mps2_min: 0.0,
        accel_mps2_max: 0.5,
        altitude_agl_m_min: 30.0,
        altitude_agl_m_max: 3000.0,
    }
}

/// **Tier 3 CRUISE (jet cluster)** — Shahed-238 variant per dossier.
/// Public-proxy: 100–150 m/s cruise (~400–520 km/h). Acceleration is
/// again bounded near zero (steady cruise); altitude band matches the
/// piston cruise envelope (jet variant has not been publicly reported
/// flying outside the piston cluster's altitude band).
pub fn cruise_kinematic_gate_jet() -> KinematicGate {
    KinematicGate {
        radial_speed_mps_min: 100.0,
        radial_speed_mps_max: 150.0,
        accel_mps2_min: 0.0,
        accel_mps2_max: 0.5,
        altitude_agl_m_min: 30.0,
        altitude_agl_m_max: 3000.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn obs_single(speed: f64, alt: f64) -> KinematicObservation {
        KinematicObservation::new(
            vec![KinematicSample::new(0.0, speed, alt)],
            10_000.0,
            20.0,
        )
    }

    fn obs_pair(speed0: f64, alt0: f64, speed1: f64, alt1: f64, dt: f64) -> KinematicObservation {
        KinematicObservation::new(
            vec![
                KinematicSample::new(0.0, speed0, alt0),
                KinematicSample::new(dt, speed1, alt1),
            ],
            10_000.0,
            20.0,
        )
    }

    #[test]
    fn boost_gate_accepts_canonical() {
        // Speed 15 m/s, accel 12 m/s² (about 1.2g), altitude 80 m AGL
        // — all comfortably within the dossier's boost envelope.
        let gate = boost_kinematic_gate();
        let obs = obs_pair(3.0, 80.0, 15.0, 80.0, 1.0);
        // dv/dt = (15 - 3) / 1 = 12 m/s² ∈ [5, 20]; speed 15 ∈ [0, 35];
        // altitude 80 ∈ [0, 200].
        assert!(gate.accepts(&obs), "boost gate must accept canonical");
    }

    #[test]
    fn boost_gate_rejects_too_high() {
        // Altitude 5000 m far outside dossier's boost band (0–200 m).
        let gate = boost_kinematic_gate();
        let obs = obs_pair(3.0, 5000.0, 15.0, 5000.0, 1.0);
        assert!(!gate.accepts(&obs), "boost gate must reject 5000 m altitude");
    }

    #[test]
    fn climb_gate_accepts_canonical() {
        // 40 m/s speed (post-boost, below piston cruise), 1.0 m/s² accel,
        // 250 m AGL altitude — squarely within the dossier's climb-out band.
        let gate = climb_kinematic_gate();
        let obs = obs_pair(39.0, 250.0, 40.0, 250.0, 1.0);
        assert!(gate.accepts(&obs), "climb gate must accept canonical");
    }

    #[test]
    fn climb_gate_rejects_aircraft_speed() {
        // Speed 80 m/s falls in the unmodeled gap (60–100 m/s) — above
        // the climb-out cap (60 m/s).
        let gate = climb_kinematic_gate();
        let obs = obs_pair(79.5, 250.0, 80.0, 250.0, 1.0);
        assert!(
            !gate.accepts(&obs),
            "climb gate must reject 80 m/s (above piston cap)"
        );
    }

    #[test]
    fn cruise_gate_piston_accepts_50mps() {
        let gate = cruise_kinematic_gate_piston();
        let obs = obs_pair(50.0, 800.0, 50.0, 800.0, 1.0);
        assert!(gate.accepts(&obs), "piston cruise gate must accept 50 m/s");
    }

    #[test]
    fn cruise_gate_jet_accepts_120mps() {
        let gate = cruise_kinematic_gate_jet();
        let obs = obs_pair(120.0, 800.0, 120.0, 800.0, 1.0);
        assert!(gate.accepts(&obs), "jet cruise gate must accept 120 m/s");
    }

    #[test]
    fn cruise_gate_rejects_ambiguous_75mps() {
        // 75 m/s sits in the explicit dossier gap [65, 100]; both
        // piston and jet cruise gates must reject.
        let piston = cruise_kinematic_gate_piston();
        let jet = cruise_kinematic_gate_jet();
        let obs = obs_pair(75.0, 800.0, 75.0, 800.0, 1.0);
        assert!(!piston.accepts(&obs), "piston gate must reject 75 m/s");
        assert!(!jet.accepts(&obs), "jet gate must reject 75 m/s");
    }

    #[test]
    fn observation_returns_none_for_empty_window() {
        let obs = KinematicObservation::new(vec![], 1000.0, 20.0);
        assert_eq!(obs.current_radial_speed_mps(), None);
        assert_eq!(obs.current_altitude_agl_m(), None);
        assert_eq!(obs.current_acceleration_mps2(), None);
        assert_eq!(obs.altitude_rate_mps(), None);
    }

    #[test]
    fn observation_single_sample_has_no_accel() {
        let obs = obs_single(50.0, 200.0);
        assert_eq!(obs.current_acceleration_mps2(), None);
        assert_eq!(obs.altitude_rate_mps(), None);
    }

    #[test]
    fn observation_finite_diff_accel_and_climb_rate() {
        // Speed 30 → 40 over 2 s = 5 m/s²; altitude 100 → 106 over 2 s = 3 m/s.
        let obs = obs_pair(30.0, 100.0, 40.0, 106.0, 2.0);
        let a = obs.current_acceleration_mps2().unwrap();
        let c = obs.altitude_rate_mps().unwrap();
        assert!((a - 5.0).abs() < 1e-9, "accel = {a}");
        assert!((c - 3.0).abs() < 1e-9, "climb_rate = {c}");
    }
}
