//! Discrete-state tier arbiter: tracks which phase a target is in and
//! publishes per-CPI transitions. The arbiter is the bookkeeping spine
//! of the 3-tier detector — it lets the receipt report per-tier Pd/Pfa
//! tables (which reviewers need to read tier-specific performance
//! instead of a mixed-bag confusion matrix).
//!
//! Transition rules (per `shahed-public-proxy-flight-envelope-v2`):
//!   * `None → Boost`: 3-of-5 boost-gate matches in the trailing 5-CPI
//!     window (accel ∈ [5, 20] m/s² AND speed ∈ [0, 35] m/s).
//!   * `Boost → ClimbOut`: |dv/dt| drops below 5 m/s² (boost burnout)
//!     AND speed reaches ≥ 25 m/s.
//!   * `ClimbOut → Cruise`: |dv/dt| < 0.5 m/s² for 10 consecutive CPIs
//!     AND |dh/dt| < 1 m/s for 10 CPIs (steady level flight).
//!   * `Cruise → None`: 5 missed CPIs (no in-gate radial speed).
//!
//! The arbiter does NOT compute per-tier detection logic — it only
//! decides which tier "owns" the current CPI. The `PhaseTieredDetector`
//! in `mod.rs` plumbs the tier-specific detectors against the arbiter's
//! verdict.
//!
//! Strict-open posture: transition thresholds are *public-proxy
//! expected* values derived from the dossier and do NOT claim
//! platform-specific signature truth.

use super::kinematic_gate::{boost_kinematic_gate, KinematicObservation, KinematicSample};
use super::Tier;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TierTransition {
    pub from: Tier,
    pub to: Tier,
    pub triggered_at_cpi: usize,
}

/// Discrete-state arbiter over `Tier`. Holds a history of (tier,
/// confidence) tuples per CPI evaluated.
#[derive(Debug, Clone)]
pub struct TierArbiter {
    pub current: Tier,
    pub consecutive_in_current: usize,
    pub history: Vec<(Tier, f32)>,
    /// CPI counter incremented on every call to `step`; used for
    /// emitted `TierTransition.triggered_at_cpi`.
    cpi_counter: usize,
    /// Trailing-window cache of recent climb steadiness signals
    /// (acceleration magnitude, altitude rate magnitude). Used by the
    /// `ClimbOut → Cruise` rule to count consecutive steady CPIs.
    steady_count: usize,
    /// Trailing miss count for `Cruise → None`.
    miss_count: usize,
}

impl Default for TierArbiter {
    fn default() -> Self {
        Self::new()
    }
}

impl TierArbiter {
    pub fn new() -> Self {
        Self {
            current: Tier::None,
            consecutive_in_current: 0,
            history: Vec::new(),
            cpi_counter: 0,
            steady_count: 0,
            miss_count: 0,
        }
    }

    /// Advance the arbiter one CPI. Returns `Some(transition)` if the
    /// arbiter changed tier during this step, `None` otherwise.
    pub fn step(&mut self, observation: &KinematicObservation) -> Option<TierTransition> {
        let from = self.current;
        let next = self.compute_next_tier(observation);
        self.cpi_counter += 1;
        // Update steady/miss bookkeeping based on the *observation*,
        // independent of whether the tier moved.
        self.update_steady_bookkeeping(observation);
        self.history
            .push((next, self.confidence(observation, next)));
        if next == from {
            self.consecutive_in_current += 1;
            return None;
        }
        self.current = next;
        self.consecutive_in_current = 1;
        // Reset steady/miss counters on transition.
        self.steady_count = 0;
        self.miss_count = 0;
        Some(TierTransition {
            from,
            to: next,
            triggered_at_cpi: self.cpi_counter,
        })
    }

    /// Pure-function tier computation given the current state and a
    /// fresh observation. Does not mutate `self` so it can be unit-tested
    /// in isolation.
    fn compute_next_tier(&self, observation: &KinematicObservation) -> Tier {
        match self.current {
            Tier::None => {
                if boost_gate_3_of_5(observation) {
                    Tier::Boost
                } else {
                    Tier::None
                }
            }
            Tier::Boost => {
                let speed = observation.abs_radial_speed();
                let accel = observation.abs_acceleration();
                if accel < 5.0 && speed >= 25.0 {
                    Tier::ClimbOut
                } else {
                    Tier::Boost
                }
            }
            Tier::ClimbOut => {
                // Hold ClimbOut until 10 consecutive steady CPIs trip.
                if self.steady_count + 1 >= 10 && climb_to_cruise_signal(observation) {
                    Tier::Cruise
                } else {
                    Tier::ClimbOut
                }
            }
            Tier::Cruise => {
                // Hold Cruise until 5 consecutive missed CPIs.
                if self.miss_count + 1 >= 5 {
                    Tier::None
                } else {
                    Tier::Cruise
                }
            }
        }
    }

    fn update_steady_bookkeeping(&mut self, observation: &KinematicObservation) {
        // Steady = accel < 0.5 m/s² AND altitude rate < 1 m/s.
        let accel_mag = observation.abs_acceleration();
        let climb_mag = observation.abs_altitude_rate();
        if accel_mag < 0.5 && climb_mag < 1.0 {
            self.steady_count = self.steady_count.saturating_add(1);
        } else {
            self.steady_count = 0;
        }

        // Miss = no in-gate radial speed (i.e. nothing classified as
        // an expected cruise speed). We use a coarse rule: speed < 25
        // m/s or speed > 200 m/s counts as a miss while in Cruise.
        let speed = observation.abs_radial_speed();
        let in_cruise_band = (40.0..=60.0).contains(&speed) || (100.0..=150.0).contains(&speed);
        if !in_cruise_band {
            self.miss_count = self.miss_count.saturating_add(1);
        } else {
            self.miss_count = 0;
        }
    }

    /// Coarse per-CPI confidence proxy: 1.0 if the next-tier gate fully
    /// accepts, 0.5 if the kinematic envelope is satisfied but the
    /// transition criteria are not yet met, 0.0 otherwise. Used so the
    /// arbiter publishes a stable `history` of (tier, confidence) pairs
    /// for downstream reports.
    fn confidence(&self, observation: &KinematicObservation, next: Tier) -> f32 {
        let speed = observation.abs_radial_speed();
        let accel = observation.abs_acceleration();
        match next {
            Tier::None => 0.0,
            Tier::Boost => {
                if (0.0..=35.0).contains(&speed) && (5.0..=20.0).contains(&accel) {
                    1.0
                } else {
                    0.5
                }
            }
            Tier::ClimbOut => {
                if (25.0..=60.0).contains(&speed) {
                    1.0
                } else {
                    0.5
                }
            }
            Tier::Cruise => {
                if (40.0..=60.0).contains(&speed) || (100.0..=150.0).contains(&speed) {
                    1.0
                } else {
                    0.5
                }
            }
        }
    }
}

/// Return `true` if the trailing 5-CPI window contains at least 3
/// boost-gate matches under the dossier's boost envelope.
fn boost_gate_3_of_5(observation: &KinematicObservation) -> bool {
    let gate = boost_kinematic_gate();
    let trailing = observation.trailing(6);
    if trailing.len() < 2 {
        return false;
    }
    let mut matches = 0usize;
    for pair in trailing.windows(2) {
        if cpi_pair_matches_boost(pair, &gate) {
            matches += 1;
        }
    }
    matches >= 3
}

/// Climb → cruise signal: in addition to steady_count bookkeeping
/// (handled by the arbiter), we also require the current sample to be
/// inside the dossier's cruise speed envelopes.
fn climb_to_cruise_signal(observation: &KinematicObservation) -> bool {
    let speed = observation.abs_radial_speed();
    let in_speed = (40.0..=60.0).contains(&speed) || (100.0..=150.0).contains(&speed);
    in_speed && observation.abs_acceleration() < 0.5 && observation.abs_altitude_rate() < 1.0
}

fn cpi_pair_matches_boost(
    pair: &[KinematicSample],
    gate: &crate::detectors::phase_tiered::kinematic_gate::KinematicGate,
) -> bool {
    let dt = pair[1].time_s - pair[0].time_s;
    if dt <= 0.0 {
        return false;
    }
    let accel = ((pair[1].radial_speed_mps - pair[0].radial_speed_mps) / dt).abs();
    let speed = pair[1].radial_speed_mps.abs();
    let alt = pair[1].altitude_agl_m;
    speed >= gate.radial_speed_mps_min
        && speed <= gate.radial_speed_mps_max
        && accel >= gate.accel_mps2_min
        && accel <= gate.accel_mps2_max
        && alt >= gate.altitude_agl_m_min
        && alt <= gate.altitude_agl_m_max
}

#[cfg(test)]
mod tests {
    use super::super::kinematic_gate::KinematicSample;
    use super::*;

    fn obs(samples: Vec<(f64, f64, f64)>) -> KinematicObservation {
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
    fn arbiter_starts_in_none() {
        let arb = TierArbiter::new();
        assert_eq!(arb.current, Tier::None);
        assert_eq!(arb.history.len(), 0);
    }

    #[test]
    fn arbiter_none_to_boost_transition() {
        // Provide a 6-sample window where every consecutive pair has
        // accel in [5, 20] m/s² at altitude 50 m AGL — the 3-of-5
        // rule trips and the arbiter moves to Boost on the first step.
        let mut arb = TierArbiter::new();
        let observation = obs(vec![
            (0.0, 0.0, 50.0),
            (1.0, 10.0, 60.0),
            (2.0, 20.0, 70.0),
            (3.0, 30.0, 80.0),
            (4.0, 35.0, 90.0),
            (5.0, 35.0, 100.0),
        ]);
        let transition = arb.step(&observation);
        assert!(transition.is_some(), "expected transition None → Boost");
        let t = transition.unwrap();
        assert_eq!(t.from, Tier::None);
        assert_eq!(t.to, Tier::Boost);
        assert_eq!(arb.current, Tier::Boost);
    }

    #[test]
    fn arbiter_climb_to_cruise_transition() {
        // Drop the arbiter directly into ClimbOut and feed 10 steady
        // CPIs in the piston cruise band → transitions to Cruise.
        let mut arb = TierArbiter {
            current: Tier::ClimbOut,
            consecutive_in_current: 5,
            history: Vec::new(),
            cpi_counter: 5,
            steady_count: 0,
            miss_count: 0,
        };
        // 12 samples → 11 finite-diff pairs. Each step is steady
        // (accel = 0, climb = 0) and at a piston-cluster speed.
        let mut last_transition = None;
        for k in 0..12usize {
            let observation = obs(vec![(k as f64, 50.0, 800.0), ((k + 1) as f64, 50.0, 800.0)]);
            let t = arb.step(&observation);
            if t.is_some() {
                last_transition = t;
            }
        }
        assert!(
            last_transition.is_some(),
            "expected ClimbOut → Cruise after 10 steady CPIs"
        );
        let t = last_transition.unwrap();
        assert_eq!(t.to, Tier::Cruise);
        assert_eq!(arb.current, Tier::Cruise);
    }

    #[test]
    fn arbiter_cruise_to_none_on_5_misses() {
        let mut arb = TierArbiter {
            current: Tier::Cruise,
            consecutive_in_current: 10,
            history: Vec::new(),
            cpi_counter: 50,
            steady_count: 10,
            miss_count: 0,
        };
        // 5 consecutive observations with speed 0 (lost track) →
        // transitions to None.
        let mut last_transition = None;
        for k in 0..5usize {
            let observation = obs(vec![(k as f64, 0.0, 800.0), ((k + 1) as f64, 0.0, 800.0)]);
            let t = arb.step(&observation);
            if t.is_some() {
                last_transition = t;
            }
        }
        assert!(last_transition.is_some(), "expected Cruise → None");
        assert_eq!(last_transition.unwrap().to, Tier::None);
    }
}
