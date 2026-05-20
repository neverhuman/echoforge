//! Phase-aware 3-tier detector for Iranian Shahed-136-class one-way-attack
//! drone family. Public-proxy parameters cited to the Wave-A
//! `shahed-public-proxy-flight-envelope-v2` physics dossier
//! (`object-packs/public-proxy-v1/physics_dossier.md`).
//!
//! Strict-open posture: parameters are sourced from open literature and
//! the public-proxy dossier; this module does NOT claim platform-specific
//! signature truth. The tier-arbiter publishes per-tier Pd/Pfa tables
//! exactly because reviewers need a separable per-phase confusion matrix.
//!
//! The detector layers three phase-specific rules through a discrete
//! arbiter state machine:
//!
//! | Tier      | Speed (m/s) | Accel (m/s²) | Altitude AGL (m) | Dossier phase |
//! |-----------|-------------|--------------|------------------|---------------|
//! | Boost     | 0 – 35      | 5 – 20       | 0 – 200          | RATO burn 1–3 s |
//! | ClimbOut  | 25 – 60     | 0.1 – 2.0    | 30 – 1500        | Post-boost piston |
//! | Cruise (piston) | 40 – 60 | 0.0 – 0.5 | 30 – 3000        | Steady piston cruise |
//! | Cruise (jet)    | 100 – 150 | 0.0 – 0.5 | 30 – 3000      | Shahed-238 cruise |
//!
//! Each per-tier detector publishes its own decision struct
//! (`BoostDecision`, `ClimbDecision`, `CruiseDecision`); the
//! `PhaseTieredDetector::evaluate_cpi` entry point fuses them into a
//! single `PhaseTieredDecision` with `tier`, `confidence`,
//! `speed_estimate_mps`, `propulsion_class`, `horizon_blocked`,
//! `kinematic_consistency`, and `micro_doppler_confirmed`.

pub mod kinematic_gate;
pub mod speed_classifier;
pub mod tier_arbiter;
pub mod tier_boost;
pub mod tier_climb;
pub mod tier_cruise;
mod tier_cruise_helpers;

pub use kinematic_gate::{
    boost_kinematic_gate, climb_kinematic_gate, cruise_kinematic_gate_jet,
    cruise_kinematic_gate_piston, KinematicGate, KinematicObservation, KinematicSample,
};
pub use speed_classifier::{PropulsionClass, SpeedClassifier};
pub use tier_arbiter::{TierArbiter, TierTransition};
pub use tier_boost::{
    classify_boost_sub_state, BoostDecision, BoostSubState, BoostThrustProfile, BoostTierConfig,
    BoostTierDetector,
};
pub use tier_climb::{
    ClimbDecision, ClimbOutTierDetector, ClimbTierConfig, MTI_NOTCH_BODY_DOPPLER_HZ,
};
pub use tier_cruise::{CruiseDecision, CruiseTierConfig, CruiseTierDetector};

/// Discrete tier states a track can be in. The arbiter publishes one of
/// these per CPI; per-tier Pd/Pfa reports separate detections by the
/// emitting tier so reviewers can read tier-specific performance.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Tier {
    None,
    Boost,
    ClimbOut,
    Cruise,
}

/// Outcome of a per-CPI evaluation by the full tier-aware pipeline.
#[derive(Debug, Clone)]
pub struct PhaseTieredDecision {
    pub tier: Tier,
    pub confidence: f32,
    pub speed_estimate_mps: Option<f64>,
    pub speed_uncertainty_mps: Option<f64>,
    pub propulsion_class: PropulsionClass,
    /// Honest physics outcome (Tier 1): when `true`, the geometry is
    /// below the radar's LOS and the detector cannot see it. Published
    /// separately from `tier`/`confidence` so reviewers see "blocked by
    /// physics" instead of a silent miss.
    pub horizon_blocked: bool,
    /// Kalman-residual-based kinematic consistency score in [0, 1].
    pub kinematic_consistency: f32,
    /// Tier 3 propeller-line check: when `true`, a blade-pass line at
    /// 150–220 Hz ± 15% (piston) or compressor-band line (jet) was
    /// observed in the supplied Doppler spectrum.
    pub micro_doppler_confirmed: bool,
}

impl PhaseTieredDecision {
    pub fn empty() -> Self {
        Self {
            tier: Tier::None,
            confidence: 0.0,
            speed_estimate_mps: None,
            speed_uncertainty_mps: None,
            propulsion_class: PropulsionClass::None,
            horizon_blocked: false,
            kinematic_consistency: 0.0,
            micro_doppler_confirmed: false,
        }
    }
}

/// Configuration for the full phase-aware detector. Holds tier-specific
/// sub-configurations so each tier can be tuned independently.
#[derive(Debug, Clone, Copy, Default)]
pub struct PhaseTieredConfig {
    pub boost: BoostTierConfig,
    pub climb: ClimbTierConfig,
    pub cruise: CruiseTierConfig,
}

/// The 3-tier phase-aware detector. Holds the arbiter and the per-tier
/// detectors. Each CPI is run through the arbiter to determine the
/// current tier; the matching detector emits its decision; the fused
/// `PhaseTieredDecision` is published.
#[derive(Debug, Clone)]
pub struct PhaseTieredDetector {
    pub arbiter: TierArbiter,
    pub boost: BoostTierDetector,
    pub climb: ClimbOutTierDetector,
    pub cruise: CruiseTierDetector,
}

impl Default for PhaseTieredDetector {
    fn default() -> Self {
        Self::new(PhaseTieredConfig::default())
    }
}

impl PhaseTieredDetector {
    pub fn new(config: PhaseTieredConfig) -> Self {
        Self {
            arbiter: TierArbiter::new(),
            boost: BoostTierDetector::new(config.boost),
            climb: ClimbOutTierDetector::new(config.climb),
            cruise: CruiseTierDetector::new(config.cruise),
        }
    }

    pub fn current_tier(&self) -> Tier {
        self.arbiter.current
    }

    /// Evaluate one CPI given the kinematic observation window and an
    /// optional post-MTD power spectrum (used by Tier 3 for OS-CFAR and
    /// micro-Doppler confirmation).
    pub fn evaluate_cpi(
        &mut self,
        observation: &KinematicObservation,
        mtd_power_spectrum: Option<&[f32]>,
        doppler_bin_hz: Option<f64>,
    ) -> PhaseTieredDecision {
        // (a) Advance the arbiter.
        let _ = self.arbiter.step(observation);

        // (b) Run the per-tier detector and collect tier-specific fields.
        let mut out = PhaseTieredDecision::empty();
        out.tier = self.arbiter.current;
        out.speed_estimate_mps = observation.current_radial_speed_mps();

        match self.arbiter.current {
            Tier::None => {
                // Run the boost detector to publish horizon_blocked
                // honesty even before the arbiter has latched a tier.
                let boost = self.boost.evaluate(observation);
                out.horizon_blocked = boost.horizon_blocked;
                out.confidence = 0.0;
            }
            Tier::Boost => {
                let boost = self.boost.evaluate(observation);
                out.horizon_blocked = boost.horizon_blocked;
                out.confidence = if boost.detected {
                    boost.mof_n_ratio.max(0.5)
                } else {
                    boost.mof_n_ratio
                };
            }
            Tier::ClimbOut => {
                let climb = self.climb.evaluate(observation);
                out.kinematic_consistency = climb.kalman_consistency;
                out.confidence = if climb.detected {
                    climb.mof_n_ratio.max(0.5)
                } else {
                    climb.mof_n_ratio
                };
            }
            Tier::Cruise => {
                let cruise = self
                    .cruise
                    .evaluate(observation, mtd_power_spectrum, doppler_bin_hz);
                out.kinematic_consistency = cruise.kalman_consistency;
                out.propulsion_class = cruise.propulsion_class;
                out.micro_doppler_confirmed = cruise.micro_doppler_confirmed;
                out.confidence = if cruise.detected {
                    cruise.mof_n_ratio.max(0.5)
                } else {
                    cruise.mof_n_ratio
                };
            }
        }

        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kinematic_gate::KinematicSample;

    fn obs(samples: Vec<(f64, f64, f64)>) -> KinematicObservation {
        KinematicObservation::new(
            samples
                .into_iter()
                .map(|(t, v, h)| KinematicSample::new(t, v, h))
                .collect(),
            5_000.0,
            20.0,
        )
    }

    #[test]
    fn detector_starts_in_none_with_empty_history() {
        let det = PhaseTieredDetector::default();
        assert_eq!(det.current_tier(), Tier::None);
    }

    #[test]
    fn detector_advances_to_boost_on_boost_kinematics() {
        let mut det = PhaseTieredDetector::default();
        let observation = obs(vec![
            (0.0, 0.0, 50.0),
            (1.0, 10.0, 60.0),
            (2.0, 20.0, 70.0),
            (3.0, 30.0, 80.0),
            (4.0, 35.0, 90.0),
            (5.0, 35.0, 100.0),
        ]);
        let dec = det.evaluate_cpi(&observation, None, None);
        assert_eq!(dec.tier, Tier::Boost);
    }
}
