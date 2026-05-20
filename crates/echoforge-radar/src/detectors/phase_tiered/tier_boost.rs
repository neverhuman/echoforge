//! Tier 1 — BOOST phase detector. Rocket-assisted launch transient
//! detection with explicit line-of-sight horizon honesty.
//!
//! Public-proxy timing per the Wave-A
//! `shahed-public-proxy-flight-envelope-v2` dossier
//! (`object-packs/public-proxy-v1/physics_dossier.md`):
//! solid-propellant booster burn typically 1–3 s, produces body
//! acceleration of ~1–2g (5–20 m/s²) as the airframe goes from rail-exit
//! velocity (~9 m/s) to release velocity (~25–35 m/s). Initial climb
//! angle 30–45° above horizontal. Often below the radar horizon for
//! ground radar at >30 km range.
//!
//! **Wave 4.5 Lane H5 refinement (booster-burn thrust profile).** The
//! pre-H5 model treated boost as a single constant-acceleration band.
//! Real solid-rocket-motor (SRM) thrust profiles have three distinct
//! sub-phases — boost burn, separation transient, and post-separation
//! sustain — per RUSI/CSIS open reporting + standard progressive-grain
//! SRM thrust curves in Sutton & Biblarz, *Rocket Propulsion Elements*,
//! 9th ed., ch. 12 (esp. fig. 12-7). The `BoostThrustProfile` model
//! below captures the rise-plateau-decay shape and the post-boost
//! piston phase; `BoostTierDetector::evaluate` classifies the measured
//! acceleration into a `BoostSubState` accordingly.
//!
//! Strict-open posture: the boost-phase detection rule, LOS-horizon
//! honesty, AND the SRM thrust profile publish *public-proxy expected*
//! Pd/Pfa for a Shahed-class launch transient and do NOT claim
//! platform-specific signature truth. The thrust-profile parameters
//! are typical small-RATO numbers from Sutton/Biblarz with dossier
//! envelope bounds, NOT measured values from any specific motor.

#[cfg(not(test))]
use crate::propagation::STANDARD_K_FACTOR;
#[cfg(test)]
pub use crate::propagation::{min_target_altitude_for_los_m, STANDARD_K_FACTOR};

use super::kinematic_gate::{boost_kinematic_gate, KinematicGate, KinematicObservation};

#[path = "tier_boost_impl.rs"]
mod tier_boost_impl;

/// Solid-rocket-motor thrust profile for Shahed-class booster-assisted
/// launch. Per RUSI/CSIS open reporting (Wave-A `physics_dossier.md`
/// §5 *Launch mode*) + standard progressive-grain SRM thrust curves
/// from Sutton & Biblarz, *Rocket Propulsion Elements*, 9th ed., ch.
/// 12 (esp. fig. 12-7).
///
/// The thrust curve has three sub-phases:
///   1. **Boost burn** (`0` → `burn_duration_s`): rapid rise to peak
///      thrust then plateau, ~1-3 s total duration.
///   2. **Separation transient** (`burn_duration_s` → `burn_duration_s
///      + separation_transient_s`): brief drop-and-decelerate as the
///      booster physically decouples from the airframe.
///   3. **Post-separation sustain** (`t > burn_duration_s +
///      separation_transient_s`): piston-engine thrust at a much
///      lower acceleration than the boost burn.
///
/// **Public-proxy parameters (NOT measured-truth):**
///   - `peak_acceleration_mps2` ≈ 20–30 m/s² (~2–3g) per Sutton/Biblarz
///     typical small SRM impulse.
///   - `burn_duration_s` ≈ 1–2 s per dossier "boost-out" discussion.
///   - `sustain_acceleration_mps2` ≈ 0.5 m/s² (piston engine,
///     post-boost cruise build-up).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BoostThrustProfile {
    pub peak_acceleration_mps2: f64,
    pub burn_duration_s: f64,
    pub separation_transient_s: f64,
    pub sustain_acceleration_mps2: f64,
}

impl BoostThrustProfile {
    /// Shahed-class default per the dossier (1–2g boost over 1.5 s +
    /// piston). The peak value sits comfortably inside the
    /// `boost_kinematic_gate` accel band of 5–20 m/s² so the
    /// kinematic gate accepts a profile-driven launch.
    pub fn shahed_class_default() -> Self {
        Self {
            peak_acceleration_mps2: 18.0, // ~1.8g per dossier average
            burn_duration_s: 1.5,
            separation_transient_s: 0.2,
            sustain_acceleration_mps2: 0.5,
        }
    }

    /// Instantaneous acceleration (m/s²) at time `t_s` (seconds since
    /// launch). Uses a smooth ramp-plateau-drop-sustain shape per
    /// Sutton/Biblarz fig. 12-7 typical progressive-grain SRM:
    ///   * first 30 % of the burn → smoothstep ramp from 0 → peak;
    ///   * remaining 70 % of the burn → hold at peak;
    ///   * separation transient → linear drop from peak with a small
    ///     deceleration spike at the discontinuity;
    ///   * sustain → piston engine constant.
    pub fn acceleration_at(&self, t_s: f64) -> f64 {
        if t_s < 0.0 {
            return 0.0;
        }
        if t_s < self.burn_duration_s {
            // Boost phase: smooth ramp-up (smoothstep) to peak, then
            // plateau. Ramp fraction = 0.3 of burn duration.
            let phase = t_s / self.burn_duration_s;
            let shape = if phase < 0.3 {
                let p = phase / 0.3;
                p * p * (3.0 - 2.0 * p) // smoothstep
            } else {
                1.0
            };
            self.peak_acceleration_mps2 * shape
        } else if t_s < self.burn_duration_s + self.separation_transient_s {
            // Separation transient: rapid drop with a small deceleration
            // spike capturing the brief negative-acceleration impulse
            // as the spent booster physically decouples.
            let phase = (t_s - self.burn_duration_s) / self.separation_transient_s;
            let drop = 1.0 - phase; // linear from 1.0 at t=burn_end → 0.0 at t=burn_end+transient
            self.peak_acceleration_mps2 * drop * 0.3 - 1.0
        } else {
            // Post-separation sustain: piston engine thrust.
            self.sustain_acceleration_mps2
        }
    }

    /// Velocity (m/s) reached at time `t_s`, starting from a given
    /// rail-exit velocity. Numerically integrates `acceleration_at`
    /// over 10 ms steps because an analytical closed form would
    /// require splitting at the burn / separation boundaries and the
    /// smoothstep is not a clean integral.
    pub fn velocity_at(&self, t_s: f64, rail_exit_velocity_mps: f64) -> f64 {
        let dt = 0.01;
        let mut v = rail_exit_velocity_mps;
        let mut t = 0.0;
        while t < t_s {
            v += self.acceleration_at(t) * dt;
            t += dt;
        }
        v
    }
}

impl Default for BoostThrustProfile {
    fn default() -> Self {
        Self::shahed_class_default()
    }
}

/// Wave 4.5 Lane H5 — sub-phase of the boost transient, classified
/// from the measured instantaneous acceleration. Reviewers need this
/// extra resolution because the booster-burn / separation /
/// post-separation-sustain are physically very different signatures
/// (different micro-Doppler, different thermal envelope, etc.) and
/// folding them under one "Tier 1" was hiding model error in the
/// pre-H5 baseline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoostSubState {
    /// During the high-thrust SRM burn — measured |accel| in the
    /// boost-burn band (≈ 10–30 m/s² covering both the smoothstep
    /// ramp and the plateau).
    BoostBurn,
    /// Booster decoupling — measured accel between roughly −2 and
    /// +5 m/s² with a characteristic deceleration spike at the
    /// physical separation event.
    SeparationTransient,
    /// Piston-engine phase — measured |accel| below ~1 m/s², the
    /// post-boost sustained-thrust regime.
    PostSeparationSustain,
}

/// Outcome of a single-CPI Tier 1 evaluation. The detector publishes
/// `horizon_blocked` independently of `detected` so reviewers can read
/// "saw nothing because the horizon ate it" vs "saw nothing because the
/// kinematic envelope did not match".
///
/// **Wave 4.5 Lane H5 fields:** `sub_state` resolves the Tier 1
/// classification into one of the three SRM thrust sub-phases
/// (`BoostSubState`), and `instantaneous_acceleration_mps2` carries
/// the finite-difference accel value used to classify it. These let
/// downstream consumers tell a "boost burn" vs a "post-separation
/// sustain" CPI apart without re-deriving the value.
#[derive(Debug, Clone, PartialEq)]
pub struct BoostDecision {
    pub detected: bool,
    pub horizon_blocked: bool,
    pub mof_n_ratio: f32,
    pub min_target_altitude_for_los_m: f64,
    pub propagation_factor_magnitude: Option<f64>,
    pub note: &'static str,
    /// Wave 4.5 Lane H5 — sub-phase classification of the current CPI.
    pub sub_state: BoostSubState,
    /// Wave 4.5 Lane H5 — finite-difference acceleration (m/s²) on
    /// which the sub-state classification is based. `0.0` when there
    /// is insufficient history to compute it.
    pub instantaneous_acceleration_mps2: f64,
}

impl BoostDecision {
    pub fn empty() -> Self {
        Self {
            detected: false,
            horizon_blocked: false,
            mof_n_ratio: 0.0,
            min_target_altitude_for_los_m: 0.0,
            propagation_factor_magnitude: None,
            note: "",
            sub_state: BoostSubState::PostSeparationSustain,
            instantaneous_acceleration_mps2: 0.0,
        }
    }
}

/// Classify a measured instantaneous acceleration (m/s², signed —
/// positive = thrusting, negative = decelerating) into the matching
/// SRM thrust sub-state. Public-proxy thresholds per the dossier +
/// Sutton/Biblarz typical small-RATO impulse:
///
/// | Range (m/s²) | Sub-state |
/// |---|---|
/// | 10 ≤ a ≤ 30 | `BoostBurn` |
/// | 0 ≤ a < 1 | `PostSeparationSustain` |
/// | −2 ≤ a < 5 (else) | `SeparationTransient` |
/// | other | `PostSeparationSustain` |
///
/// The `BoostBurn` band overlaps the `boost_kinematic_gate` accel
/// band of 5–20 m/s² on its lower edge; the H5 classifier accepts
/// up to 30 m/s² (≈ 3g) to admit the upper-bound public-proxy peak
/// per Sutton/Biblarz typical small SRM. `PostSeparationSustain` is
/// the small positive piston-engine band (0–1 m/s²) corresponding to
/// the dossier's post-boost cruise build-up. The `SeparationTransient`
/// band captures both the deceleration spike at the physical decoupling
/// event (a ≈ −1 m/s²) and the brief positive-but-sub-burn window as
/// the SRM thrust falls below burn levels.
pub fn classify_boost_sub_state(accel_mps2: f64) -> BoostSubState {
    if (10.0..=30.0).contains(&accel_mps2) {
        BoostSubState::BoostBurn
    } else if (0.0..1.0).contains(&accel_mps2) {
        // Small-positive piston sustain band: the dossier's typical
        // post-boost piston accel (~0.5 m/s²) lives here.
        BoostSubState::PostSeparationSustain
    } else if (-2.0..5.0).contains(&accel_mps2) {
        // Deceleration spike (≈ −1 m/s²) or post-burn positive-but-
        // sub-burn window — both are the booster-separation signature.
        BoostSubState::SeparationTransient
    } else {
        // Anything else (deep negative, or super-burn thrust we cannot
        // attribute to this Tier 1 model) defaults to sustain; the
        // boost-gate's M-of-N rule already rejects out-of-band cases.
        BoostSubState::PostSeparationSustain
    }
}

/// Configuration for the Tier 1 boost-phase detector.
#[derive(Debug, Clone, Copy)]
pub struct BoostTierConfig {
    /// Carrier frequency (Hz) used in the two-ray propagation factor
    /// when the geometry is in the marginal LOS band.
    pub carrier_freq_hz: f64,
    /// Earth-radius scaling factor for refraction (4/3 = STANDARD_K_FACTOR
    /// for standard atmosphere). The boost LOS check uses this directly.
    pub k_factor: f64,
    /// Two-ray reflection-coefficient magnitude (0.5–1.0 typical for
    /// horizontal polarization at low grazing).
    pub reflection_coeff_magnitude: f64,
    /// Marginal-LOS band half-width (m). If
    /// `|target_alt - min_los| <= marginal_los_band_m`, the detector
    /// falls into the marginal escape rule rather than a strict
    /// sub-horizon block.
    pub marginal_los_band_m: f64,
    /// Sub-horizon depth (m): if `target_alt < min_los - sub_horizon_depth_m`,
    /// the target is firmly below the horizon → publish `horizon_blocked`.
    pub sub_horizon_depth_m: f64,
    /// First-null residual threshold on |F|² in the marginal LOS band.
    /// |F|² < this value publishes `horizon_blocked` with the
    /// "first-null residual" note.
    pub first_null_residual_pwr_threshold: f64,
    /// M-of-N denominator (window length in CPIs).
    pub mof_n_window: usize,
    /// M-of-N threshold M.
    pub mof_n_threshold: usize,
}

impl Default for BoostTierConfig {
    fn default() -> Self {
        // S-band defaults: 3 GHz carrier, standard 4/3-Earth refraction,
        // mid-grazing reflection (0.8) per Skolnik §2.10 conventions.
        Self {
            carrier_freq_hz: 3.0e9,
            k_factor: STANDARD_K_FACTOR,
            reflection_coeff_magnitude: 0.8,
            marginal_los_band_m: 50.0,
            sub_horizon_depth_m: 50.0,
            first_null_residual_pwr_threshold: 0.05,
            mof_n_window: 5,
            mof_n_threshold: 4,
        }
    }
}

/// Tier 1 boost-phase detector.
#[derive(Debug, Clone, Copy)]
pub struct BoostTierDetector {
    pub config: BoostTierConfig,
    pub gate: KinematicGate,
}

impl BoostTierDetector {
    pub fn new(config: BoostTierConfig) -> Self {
        Self {
            config,
            gate: boost_kinematic_gate(),
        }
    }

    pub fn with_default() -> Self {
        Self::new(BoostTierConfig::default())
    }

    /// Evaluate one CPI's worth of kinematic observation. Order of
    /// checks (each gates the next):
    ///
    /// 1. **LOS-horizon** (Skolnik §2.10 with 4/3-Earth refraction):
    ///    compute the minimum target altitude for LOS at the observation
    ///    range. If `target_alt < min_los - sub_horizon_depth_m`, return
    ///    `horizon_blocked: true, detected: false`.
    /// 2. **Marginal-LOS escape**: if `|target_alt - min_los| <=
    ///    marginal_los_band_m`, evaluate the two-ray propagation factor
    ///    |F|. If |F|² is below `first_null_residual_pwr_threshold`,
    ///    return `horizon_blocked: true` with the "first-null residual"
    ///    note.
    /// 3. **Boost gate**: check the boost kinematic envelope (speed
    ///    0–35 m/s, accel 5–20 m/s², altitude 0–200 m AGL). If fail,
    ///    return `detected: false`.
    /// 4. **M-of-N=4-of-5 over a trailing 5-CPI window**: count how
    ///    many CPIs in the trailing window have |dv/dt| in the boost
    ///    accel band [5, 20] m/s² (computed as finite differences of
    ///    consecutive samples). Return `detected: true` only if the
    ///    M-of-N count meets the threshold.
    ///
    /// **Wave 4.5 Lane H5 additions:** regardless of which branch the
    /// evaluation takes, the returned `BoostDecision` carries the
    /// signed finite-difference acceleration of the most recent
    /// sample pair (or `0.0` when the window has < 2 samples) and a
    /// `sub_state` classification per `classify_boost_sub_state`.
    pub fn evaluate(&self, obs: &KinematicObservation) -> BoostDecision {
        tier_boost_impl::evaluate_impl(self, obs)
    }
}

#[cfg(test)]
#[path = "tier_boost_tests.rs"]
mod tests;
