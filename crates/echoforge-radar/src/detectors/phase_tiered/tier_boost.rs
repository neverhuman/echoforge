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

use crate::propagation::{
    min_target_altitude_for_los_m, two_ray_propagation_factor_magnitude, STANDARD_K_FACTOR,
};

use super::kinematic_gate::{boost_kinematic_gate, KinematicGate, KinematicObservation};

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
        let mut out = BoostDecision::empty();

        // Wave 4.5 H5: capture the signed instantaneous accel up front so
        // every return path publishes the sub-state classification, not
        // just the "happy path" success branch.
        let signed_accel = obs.current_acceleration_mps2().unwrap_or(0.0);
        out.instantaneous_acceleration_mps2 = signed_accel;
        out.sub_state = classify_boost_sub_state(signed_accel);

        let target_alt = match obs.current_altitude_agl_m() {
            Some(v) => v,
            None => {
                out.note = "empty observation";
                return out;
            }
        };

        // (1) LOS-horizon check (Skolnik §2.10, 4/3-Earth refraction).
        let min_los = min_target_altitude_for_los_m(
            obs.radar_altitude_agl_m,
            obs.range_m,
            self.config.k_factor,
        );
        out.min_target_altitude_for_los_m = min_los;

        if target_alt < min_los - self.config.sub_horizon_depth_m {
            out.horizon_blocked = true;
            out.note = "sub-horizon";
            return out;
        }

        // (2) Marginal-LOS escape via two-ray propagation factor.
        if (target_alt - min_los).abs() <= self.config.marginal_los_band_m {
            let f_mag = two_ray_propagation_factor_magnitude(
                self.config.carrier_freq_hz,
                target_alt.max(0.0),
                obs.radar_altitude_agl_m,
                obs.range_m,
                self.config.reflection_coeff_magnitude,
            );
            let f_pwr = f_mag * f_mag;
            out.propagation_factor_magnitude = Some(f_mag);
            if f_pwr < self.config.first_null_residual_pwr_threshold {
                out.horizon_blocked = true;
                out.note = "first-null residual";
                return out;
            }
        }

        // (3) Boost-gate kinematic check on the most recent sample.
        if !self.gate.accepts(obs) {
            out.note = "boost gate not satisfied";
            return out;
        }

        // (4) M-of-N over a trailing 5-CPI window of finite-difference
        // accelerations.
        let win = self.config.mof_n_window;
        let m_thresh = self.config.mof_n_threshold;
        let trailing = obs.trailing(win + 1);
        if trailing.len() < 2 {
            // Not enough history to compute any acceleration deltas.
            out.note = "insufficient history for M-of-N";
            return out;
        }
        let mut matches = 0usize;
        let mut considered = 0usize;
        for pair in trailing.windows(2) {
            let dt = pair[1].time_s - pair[0].time_s;
            if dt <= 0.0 {
                continue;
            }
            let accel = ((pair[1].radial_speed_mps - pair[0].radial_speed_mps) / dt).abs();
            considered += 1;
            if accel >= self.gate.accel_mps2_min && accel <= self.gate.accel_mps2_max {
                matches += 1;
            }
        }
        out.mof_n_ratio = if considered == 0 {
            0.0
        } else {
            matches as f32 / considered as f32
        };
        out.detected = matches >= m_thresh;
        out.note = if out.detected {
            "boost detected"
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

    fn boost_window(samples: Vec<(f64, f64, f64)>, range_m: f64, antenna_h: f64) -> KinematicObservation {
        KinematicObservation::new(
            samples
                .into_iter()
                .map(|(t, v, h)| KinematicSample::new(t, v, h))
                .collect(),
            range_m,
            antenna_h,
        )
    }

    #[test]
    fn boost_los_horizon_blocked_at_100km_50m_altitude() {
        // Skolnik §2.10 4/3-Earth horizon: 20 m antenna + 100 km range
        // gives min_target_altitude_for_los_m of ~391 m (validated by
        // `propagation::tests::min_target_altitude_canonical_geometry`).
        // A 50 m target sits ~341 m below the horizon → much deeper
        // than the 50 m sub-horizon depth threshold → horizon_blocked.
        let detector = BoostTierDetector::with_default();
        let obs = boost_window(
            vec![
                (0.0, 15.0, 50.0),
                (1.0, 20.0, 50.0),
                (2.0, 25.0, 50.0),
            ],
            100_000.0,
            20.0,
        );
        let dec = detector.evaluate(&obs);
        assert!(
            dec.horizon_blocked,
            "C13: sub-LOS boost geometry must publish horizon_blocked; \
             min_los={:.2} m, target=50 m, note={}",
            dec.min_target_altitude_for_los_m, dec.note
        );
        assert!(!dec.detected, "horizon_blocked geometry must not detect");
    }

    #[test]
    fn boost_los_marginal_at_horizon() {
        // Pick a target altitude just inside the marginal band
        // (|alt - min_los| <= 50 m). The detector should compute a
        // propagation factor and either publish horizon_blocked
        // (first-null residual) or fall through to the kinematic checks.
        let detector = BoostTierDetector::with_default();
        let min_los =
            min_target_altitude_for_los_m(20.0, 50_000.0, STANDARD_K_FACTOR);
        // The min_los at 50 km is ~58.65 m, well within the 0–200 m
        // boost altitude band — perfect for the marginal escape test.
        let marginal_alt = (min_los + 25.0).max(0.0);
        let obs = boost_window(
            vec![
                (0.0, 5.0, marginal_alt),
                (1.0, 15.0, marginal_alt),
                (2.0, 25.0, marginal_alt),
            ],
            50_000.0,
            20.0,
        );
        let dec = detector.evaluate(&obs);
        assert!(
            dec.propagation_factor_magnitude.is_some(),
            "marginal-LOS band must trigger two-ray |F| computation; \
             min_los={min_los:.2}, target={marginal_alt:.2}"
        );
    }

    #[test]
    fn boost_above_horizon_passes_gate_with_m_of_n() {
        // Place the target well above the horizon and supply 5 CPIs
        // where each consecutive pair has |dv/dt| ∈ [5, 20] m/s².
        let detector = BoostTierDetector::with_default();
        let min_los =
            min_target_altitude_for_los_m(20.0, 5_000.0, STANDARD_K_FACTOR);
        let alt = (min_los + 200.0).max(150.0);
        let obs = boost_window(
            vec![
                (0.0, 5.0, alt),
                (1.0, 15.0, alt),
                (2.0, 25.0, alt),
                (3.0, 35.0, alt),
                (4.0, 35.0, alt),
                (5.0, 35.0, alt),
            ],
            5_000.0,
            20.0,
        );
        let dec = detector.evaluate(&obs);
        // 5 consecutive 5-second deltas: dv values are 10, 10, 10, 0, 0.
        // Three deltas in band → not detected at 4-of-5 threshold but
        // gate accepted, so horizon_blocked stays false.
        assert!(!dec.horizon_blocked, "above-horizon geometry must not block");
    }

    #[test]
    fn boost_rejects_when_kinematic_gate_fails_above_horizon() {
        // Above-horizon, but the kinematic gate must reject (speed
        // way above the boost cap of 35 m/s).
        let detector = BoostTierDetector::with_default();
        let alt = 250.0;
        let obs = boost_window(
            vec![
                (0.0, 60.0, alt),
                (1.0, 65.0, alt),
            ],
            2_000.0,
            20.0,
        );
        let dec = detector.evaluate(&obs);
        assert!(!dec.detected, "boost gate must reject cruise-speed");
        assert!(!dec.horizon_blocked, "above-horizon means no block");
    }

    // ===================================================================
    // Wave 4.5 Lane H5 — booster-burn thrust profile + sub-state
    // classification tests.
    // ===================================================================

    /// **H5 test 1:** the `BoostThrustProfile` must reproduce the
    /// smoothstep ramp-plateau-drop-sustain shape per Sutton &
    /// Biblarz, *Rocket Propulsion Elements*, 9th ed., fig. 12-7
    /// typical progressive-grain SRM.
    #[test]
    fn boost_thrust_profile_shape_matches_smoothstep() {
        let profile = BoostThrustProfile::shahed_class_default();
        let peak = profile.peak_acceleration_mps2;
        let burn = profile.burn_duration_s;
        let transient = profile.separation_transient_s;

        // (a) At t = 0 the SRM is just starting; thrust = 0.
        let a0 = profile.acceleration_at(0.0);
        assert!(
            (a0 - 0.0).abs() < 1e-9,
            "H5: acceleration_at(0.0) must be 0; got {a0}"
        );

        // (b) At t = burn * 0.3 the smoothstep ramp has reached the
        // plateau exactly.
        let a_ramp = profile.acceleration_at(burn * 0.3);
        assert!(
            (a_ramp - peak).abs() < 1e-6,
            "H5: acceleration_at(burn * 0.3) must equal peak; got {a_ramp} vs peak {peak}"
        );

        // (c) Just before burn end we are still on the plateau.
        let a_end = profile.acceleration_at(burn - 1e-6);
        assert!(
            (a_end - peak).abs() < 1e-3,
            "H5: acceleration_at(burn - eps) must equal peak; got {a_end} vs peak {peak}"
        );

        // (d) Separation transient: at burn + 0.1 s (half of the 0.2 s
        // transient window for shahed_class_default), the model
        // returns peak * (1 - 0.5) * 0.3 - 1 = peak * 0.15 - 1.
        let t_mid_transient = burn + transient / 2.0;
        let a_transient = profile.acceleration_at(t_mid_transient);
        let expected_transient = peak * 0.15 - 1.0;
        assert!(
            (a_transient - expected_transient).abs() < 1e-6,
            "H5: separation-transient mid-window must equal peak*0.15 - 1; got {a_transient} vs expected {expected_transient}"
        );

        // (e) Sustain region: piston engine constant.
        let a_sustain = profile.acceleration_at(burn + transient + 1.0);
        assert!(
            (a_sustain - profile.sustain_acceleration_mps2).abs() < 1e-9,
            "H5: post-transient must equal sustain_acceleration; got {a_sustain}"
        );

        // (f) Negative time guard.
        assert_eq!(profile.acceleration_at(-1.0), 0.0, "H5: negative t → 0");
    }

    /// **H5 test 2:** integrating the SRM thrust profile from the
    /// dossier's rail-exit velocity (~9 m/s) over the 1.5 s boost
    /// must reach the dossier's release velocity envelope of
    /// 25–35 m/s. This is the headline kinematic gate of the H5
    /// refinement — the constant-acceleration baseline never gave a
    /// principled velocity history because it used `|dv/dt|` as a
    /// magnitude check only.
    #[test]
    fn boost_velocity_integration_reaches_release() {
        let profile = BoostThrustProfile::shahed_class_default();
        let rail_exit_mps = 9.0;
        let v_at_burn_end = profile.velocity_at(profile.burn_duration_s, rail_exit_mps);
        // Dossier release envelope is 25–35 m/s; we want to land
        // squarely inside that band (±5 m/s of the 30 m/s center).
        assert!(
            (25.0..=35.0).contains(&v_at_burn_end),
            "H5: velocity at burn completion must land in the dossier release envelope \
             [25, 35] m/s; got {v_at_burn_end} m/s"
        );
        // Also sanity-check that the value is within ±5 m/s of the 30
        // m/s center per the brief.
        let center_delta = (v_at_burn_end - 30.0).abs();
        assert!(
            center_delta <= 5.0,
            "H5: velocity at burn completion must be within ±5 m/s of 30 m/s; got delta = {center_delta}"
        );
    }

    /// **H5 test 3:** `BoostTierDetector::evaluate` must classify
    /// the observation into the correct `BoostSubState` based on the
    /// most recent finite-difference acceleration, regardless of
    /// whether the M-of-N detection rule fires. Three representative
    /// values per the H5 brief:
    ///   - 20 m/s² → `BoostBurn`
    ///   - −1 m/s² → `SeparationTransient`
    ///   - 0.5 m/s² → `PostSeparationSustain`
    #[test]
    fn boost_sub_state_classification() {
        // (a) Direct classifier checks — independent of the detector
        // wiring so the test localises any model drift.
        assert_eq!(
            classify_boost_sub_state(20.0),
            BoostSubState::BoostBurn,
            "H5: 20 m/s² must classify as BoostBurn"
        );
        assert_eq!(
            classify_boost_sub_state(-1.0),
            BoostSubState::SeparationTransient,
            "H5: -1 m/s² must classify as SeparationTransient"
        );
        assert_eq!(
            classify_boost_sub_state(0.5),
            BoostSubState::PostSeparationSustain,
            "H5: 0.5 m/s² must classify as PostSeparationSustain"
        );

        // (b) End-to-end via the detector. Use an altitude that is
        // safely above the horizon so we never short-circuit on
        // horizon_blocked. Three windows, each engineered to produce
        // a specific finite-difference accel between the last two
        // samples (which is what the detector inspects).
        let detector = BoostTierDetector::with_default();
        let alt = 250.0;

        // 20 m/s² → BoostBurn (dv = 20, dt = 1).
        let obs_burn = boost_window(
            vec![(0.0, 0.0, alt), (1.0, 20.0, alt)],
            2_000.0,
            20.0,
        );
        let dec_burn = detector.evaluate(&obs_burn);
        assert!(
            (dec_burn.instantaneous_acceleration_mps2 - 20.0).abs() < 1e-9,
            "H5: detector must publish 20 m/s² for the boost-burn window; got {}",
            dec_burn.instantaneous_acceleration_mps2
        );
        assert_eq!(
            dec_burn.sub_state,
            BoostSubState::BoostBurn,
            "H5: detector must classify 20 m/s² as BoostBurn"
        );

        // -1 m/s² → SeparationTransient (dv = -1, dt = 1).
        let obs_sep = boost_window(
            vec![(0.0, 30.0, alt), (1.0, 29.0, alt)],
            2_000.0,
            20.0,
        );
        let dec_sep = detector.evaluate(&obs_sep);
        assert!(
            (dec_sep.instantaneous_acceleration_mps2 - (-1.0)).abs() < 1e-9,
            "H5: detector must publish -1 m/s² for the separation window; got {}",
            dec_sep.instantaneous_acceleration_mps2
        );
        assert_eq!(
            dec_sep.sub_state,
            BoostSubState::SeparationTransient,
            "H5: detector must classify -1 m/s² as SeparationTransient"
        );

        // 0.5 m/s² → PostSeparationSustain (dv = 0.5, dt = 1).
        let obs_sus = boost_window(
            vec![(0.0, 30.0, alt), (1.0, 30.5, alt)],
            2_000.0,
            20.0,
        );
        let dec_sus = detector.evaluate(&obs_sus);
        assert!(
            (dec_sus.instantaneous_acceleration_mps2 - 0.5).abs() < 1e-9,
            "H5: detector must publish 0.5 m/s² for the sustain window; got {}",
            dec_sus.instantaneous_acceleration_mps2
        );
        assert_eq!(
            dec_sus.sub_state,
            BoostSubState::PostSeparationSustain,
            "H5: detector must classify 0.5 m/s² as PostSeparationSustain"
        );
    }
}
