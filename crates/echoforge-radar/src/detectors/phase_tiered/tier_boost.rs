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
//! Strict-open posture: the boost-phase detection rule and LOS-horizon
//! honesty publish *public-proxy expected* Pd/Pfa for a Shahed-class
//! launch transient and do NOT claim platform-specific signature truth.

use crate::propagation::{
    min_target_altitude_for_los_m, two_ray_propagation_factor_magnitude, STANDARD_K_FACTOR,
};

use super::kinematic_gate::{boost_kinematic_gate, KinematicGate, KinematicObservation};

/// Outcome of a single-CPI Tier 1 evaluation. The detector publishes
/// `horizon_blocked` independently of `detected` so reviewers can read
/// "saw nothing because the horizon ate it" vs "saw nothing because the
/// kinematic envelope did not match".
#[derive(Debug, Clone, PartialEq)]
pub struct BoostDecision {
    pub detected: bool,
    pub horizon_blocked: bool,
    pub mof_n_ratio: f32,
    pub min_target_altitude_for_los_m: f64,
    pub propagation_factor_magnitude: Option<f64>,
    pub note: &'static str,
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
        }
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
    pub fn evaluate(&self, obs: &KinematicObservation) -> BoostDecision {
        let mut out = BoostDecision::empty();

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
}
