//! `BoostTierDetector::evaluate` implementation — extracted for LOC compliance.

use super::{classify_boost_sub_state, BoostDecision, BoostTierDetector};
use crate::detectors::phase_tiered::kinematic_gate::KinematicObservation;
use crate::propagation::{min_target_altitude_for_los_m, two_ray_propagation_factor_magnitude};

/// Core evaluate implementation. Called from `BoostTierDetector::evaluate`.
pub(super) fn evaluate_impl(
    detector: &BoostTierDetector,
    obs: &KinematicObservation,
) -> BoostDecision {
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
        detector.config.k_factor,
    );
    out.min_target_altitude_for_los_m = min_los;

    if target_alt < min_los - detector.config.sub_horizon_depth_m {
        out.horizon_blocked = true;
        out.note = "sub-horizon";
        return out;
    }

    // (2) Marginal-LOS escape via two-ray propagation factor.
    if (target_alt - min_los).abs() <= detector.config.marginal_los_band_m {
        let f_mag = two_ray_propagation_factor_magnitude(
            detector.config.carrier_freq_hz,
            target_alt.max(0.0),
            obs.radar_altitude_agl_m,
            obs.range_m,
            detector.config.reflection_coeff_magnitude,
        );
        let f_pwr = f_mag * f_mag;
        out.propagation_factor_magnitude = Some(f_mag);
        if f_pwr < detector.config.first_null_residual_pwr_threshold {
            out.horizon_blocked = true;
            out.note = "first-null residual";
            return out;
        }
    }

    // (3) Boost-gate kinematic check on the most recent sample.
    if !detector.gate.accepts(obs) {
        out.note = "boost gate not satisfied";
        return out;
    }

    // (4) M-of-N over a trailing 5-CPI window of finite-difference
    // accelerations.
    let win = detector.config.mof_n_window;
    let m_thresh = detector.config.mof_n_threshold;
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
        if accel >= detector.gate.accel_mps2_min && accel <= detector.gate.accel_mps2_max {
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
