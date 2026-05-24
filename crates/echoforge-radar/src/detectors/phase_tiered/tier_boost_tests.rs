use super::super::kinematic_gate::KinematicSample;
use super::*;

fn boost_window(
    samples: Vec<(f64, f64, f64)>,
    range_m: f64,
    antenna_h: f64,
) -> KinematicObservation {
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
        vec![(0.0, 15.0, 50.0), (1.0, 20.0, 50.0), (2.0, 25.0, 50.0)],
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
    let min_los = min_target_altitude_for_los_m(20.0, 50_000.0, STANDARD_K_FACTOR);
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
    let min_los = min_target_altitude_for_los_m(20.0, 5_000.0, STANDARD_K_FACTOR);
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
    assert!(
        !dec.horizon_blocked,
        "above-horizon geometry must not block"
    );
}

#[test]
fn boost_rejects_when_kinematic_gate_fails_above_horizon() {
    // Above-horizon, but the kinematic gate must reject (speed
    // way above the boost cap of 35 m/s).
    let detector = BoostTierDetector::with_default();
    let alt = 250.0;
    let obs = boost_window(vec![(0.0, 60.0, alt), (1.0, 65.0, alt)], 2_000.0, 20.0);
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
    let obs_burn = boost_window(vec![(0.0, 0.0, alt), (1.0, 20.0, alt)], 2_000.0, 20.0);
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
    let obs_sep = boost_window(vec![(0.0, 30.0, alt), (1.0, 29.0, alt)], 2_000.0, 20.0);
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
    let obs_sus = boost_window(vec![(0.0, 30.0, alt), (1.0, 30.5, alt)], 2_000.0, 20.0);
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
