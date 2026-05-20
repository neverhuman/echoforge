use super::*;

fn obs_single(speed: f64, alt: f64) -> KinematicObservation {
    KinematicObservation::new(vec![KinematicSample::new(0.0, speed, alt)], 10_000.0, 20.0)
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
    let gate = boost_kinematic_gate();
    let obs = obs_pair(3.0, 80.0, 15.0, 80.0, 1.0);
    assert!(gate.accepts(&obs), "boost gate must accept canonical");
}

#[test]
fn boost_gate_rejects_too_high() {
    let gate = boost_kinematic_gate();
    let obs = obs_pair(3.0, 5000.0, 15.0, 5000.0, 1.0);
    assert!(
        !gate.accepts(&obs),
        "boost gate must reject 5000 m altitude"
    );
}

#[test]
fn climb_gate_accepts_canonical() {
    let gate = climb_kinematic_gate();
    let obs = obs_pair(39.0, 250.0, 40.0, 250.0, 1.0);
    assert!(gate.accepts(&obs), "climb gate must accept canonical");
}

#[test]
fn climb_gate_rejects_aircraft_speed() {
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
    let obs = obs_pair(30.0, 100.0, 40.0, 106.0, 2.0);
    let a = obs.current_acceleration_mps2().unwrap();
    let c = obs.altitude_rate_mps().unwrap();
    assert!((a - 5.0).abs() < 1e-9, "accel = {a}");
    assert!((c - 3.0).abs() < 1e-9, "climb_rate = {c}");
}
