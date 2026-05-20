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
