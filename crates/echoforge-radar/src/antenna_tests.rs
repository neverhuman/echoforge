use super::*;

#[test]
fn isotropic_returns_zero_db_everywhere() {
    let antenna = IsotropicAntenna::new();
    assert_eq!(antenna.gain_db(0.0, 0.0), 0.0);
    assert_eq!(antenna.gain_db(45.0, -30.0), 0.0);
    assert_eq!(antenna.gain_db(-89.0, 89.0), 0.0);
    assert_eq!(antenna.gain_db(180.0, 0.0), 0.0);
}

#[test]
fn cosine_pattern_peaks_at_boresight() {
    let antenna = CosinePatternAntenna::new(12.0, 30.0);
    let on = antenna.gain_db(0.0, 0.0);
    let off_az = antenna.gain_db(5.0, 0.0);
    let off_el = antenna.gain_db(0.0, 5.0);
    assert!((on - 12.0).abs() < 1e-6);
    assert!(on > off_az);
    assert!(on > off_el);
}

#[test]
fn cosine_pattern_half_beamwidth_is_three_db_down() {
    let antenna = CosinePatternAntenna::new(10.0, 20.0);
    let half = antenna.gain_db(10.0, 0.0);
    assert!(
        (half - (10.0 - 3.0)).abs() < 0.2,
        "expected ~7 dB at half-beamwidth, got {half}"
    );
}

#[test]
fn cosine_pattern_is_very_low_at_back() {
    let antenna = CosinePatternAntenna::new(20.0, 10.0);
    let back = antenna.gain_db(90.0, 0.0);
    let side = antenna.gain_db(-90.0, 0.0);
    assert!(back <= -30.0);
    assert!(side <= -30.0);
}

#[test]
fn table_lookup_interpolates_between_entries() {
    let samples = vec![
        (-10.0, 0.0, 0.0),
        (10.0, 0.0, 20.0),
        (-10.0, 5.0, 10.0),
        (10.0, 5.0, 30.0),
    ];
    let antenna = TableLookupAntenna::new(samples);
    let mid_az = antenna.gain_db(0.0, 0.0);
    let mid_el = antenna.gain_db(-10.0, 2.5);
    let center = antenna.gain_db(0.0, 2.5);
    assert!((mid_az - 10.0).abs() < 1e-6);
    assert!((mid_el - 5.0).abs() < 1e-6);
    assert!((center - 15.0).abs() < 1e-6);
}

#[test]
fn table_lookup_returns_exact_values_at_grid_points() {
    let samples = vec![
        (0.0, 0.0, 5.0),
        (10.0, 0.0, 7.0),
        (0.0, 10.0, 3.0),
        (10.0, 10.0, 9.0),
    ];
    let antenna = TableLookupAntenna::new(samples);
    assert!((antenna.gain_db(0.0, 0.0) - 5.0).abs() < 1e-6);
    assert!((antenna.gain_db(10.0, 0.0) - 7.0).abs() < 1e-6);
    assert!((antenna.gain_db(0.0, 10.0) - 3.0).abs() < 1e-6);
    assert!((antenna.gain_db(10.0, 10.0) - 9.0).abs() < 1e-6);
}

#[test]
fn table_lookup_clamps_to_grid_extents() {
    let samples = vec![
        (-5.0, 0.0, 1.0),
        (5.0, 0.0, 3.0),
        (-5.0, 1.0, 1.0),
        (5.0, 1.0, 3.0),
    ];
    let antenna = TableLookupAntenna::new(samples);
    assert!((antenna.gain_db(-100.0, -100.0) - 1.0).abs() < 1e-6);
    assert!((antenna.gain_db(100.0, 100.0) - 3.0).abs() < 1e-6);
}

#[test]
fn phased_array_main_beam_at_steering_direction() {
    let antenna = PhasedArrayManifold::new(16, 0.015, 10_000_000_000.0, 0.0, 0.0);
    let on = antenna.gain_db(0.0, 0.0);
    let off = antenna.gain_db(30.0, 0.0);
    assert!(on > off + 10.0, "on={on}, off={off}");
    assert!(on.abs() < 1e-3, "boresight should be ~0 dB, got {on}");
}

#[test]
fn phased_array_sidelobes_below_minus_ten_db() {
    let antenna = PhasedArrayManifold::new(32, 0.015, 10_000_000_000.0, 0.0, 0.0);
    // Sample the response away from the main lobe and confirm we are
    // well below the boresight value. A cosine-tapered array beats the
    // −13 dB uniform sidelobe by a healthy margin.
    let mut peak_sidelobe_db = f64::NEG_INFINITY;
    for az in (10..=80).step_by(2) {
        let g = antenna.gain_db(az as f64, 0.0);
        if g > peak_sidelobe_db {
            peak_sidelobe_db = g;
        }
    }
    assert!(
        peak_sidelobe_db < -10.0,
        "peak sidelobe {peak_sidelobe_db} dB exceeds threshold"
    );
}

#[test]
fn phased_array_steering_shifts_main_beam() {
    let antenna = PhasedArrayManifold::new(16, 0.015, 10_000_000_000.0, 20.0, 0.0);
    let at_steer = antenna.gain_db(20.0, 0.0);
    let at_zero = antenna.gain_db(0.0, 0.0);
    assert!(at_steer > at_zero + 5.0);
    assert!(
        at_steer.abs() < 1e-3,
        "main beam should be ~0 dB, got {at_steer}"
    );
}

#[test]
fn phased_array_handles_single_element_gracefully() {
    let antenna = PhasedArrayManifold::new(1, 0.015, 10_000_000_000.0, 0.0, 0.0);
    assert_eq!(antenna.gain_db(0.0, 0.0), 0.0);
    assert_eq!(antenna.gain_db(45.0, 0.0), 0.0);
}
