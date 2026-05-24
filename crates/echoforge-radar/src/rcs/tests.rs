use super::*;

fn small_grid_table() -> RcsLookup {
    RcsLookup {
        target_class: "test".to_string(),
        frequency_ghz: 10.0,
        polarization: Polarization::Vv,
        aspect_grid: AspectGrid {
            azimuth_deg: vec![0.0, 90.0, 180.0],
            elevation_deg: vec![0.0, 30.0],
        },
        // Row-major [az0_el0, az0_el30, az90_el0, az90_el30, az180_el0, az180_el30].
        rcs_dbsm: vec![-20.0, -22.0, -10.0, -12.0, -20.0, -22.0],
        fluctuation: SwerlingModel::Swerling0,
        citation: "synthetic fixture".to_string(),
        citation_url: None,
    }
}

fn assert_close(a: f64, b: f64, eps: f64) {
    assert!((a - b).abs() < eps, "expected {a} ~= {b} (tolerance {eps})");
}

#[test]
fn evaluate_static_returns_grid_value_at_grid_point() {
    let mut rcs = Rcs::empty();
    rcs.add_table(small_grid_table());
    let v = rcs.evaluate_static("test", 0.0, 0.0, 10.0, Polarization::Vv);
    assert_close(v, -20.0, 1e-12);
    let v2 = rcs.evaluate_static("test", 90.0, 30.0, 10.0, Polarization::Vv);
    assert_close(v2, -12.0, 1e-12);
}

#[test]
fn bilinear_interp_midway_between_grid_points_is_mid_value() {
    let mut rcs = Rcs::empty();
    rcs.add_table(small_grid_table());
    // Midway in azimuth (45 between 0 and 90) at elevation 0: (-20 + -10) / 2 = -15.
    let v = rcs.evaluate_static("test", 45.0, 0.0, 10.0, Polarization::Vv);
    assert_close(v, -15.0, 1e-12);
    // Midway in elevation (15 between 0 and 30) at azimuth 0: (-20 + -22) / 2 = -21.
    let v2 = rcs.evaluate_static("test", 0.0, 15.0, 10.0, Polarization::Vv);
    assert_close(v2, -21.0, 1e-12);
    // Center of (0..90, 0..30) cell: average of four corners (-20,-22,-10,-12) = -16.
    let v3 = rcs.evaluate_static("test", 45.0, 15.0, 10.0, Polarization::Vv);
    assert_close(v3, -16.0, 1e-12);
}

#[test]
fn polarization_exact_match_returns_that_table() {
    let mut rcs = Rcs::empty();
    let mut vv = small_grid_table();
    vv.polarization = Polarization::Vv;
    vv.rcs_dbsm = vec![-20.0, -20.0, -20.0, -20.0, -20.0, -20.0];
    let mut hh = small_grid_table();
    hh.polarization = Polarization::Hh;
    hh.rcs_dbsm = vec![-30.0, -30.0, -30.0, -30.0, -30.0, -30.0];
    rcs.add_table(vv);
    rcs.add_table(hh);
    let v_vv = rcs.evaluate_static("test", 45.0, 15.0, 10.0, Polarization::Vv);
    assert_close(v_vv, -20.0, 1e-12);
    let v_hh = rcs.evaluate_static("test", 45.0, 15.0, 10.0, Polarization::Hh);
    assert_close(v_hh, -30.0, 1e-12);
}

#[test]
fn polarization_miss_returns_averaged_value() {
    let mut rcs = Rcs::empty();
    let mut vv = small_grid_table();
    vv.polarization = Polarization::Vv;
    vv.rcs_dbsm = vec![-20.0; 6];
    let mut hh = small_grid_table();
    hh.polarization = Polarization::Hh;
    hh.rcs_dbsm = vec![-30.0; 6];
    rcs.add_table(vv);
    rcs.add_table(hh);
    // Request Hv — neither table has it, fall back to mean(-20, -30) = -25.
    let v = rcs.evaluate_static("test", 45.0, 15.0, 10.0, Polarization::Hv);
    assert_close(v, -25.0, 1e-12);
}

#[test]
fn frequency_nearest_in_log_space_wins() {
    let mut rcs = Rcs::empty();
    let mut t1 = small_grid_table();
    t1.frequency_ghz = 1.0;
    t1.rcs_dbsm = vec![-10.0; 6];
    let mut t10 = small_grid_table();
    t10.frequency_ghz = 10.0;
    t10.rcs_dbsm = vec![-20.0; 6];
    let mut t100 = small_grid_table();
    t100.frequency_ghz = 100.0;
    t100.rcs_dbsm = vec![-30.0; 6];
    rcs.add_table(t1);
    rcs.add_table(t10);
    rcs.add_table(t100);
    // sqrt(10*100)=31.6 GHz → closer to 100 GHz in log space.
    let v = rcs.evaluate_static("test", 0.0, 0.0, 31.7, Polarization::Vv);
    assert_close(v, -30.0, 1e-12);
    let v2 = rcs.evaluate_static("test", 0.0, 0.0, 9.0, Polarization::Vv);
    assert_close(v2, -20.0, 1e-12);
}

#[test]
fn swerling0_returns_deterministic_value() {
    let mut rcs = Rcs::empty();
    rcs.add_table(small_grid_table());
    let median = rcs.evaluate_static("test", 30.0, 10.0, 10.0, Polarization::Vv);
    let a = rcs.evaluate("test", 30.0, 10.0, 10.0, Polarization::Vv, 12345, 0);
    let b = rcs.evaluate("test", 30.0, 10.0, 10.0, Polarization::Vv, 12345, 17);
    let c = rcs.evaluate("test", 30.0, 10.0, 10.0, Polarization::Vv, 99, 9999);
    assert_close(a, median, 1e-12);
    assert_close(b, median, 1e-12);
    assert_close(c, median, 1e-12);
}

#[test]
fn swerling1_correlated_within_scan_decorrelated_across_scans() {
    let mut table = small_grid_table();
    table.fluctuation = SwerlingModel::Swerling1;
    let mut rcs = Rcs::empty();
    rcs.add_table(table);
    let same_scan_a = rcs.evaluate("test", 30.0, 10.0, 10.0, Polarization::Vv, 7, 0);
    let same_scan_b = rcs.evaluate("test", 30.0, 10.0, 10.0, Polarization::Vv, 7, 5);
    assert_close(same_scan_a, same_scan_b, 1e-12);
    let next_scan = rcs.evaluate(
        "test",
        30.0,
        10.0,
        10.0,
        Polarization::Vv,
        7,
        SWERLING_DEFAULT_SCAN_SIZE,
    );
    assert!(
        (same_scan_a - next_scan).abs() > 1e-6,
        "expected Swerling 1 to decorrelate across scans (got {same_scan_a} vs {next_scan})",
    );
}

#[test]
fn swerling2_decorrelates_pulse_to_pulse() {
    let mut table = small_grid_table();
    table.fluctuation = SwerlingModel::Swerling2;
    let mut rcs = Rcs::empty();
    rcs.add_table(table);
    let p0 = rcs.evaluate("test", 30.0, 10.0, 10.0, Polarization::Vv, 99, 0);
    let p1 = rcs.evaluate("test", 30.0, 10.0, 10.0, Polarization::Vv, 99, 1);
    let p2 = rcs.evaluate("test", 30.0, 10.0, 10.0, Polarization::Vv, 99, 2);
    assert!(
        (p0 - p1).abs() > 1e-6 && (p1 - p2).abs() > 1e-6,
        "Swerling 2 should decorrelate pulse-to-pulse (got {p0}, {p1}, {p2})",
    );
}

#[test]
fn determinism_same_call_same_value() {
    let mut table = small_grid_table();
    table.fluctuation = SwerlingModel::Swerling4;
    let mut rcs = Rcs::empty();
    rcs.add_table(table);
    let a = rcs.evaluate("test", 30.0, 10.0, 10.0, Polarization::Vv, 0xC0FFEE, 9);
    let b = rcs.evaluate("test", 30.0, 10.0, 10.0, Polarization::Vv, 0xC0FFEE, 9);
    assert_eq!(a.to_bits(), b.to_bits());
}

#[test]
fn aspect_wraps_around_360() {
    let mut rcs = Rcs::empty();
    rcs.add_table(small_grid_table());
    let a0 = rcs.evaluate_static("test", 0.0, 0.0, 10.0, Polarization::Vv);
    let a360 = rcs.evaluate_static("test", 360.0, 0.0, 10.0, Polarization::Vv);
    let a720 = rcs.evaluate_static("test", 720.0, 0.0, 10.0, Polarization::Vv);
    let a_neg = rcs.evaluate_static("test", -360.0, 0.0, 10.0, Polarization::Vv);
    assert_close(a0, a360, 1e-12);
    assert_close(a0, a720, 1e-12);
    assert_close(a0, a_neg, 1e-12);
}

#[test]
fn out_of_grid_elevation_clamps_to_nearest_edge() {
    let mut rcs = Rcs::empty();
    rcs.add_table(small_grid_table());
    let v_low = rcs.evaluate_static("test", 0.0, -45.0, 10.0, Polarization::Vv);
    let v_at_zero = rcs.evaluate_static("test", 0.0, 0.0, 10.0, Polarization::Vv);
    assert_close(v_low, v_at_zero, 1e-12);
    let v_hi = rcs.evaluate_static("test", 0.0, 90.0, 10.0, Polarization::Vv);
    let v_at_top = rcs.evaluate_static("test", 0.0, 30.0, 10.0, Polarization::Vv);
    assert_close(v_hi, v_at_top, 1e-12);
}

#[test]
fn seeded_public_proxy_v1_has_three_valid_cited_tables() {
    let rcs = Rcs::seeded_public_proxy_v1();
    assert!(
        rcs.tables.len() >= 3,
        "expected at least 3 reference tables"
    );
    let class_names: Vec<&str> = rcs.tables.iter().map(|t| t.target_class.as_str()).collect();
    for needed in ["fixed-wing-uas-small", "bird-large-single", "quadrotor"] {
        assert!(
            class_names.contains(&needed),
            "missing seeded table for {needed}"
        );
    }
    for t in &rcs.tables {
        assert!(t.is_valid(), "table {} is invalid", t.target_class);
        assert!(
            !t.citation.is_empty(),
            "citation must not be empty for {}",
            t.target_class
        );
        assert!(
            t.aspect_grid.azimuth_deg.len() >= 4,
            "aspect grid too coarse for {}",
            t.target_class
        );
        assert!(
            t.aspect_grid.elevation_deg.len() >= 2,
            "elevation grid too coarse for {}",
            t.target_class
        );
        for v in &t.rcs_dbsm {
            assert!(v.is_finite(), "non-finite RCS in {}", t.target_class);
        }
    }
}

#[test]
fn unknown_target_class_returns_neg_infinity() {
    let rcs = Rcs::seeded_public_proxy_v1();
    let v = rcs.evaluate_static("nonexistent", 0.0, 0.0, 10.0, Polarization::Vv);
    assert!(v.is_infinite() && v.is_sign_negative());
    let v2 = rcs.evaluate("nonexistent", 0.0, 0.0, 10.0, Polarization::Vv, 0, 0);
    assert!(v2.is_infinite() && v2.is_sign_negative());
}

#[test]
fn fluctuation_overlay_preserves_finite_dbsm() {
    let rcs = Rcs::seeded_public_proxy_v1();
    for pulse in 0..64 {
        let v = rcs.evaluate(
            "quadrotor",
            42.5,
            5.0,
            10.0,
            Polarization::Vv,
            0xDEAD_BEEF,
            pulse,
        );
        assert!(v.is_finite(), "non-finite dBsm at pulse {pulse}");
    }
}
