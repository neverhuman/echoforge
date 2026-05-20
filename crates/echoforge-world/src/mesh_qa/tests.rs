use super::*;
use crate::mesh::{plate_mesh, sphere_mesh};

fn gate_named<'a>(report: &'a QaReport, name: &str) -> &'a QaGateResult {
    match report.gates.iter().find(|g| g.name == name) {
        Some(gate) => gate,
        None => panic!("missing gate '{}'", name),
    }
}

#[test]
fn sphere_mesh_finite_and_consistent() {
    let mesh = sphere_mesh(1.0, 32, 64);
    let report = run_qa(&mesh, Some(10e9));
    assert_eq!(report.mesh_primitive_id, "sphere");
    assert_eq!(report.triangle_count, mesh.triangle_count());
    assert!(report.bbox_max[0] > report.bbox_min[0]);

    assert_eq!(
        gate_named(&report, "gate_finite_vertices").status,
        QaStatus::Pass
    );
    assert_eq!(
        gate_named(&report, "gate_consistent_winding").status,
        QaStatus::Pass
    );
    assert_eq!(
        gate_named(&report, "gate_max_electrical_size").status,
        QaStatus::Pass
    );
    let w = gate_named(&report, "gate_watertight");
    assert!(
        matches!(w.status, QaStatus::Pass | QaStatus::Warn),
        "watertight gate must not Fail on closed sphere: {}",
        w.message
    );
    assert_eq!(
        gate_named(&report, "gate_bounded_volume_vs_box").status,
        QaStatus::Pass
    );
}

#[test]
fn plate_mesh_warns_on_watertight() {
    let mesh = plate_mesh(1.0, 1.0, 10, 10);
    let report = run_qa(&mesh, None);
    assert_eq!(
        gate_named(&report, "gate_finite_vertices").status,
        QaStatus::Pass
    );
    assert_eq!(
        gate_named(&report, "gate_no_degenerate_triangles").status,
        QaStatus::Pass
    );
    let w = gate_named(&report, "gate_watertight");
    assert_eq!(w.status, QaStatus::Warn, "plate must warn on watertight");
    assert!(w.value > 0.0, "expected open edges > 0");
    assert!(
        w.message.contains("expected for open primitive"),
        "watertight message should flag plate as expected-open: {}",
        w.message
    );
    assert_eq!(
        gate_named(&report, "gate_bounded_volume_vs_box").status,
        QaStatus::Skipped
    );
    assert_eq!(
        gate_named(&report, "gate_max_electrical_size").status,
        QaStatus::Skipped
    );
    assert_eq!(report.overall_status, QaStatus::Warn);
}

#[test]
fn nan_vertex_fails_finite_gate() {
    let mut mesh = sphere_mesh(1.0, 6, 8);
    assert!(inject_nan_vertex(&mut mesh, 3));
    let report = run_qa(&mesh, None);
    let g = gate_named(&report, "gate_finite_vertices");
    assert_eq!(g.status, QaStatus::Fail);
    assert!(g.value >= 1.0, "expected at least one bad triangle counted");
    assert_eq!(report.overall_status, QaStatus::Fail);
}

#[test]
fn degenerate_triangle_fails_degenerate_gate() {
    let mut mesh = sphere_mesh(1.0, 6, 8);
    assert!(inject_degenerate_triangle(&mut mesh, 5));
    let report = run_qa(&mesh, None);
    let g = gate_named(&report, "gate_no_degenerate_triangles");
    assert_eq!(g.status, QaStatus::Fail);
    assert!(g.value >= 1.0);
    assert_eq!(report.overall_status, QaStatus::Fail);
}

#[test]
fn electrical_size_sphere_10ghz_under_budget() {
    let mesh = sphere_mesh(1.0, 16, 32);
    let report = run_qa(&mesh, Some(10e9));
    let g = gate_named(&report, "gate_max_electrical_size");
    assert_eq!(
        g.status,
        QaStatus::Pass,
        "10 GHz sphere should pass: {}",
        g.message
    );
    let expected = 2.0 * 10e9 / SPEED_OF_LIGHT_M_PER_S;
    assert!(
        (g.value - expected).abs() < 0.5,
        "expected ~{:.2} wavelengths, got {:.2}",
        expected,
        g.value
    );
}

#[test]
fn electrical_size_warns_above_threshold() {
    let mesh = sphere_mesh(5.0, 8, 12);
    let report = run_qa(&mesh, Some(100e9));
    let g = gate_named(&report, "gate_max_electrical_size");
    assert_eq!(g.status, QaStatus::Warn);
    assert!(g.value > ELECTRICAL_SIZE_WARN_LAMBDA);
}

#[test]
fn electrical_size_missing_frequency_is_skipped() {
    let mesh = sphere_mesh(1.0, 4, 6);
    let report = run_qa(&mesh, None);
    assert_eq!(
        gate_named(&report, "gate_max_electrical_size").status,
        QaStatus::Skipped
    );
}

#[test]
fn qa_status_worse_of_orders_correctly() {
    assert_eq!(QaStatus::Pass.worse_of(QaStatus::Pass), QaStatus::Pass);
    assert_eq!(QaStatus::Pass.worse_of(QaStatus::Warn), QaStatus::Warn);
    assert_eq!(QaStatus::Warn.worse_of(QaStatus::Fail), QaStatus::Fail);
    assert_eq!(QaStatus::Fail.worse_of(QaStatus::Warn), QaStatus::Fail);
    assert_eq!(QaStatus::Skipped.worse_of(QaStatus::Warn), QaStatus::Warn);
    assert_eq!(QaStatus::Pass.worse_of(QaStatus::Skipped), QaStatus::Pass);
    assert_eq!(
        QaStatus::Skipped.worse_of(QaStatus::Pass),
        QaStatus::Skipped
    );
}

#[test]
fn sphere_volume_matches_pi_over_six() {
    let mesh = sphere_mesh(1.0, 32, 64);
    let report = run_qa(&mesh, None);
    let v = gate_named(&report, "gate_bounded_volume_vs_box");
    assert_eq!(
        v.status,
        QaStatus::Pass,
        "sphere volume gate: {}",
        v.message
    );
    assert!(
        (v.value - std::f64::consts::PI / 6.0).abs() < 0.05,
        "sphere V/V_bbox should be ~pi/6, got {:.4}",
        v.value
    );
}

#[test]
fn plate_winding_is_consistent() {
    let mesh = plate_mesh(2.0, 2.0, 6, 6);
    let report = run_qa(&mesh, None);
    assert_eq!(
        gate_named(&report, "gate_consistent_winding").status,
        QaStatus::Pass
    );
}

#[test]
fn report_overall_status_is_worst_gate() {
    let mut mesh = sphere_mesh(1.0, 6, 10);
    assert!(inject_nan_vertex(&mut mesh, 0));
    assert!(inject_degenerate_triangle(&mut mesh, 2));
    let report = run_qa(&mesh, Some(1e9));
    assert_eq!(report.overall_status, QaStatus::Fail);

    let plate = plate_mesh(1.0, 1.0, 4, 4);
    let plate_report = run_qa(&plate, Some(1e9));
    assert_eq!(plate_report.overall_status, QaStatus::Warn);
}

#[test]
fn report_serialises_round_trip() {
    let mesh = plate_mesh(1.0, 1.0, 5, 5);
    let report = run_qa(&mesh, Some(2.4e9));
    let json = serde_json::to_string(&report).expect("serialise");
    let round: QaReport = serde_json::from_str(&json).expect("deserialise");
    assert_eq!(round, report);
    assert!(json.contains("\"overall_status\""));
    assert!(json.contains("\"gate_finite_vertices\""));
    assert!(
        json.contains("\"warn\""),
        "watertight gate should warn: {}",
        json
    );
}

#[test]
fn inject_helpers_return_false_on_out_of_range() {
    let mut mesh = sphere_mesh(1.0, 4, 6);
    let n = mesh.triangle_count();
    assert!(!inject_nan_vertex(&mut mesh, n + 99));
    assert!(!inject_degenerate_triangle(&mut mesh, n + 99));
}
