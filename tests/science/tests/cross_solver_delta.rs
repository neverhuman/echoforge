use echoforge_validate::{CrossSolverDelta, DeltaPair};

#[test]
fn analytic_vs_solver_with_small_perturbation_passes() {
    let pairs = vec![
        DeltaPair::new("analytic", "solver", -10.0, -9.7, 0.5),
        DeltaPair::new("analytic", "solver", 0.0, 0.3, 0.5),
        DeltaPair::new("analytic", "solver", 5.0, 5.3, 0.5),
    ];
    let cs = CrossSolverDelta::from_pairs(pairs);
    assert_eq!(cs.overall_status(), "pass");
    assert!(cs.summary.rms_delta_db < 0.5);
}

#[test]
fn rms_above_one_db_hard_fails() {
    let pairs = vec![DeltaPair::new("a", "b", 0.0, 1.5, 2.0)];
    let cs = CrossSolverDelta::from_pairs(pairs);
    assert_eq!(cs.overall_status(), "fail");
}
