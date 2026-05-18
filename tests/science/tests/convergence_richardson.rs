use echoforge_validate::richardson;

#[test]
fn synthetic_second_order() {
    // f(h) = 1 + h² is a textbook O(h²) function.
    let report = richardson(|h| 1.0 + h.powi(2), 0.1, 4, 2.0);
    assert!(report.pass, "{:?}", report);
    assert!(
        (report.p_observed - 2.0).abs() < 0.1,
        "p_observed = {}",
        report.p_observed
    );
}
