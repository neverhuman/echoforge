use echoforge_validate::primitives::{CanonicalTruth, Conditions, Status};
use echoforge_validate::{sigma_to_dbsm, PecCylinder, SPEED_OF_LIGHT};

#[test]
fn broadside_within_half_db() {
    let c = PecCylinder::new(0.05, 0.5);
    let f = 10e9;
    let lambda = SPEED_OF_LIGHT / f;
    let t = c.sigma_m2(&Conditions::broadside(f));
    assert!(matches!(t.status, Status::Pass));
    let expected = c.broadside_sigma(lambda);
    let delta_db = (sigma_to_dbsm(t.value) - sigma_to_dbsm(expected)).abs();
    assert!(delta_db < 0.5, "delta={delta_db:.3} dB");
}

#[test]
fn first_null_at_predicted_angle() {
    let c = PecCylinder::new(0.05, 0.5);
    let f = 10e9;
    let lambda = SPEED_OF_LIGHT / f;
    // k L sin θ = π → sin θ = λ / (2L)
    let theta_null = (lambda / (2.0 * c.length_m)).asin();
    let mut cond = Conditions::broadside(f);
    cond.theta_rad = theta_null;
    let t = c.sigma_m2(&cond);
    let broadside = c.sigma_m2(&Conditions::broadside(f));
    let drop_db = sigma_to_dbsm(broadside.value) - sigma_to_dbsm(t.value);
    assert!(drop_db > 30.0, "null drop only {drop_db:.3} dB");
}
