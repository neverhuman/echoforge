use echoforge_validate::primitives::{CanonicalTruth, Conditions};
use echoforge_validate::{sigma_to_dbsm, PecCone, SPEED_OF_LIGHT};

#[test]
fn tip_on_within_two_db() {
    // Approximation documented as loose; ensure the closed form is what we get.
    let cone = PecCone::new(15f64.to_radians());
    let f = 10e9;
    let lambda = SPEED_OF_LIGHT / f;
    let t = cone.sigma_m2(&Conditions::broadside(f));
    let expected = cone.tip_sigma(lambda);
    let delta_db = (sigma_to_dbsm(t.value) - sigma_to_dbsm(expected)).abs();
    assert!(delta_db < 2.0, "delta={delta_db:.3} dB");
}
