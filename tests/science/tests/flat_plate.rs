//! Flat plate broadside + first sinc² null check.

use echoforge_validate::primitives::{CanonicalTruth, Conditions};
use echoforge_validate::{sigma_to_dbsm, PecFlatPlate, SPEED_OF_LIGHT};

#[test]
fn broadside_within_half_db() {
    let p = PecFlatPlate::new(0.3, 0.3);
    let f = 10e9;
    let lambda = SPEED_OF_LIGHT / f;
    let expected = p.broadside_sigma(lambda);
    let t = p.sigma_m2(&Conditions::broadside(f));
    let delta_db = (sigma_to_dbsm(t.value) - sigma_to_dbsm(expected)).abs();
    assert!(delta_db < 0.5, "delta={delta_db:.3} dB");
}

#[test]
fn first_null_position_within_half_db() {
    let p = PecFlatPlate::new(0.3, 0.3);
    let f = 10e9;
    let lambda = SPEED_OF_LIGHT / f;
    let theta_null = (lambda / (2.0 * p.width_m)).asin();
    let mut c = Conditions::broadside(f);
    c.theta_rad = theta_null;
    let t = p.sigma_m2(&c);
    let broadside = p.sigma_m2(&Conditions::broadside(f));
    let drop_db = sigma_to_dbsm(broadside.value) - sigma_to_dbsm(t.value);
    assert!(drop_db > 40.0, "null drop only {drop_db:.3} dB");
}
