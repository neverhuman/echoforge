use echoforge_validate::primitives::{CanonicalTruth, Conditions};
use echoforge_validate::{sigma_to_dbsm, PecDihedral, SPEED_OF_LIGHT};

#[test]
fn peak_formula_within_half_db() {
    let d = PecDihedral::new(0.3, 0.3);
    let f = 10e9;
    let lambda = SPEED_OF_LIGHT / f;
    let t = d.sigma_m2(&Conditions::broadside(f));
    let expected = d.peak_sigma(lambda);
    let delta_db = (sigma_to_dbsm(t.value) - sigma_to_dbsm(expected)).abs();
    assert!(delta_db < 0.5, "delta={delta_db:.3} dB");
}
