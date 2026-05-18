use echoforge_validate::primitives::{CanonicalTruth, Conditions};
use echoforge_validate::{sigma_to_dbsm, PecTrihedral, TrihedralShape, SPEED_OF_LIGHT};

#[test]
fn both_shapes_within_half_db() {
    let f = 10e9;
    let lambda = SPEED_OF_LIGHT / f;
    for shape in [TrihedralShape::Square, TrihedralShape::Triangular] {
        let tri = PecTrihedral::new(0.3, shape);
        let t = tri.sigma_m2(&Conditions::broadside(f));
        let expected = tri.peak_sigma(lambda);
        let delta_db = (sigma_to_dbsm(t.value) - sigma_to_dbsm(expected)).abs();
        assert!(delta_db < 0.5, "{:?} delta={delta_db:.3} dB", shape);
    }
}
