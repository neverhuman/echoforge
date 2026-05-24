use echoforge_validate::primitives::{CanonicalTruth, Conditions};
use echoforge_validate::{bit_identical_f64, PecSphere, SPEED_OF_LIGHT};

fn run_sphere_grid() -> Vec<f64> {
    let s = PecSphere::new(0.1);
    let mut out = Vec::new();
    for ka in [0.5_f64, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0] {
        let lambda = 2.0 * std::f64::consts::PI * s.radius_m / ka;
        let f = SPEED_OF_LIGHT / lambda;
        let t = s.sigma_m2(&Conditions::broadside(f));
        out.push(t.value);
    }
    out
}

#[test]
fn cpu_byte_equal_on_replay() {
    let a = run_sphere_grid();
    let b = run_sphere_grid();
    assert!(
        bit_identical_f64(&a, &b),
        "non-deterministic CPU run: a={:?} b={:?}",
        a,
        b
    );
}
