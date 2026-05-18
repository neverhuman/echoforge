//! Sphere Mie sanity grid: Rayleigh + resonance + optical regimes.

use echoforge_validate::primitives::{CanonicalTruth, Conditions};
use echoforge_validate::{sigma_to_dbsm, PecSphere, SPEED_OF_LIGHT};

fn cond(f_hz: f64) -> Conditions {
    Conditions::broadside(f_hz)
}

fn freq_for_ka(radius_m: f64, ka: f64) -> f64 {
    let lambda = 2.0 * std::f64::consts::PI * radius_m / ka;
    SPEED_OF_LIGHT / lambda
}

#[test]
fn rayleigh_grid_within_tolerance() {
    // 5 (radius, ka) points in the Rayleigh regime.
    for (a, ka) in [
        (0.01, 0.05),
        (0.01, 0.1),
        (0.05, 0.1),
        (0.05, 0.2),
        (0.1, 0.15),
    ] {
        let s = PecSphere::new(a);
        let t = s.sigma_m2(&cond(freq_for_ka(a, ka)));
        let asy = s.rayleigh_sigma(ka);
        let delta_db = (sigma_to_dbsm(t.value) - sigma_to_dbsm(asy)).abs();
        assert!(delta_db < 0.5, "a={a} ka={ka} delta={delta_db:.3} dB");
        assert_eq!(t.regime, "rayleigh");
    }
}

#[test]
fn optical_grid_within_tolerance() {
    // Optical: ka large; σ → π a². Allow ±3 dB (oscillations slow to decay).
    for (a, ka) in [
        (0.1, 30.0),
        (0.1, 60.0),
        (0.5, 80.0),
        (0.5, 120.0),
        (1.0, 150.0),
    ] {
        let s = PecSphere::new(a);
        let t = s.sigma_m2(&cond(freq_for_ka(a, ka)));
        let geo = s.optical_sigma();
        let delta_db = (sigma_to_dbsm(t.value) - sigma_to_dbsm(geo)).abs();
        assert!(
            delta_db < 3.0,
            "a={a} ka={ka} delta={delta_db:.3} dB sigma={} geo={geo}",
            t.value
        );
        assert_eq!(t.regime, "optical");
    }
}

#[test]
fn resonance_region_returns_resonance_label() {
    let s = PecSphere::new(0.05);
    let t = s.sigma_m2(&cond(freq_for_ka(0.05, 5.0)));
    assert_eq!(t.regime, "resonance");
    assert!(t.value > 0.0);
}
