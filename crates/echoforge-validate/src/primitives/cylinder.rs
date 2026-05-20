//! PEC finite right cylinder. Broadside σ = 2π a L² / λ; off-broadside
//! σ(θ) = (2π a L² / λ) · sinc²(k L sin θ / π) · cos²(θ). Validity ka > 5
//! (high-frequency physical optics regime). Ruck Ch. 8.

use serde::{Deserialize, Serialize};

use super::{CanonicalTruth, Conditions, Truth};
use crate::tolerance::ToleranceBand;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PecCylinder {
    pub radius_m: f64,
    pub length_m: f64,
}

impl PecCylinder {
    pub fn new(radius_m: f64, length_m: f64) -> Self {
        Self { radius_m, length_m }
    }

    pub fn broadside_sigma(&self, lambda: f64) -> f64 {
        2.0 * std::f64::consts::PI * self.radius_m * self.length_m.powi(2) / lambda
    }
}

impl CanonicalTruth for PecCylinder {
    fn sigma_m2(&self, conditions: &Conditions) -> Truth {
        let lambda = conditions.wavelength_m();
        let ka = 2.0 * std::f64::consts::PI * self.radius_m / lambda;
        if ka <= 5.0 {
            return Truth::skipped("ka<=5, outside high-frequency validity");
        }
        let k = 2.0 * std::f64::consts::PI / lambda;
        let broadside = self.broadside_sigma(lambda);
        let theta = conditions.theta_rad;
        let s = super::sinc(k * self.length_m * theta.sin());
        let sigma = broadside * s.powi(2) * theta.cos().powi(2);
        let regime = if theta.abs() < 1e-9 {
            "broadside"
        } else {
            "main_lobe"
        };
        Truth::ok(sigma, regime)
    }

    fn validity_mask(&self, conditions: &Conditions) -> bool {
        let lambda = conditions.wavelength_m();
        let ka = 2.0 * std::f64::consts::PI * self.radius_m / lambda;
        ka > 5.0
    }

    fn tolerance(&self, _conditions: &Conditions) -> ToleranceBand {
        ToleranceBand::analytic_only(0.5)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::units::SPEED_OF_LIGHT;

    #[test]
    fn broadside_formula() {
        let c = PecCylinder::new(0.05, 0.5);
        let f = 10e9;
        let lambda = SPEED_OF_LIGHT / f;
        let t = c.sigma_m2(&Conditions::broadside(f));
        let expected = c.broadside_sigma(lambda);
        assert!((t.value / expected - 1.0).abs() < 1e-12);
        assert_eq!(t.regime, "broadside");
    }

    #[test]
    fn skipped_below_validity() {
        // Make ka small.
        let c = PecCylinder::new(0.001, 0.5);
        let f = 1e9;
        let t = c.sigma_m2(&Conditions::broadside(f));
        assert!(matches!(t.status, super::super::Status::Skipped));
    }
}
