//! PEC right-angle dihedral. Peak boresight σ = 8π w² h² / λ². Ruck Ch. 9.

use serde::{Deserialize, Serialize};

use super::{CanonicalTruth, Conditions, Status, Truth};
use crate::tolerance::ToleranceBand;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PecDihedral {
    pub width_m: f64,
    pub height_m: f64,
}

impl PecDihedral {
    pub fn new(width_m: f64, height_m: f64) -> Self {
        Self { width_m, height_m }
    }

    pub fn peak_sigma(&self, lambda: f64) -> f64 {
        8.0 * std::f64::consts::PI * self.width_m.powi(2) * self.height_m.powi(2)
            / (lambda * lambda)
    }
}

impl CanonicalTruth for PecDihedral {
    fn sigma_m2(&self, conditions: &Conditions) -> Truth {
        let lambda = conditions.wavelength_m();
        let peak = self.peak_sigma(lambda);
        // Off-boresight cos⁴ falloff (Knott §9.3); use a simple parameterization
        // valid within the main lobe of a 90° dihedral.
        let fall = conditions.theta_rad.cos().powi(4) * conditions.phi_rad.cos().powi(4);
        let sigma = peak * fall.max(0.0);
        let regime = if conditions.theta_rad.abs() < 1e-9 && conditions.phi_rad.abs() < 1e-9 {
            "boresight"
        } else {
            "main_lobe"
        };
        Truth {
            value: sigma,
            regime: regime.to_string(),
            status: Status::Pass,
        }
    }

    fn validity_mask(&self, conditions: &Conditions) -> bool {
        let lambda = conditions.wavelength_m();
        let kw = 2.0 * std::f64::consts::PI * self.width_m / lambda;
        let kh = 2.0 * std::f64::consts::PI * self.height_m / lambda;
        kw > 3.0 && kh > 3.0
    }

    fn tolerance(&self, _conditions: &Conditions) -> ToleranceBand {
        ToleranceBand {
            analytic_db: 0.5,
            numeric_db: 0.0,
            method_db: 0.0,
            total_db: 0.5,
            floor_db: 0.1,
            ceiling_db: 3.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::units::SPEED_OF_LIGHT;

    #[test]
    fn peak_formula_matches() {
        let d = PecDihedral::new(0.3, 0.3);
        let f = 10e9;
        let lambda = SPEED_OF_LIGHT / f;
        let t = d.sigma_m2(&Conditions::broadside(f));
        assert!((t.value / d.peak_sigma(lambda) - 1.0).abs() < 1e-12);
    }
}
