//! PEC right-angle dihedral. Peak boresight σ = 8π w² h² / λ². Ruck Ch. 9.

use serde::{Deserialize, Serialize};

use super::{CanonicalTruth, Conditions, Truth};
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
        8.0 * std::f64::consts::PI * self.width_m.powi(2) * self.height_m.powi(2) / (lambda * lambda)
    }
}

impl CanonicalTruth for PecDihedral {
    fn sigma_m2(&self, conditions: &Conditions) -> Truth {
        let lambda = conditions.wavelength_m();
        // Boresight peak (Ruck Ch. 9): 8π w² h² / λ²; off-boresight falls as cos⁴(θ)·cos⁴(φ).
        let peak = 8.0 * std::f64::consts::PI
            * self.width_m.powi(2) * self.height_m.powi(2)
            / (lambda * lambda);
        let cos_t = conditions.theta_rad.cos();
        let cos_p = conditions.phi_rad.cos();
        let sigma = peak * cos_t.powi(4) * cos_p.powi(4);
        let at_boresight =
            conditions.theta_rad.abs() < 1e-9 && conditions.phi_rad.abs() < 1e-9;
        Truth::ok(sigma, if at_boresight { "boresight" } else { "main_lobe" })
    }

    fn validity_mask(&self, conditions: &Conditions) -> bool {
        super::kw_kh_large(self.width_m, self.height_m, conditions.wavelength_m())
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
    fn peak_formula_matches() {
        let d = PecDihedral { width_m: 0.3, height_m: 0.3 };
        let f = 10e9;
        let lambda = SPEED_OF_LIGHT / f;
        // Expected peak sigma: 8π w² h² / λ²
        let expected = 8.0 * std::f64::consts::PI * 0.3f64.powi(2) * 0.3f64.powi(2) / (lambda * lambda);
        let t = d.sigma_m2(&Conditions::broadside(f));
        assert!((t.value / expected - 1.0).abs() < 1e-12);
    }
}
