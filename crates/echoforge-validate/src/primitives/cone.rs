//! PEC tip-on cone. σ_tip ≈ (λ² / 16π) · tan⁴(α/2). Approximation valid in
//! the small half-angle, tip-on regime (Ruck Ch. 8, eq. 8.4-15). Documented as
//! loose (±2 dB).

use serde::{Deserialize, Serialize};

use super::{CanonicalTruth, Conditions, Truth};
use crate::tolerance::ToleranceBand;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PecCone {
    pub half_angle_rad: f64,
}

impl PecCone {
    pub fn new(half_angle_rad: f64) -> Self {
        Self { half_angle_rad }
    }

    pub fn tip_sigma(&self, lambda: f64) -> f64 {
        let t = (self.half_angle_rad / 2.0).tan();
        (lambda * lambda) / (16.0 * std::f64::consts::PI) * t.powi(4)
    }
}

impl CanonicalTruth for PecCone {
    fn sigma_m2(&self, conditions: &Conditions) -> Truth {
        let lambda = conditions.wavelength_m();
        let sigma = self.tip_sigma(lambda);
        Truth::ok(sigma, "tip_on")
    }

    fn validity_mask(&self, _conditions: &Conditions) -> bool {
        // Approximation is restricted to small half-angles and tip-on aspect.
        self.half_angle_rad < std::f64::consts::PI / 4.0
    }

    fn tolerance(&self, _conditions: &Conditions) -> ToleranceBand {
        ToleranceBand::analytic_only(2.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::units::SPEED_OF_LIGHT;

    #[test]
    fn tip_formula() {
        let c = PecCone::new(15f64.to_radians());
        let f = 10e9;
        let lambda = SPEED_OF_LIGHT / f;
        let t = c.sigma_m2(&Conditions::broadside(f));
        let expected = c.tip_sigma(lambda);
        assert!((t.value / expected - 1.0).abs() < 1e-12);
    }
}
