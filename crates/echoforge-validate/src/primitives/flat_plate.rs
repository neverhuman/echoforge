//! PEC rectangular flat plate, normal-incidence backscatter + sinc² off-axis.
//!
//! References: Knott, Shaeffer, Tuley *Radar Cross Section* 2nd ed. §11; Ruck
//! Ch. 7. Geometry: plate normal aligned with the radar line of sight at
//! θ=0; rotation about the y-axis (width direction) is `theta_rad`, rotation
//! about the x-axis (height direction) is `phi_rad`.
//!
//! σ(θ, φ) = (4π A²/λ²) · cos²(θ) · cos²(φ) · sinc²(k w sin θ / π) ·
//!           sinc²(k h sin φ / π)
//!
//! where `sinc(x) = sin(πx)/(πx)`.

use serde::{Deserialize, Serialize};

use super::{CanonicalTruth, Conditions, Status, Truth};
use crate::tolerance::ToleranceBand;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PecFlatPlate {
    pub width_m: f64,
    pub height_m: f64,
}

impl PecFlatPlate {
    pub fn new(width_m: f64, height_m: f64) -> Self {
        Self { width_m, height_m }
    }

    pub fn broadside_sigma(&self, lambda: f64) -> f64 {
        let area = self.width_m * self.height_m;
        4.0 * std::f64::consts::PI * area * area / (lambda * lambda)
    }
}

/// Unnormalized sinc: sin(x)/x.
#[inline]
fn sinc(x: f64) -> f64 {
    if x.abs() < 1e-12 {
        1.0
    } else {
        x.sin() / x
    }
}

impl CanonicalTruth for PecFlatPlate {
    fn sigma_m2(&self, conditions: &Conditions) -> Truth {
        let lambda = conditions.wavelength_m();
        let k = 2.0 * std::f64::consts::PI / lambda;
        let theta = conditions.theta_rad;
        let phi = conditions.phi_rad;
        let area = self.width_m * self.height_m;
        let prefactor = 4.0 * std::f64::consts::PI * area * area / (lambda * lambda);
        let cos_t = theta.cos();
        let cos_p = phi.cos();
        let sw = sinc(k * self.width_m * theta.sin());
        let sh = sinc(k * self.height_m * phi.sin());
        let sigma = prefactor * cos_t.powi(2) * cos_p.powi(2) * sw.powi(2) * sh.powi(2);
        let regime = if theta.abs() < 1e-9 && phi.abs() < 1e-9 {
            "broadside"
        } else if sw.abs() < 1e-3 || sh.abs() < 1e-3 {
            "null"
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
            analytic_db: 0.3,
            numeric_db: 0.0,
            method_db: 0.0,
            total_db: 0.3,
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
    fn broadside_formula() {
        let p = PecFlatPlate::new(0.3, 0.3);
        let f = 10e9;
        let lambda = SPEED_OF_LIGHT / f;
        let expected = p.broadside_sigma(lambda);
        let got = p.sigma_m2(&Conditions::broadside(f));
        assert!((got.value / expected - 1.0).abs() < 1e-12);
    }

    #[test]
    fn predicted_null_position() {
        // First null in theta: k·w·sin(θ) = π → sin(θ) = π/(k·w) = λ/(2w).
        let p = PecFlatPlate::new(0.3, 0.3);
        let f = 10e9;
        let lambda = SPEED_OF_LIGHT / f;
        let theta_null = (lambda / (2.0 * p.width_m)).asin();
        let mut c = Conditions::broadside(f);
        c.theta_rad = theta_null;
        let t = p.sigma_m2(&c);
        let broadside = p.sigma_m2(&Conditions::broadside(f));
        let drop_db = 10.0 * (broadside.value / t.value.max(1e-30)).log10();
        assert!(drop_db > 40.0, "null suppression {drop_db} dB too small");
    }
}
