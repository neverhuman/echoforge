//! PEC sphere ground truth via the Mie series (see [`crate::bessel`]).
//!
//! Three regimes are tracked for tolerance assignment:
//!   * **Rayleigh** (ka < 0.4): σ ≈ 9π a² (ka)^4. Mie series collapses to this.
//!   * **Resonance** (0.4 ≤ ka ≤ 20): oscillatory; full Mie series required.
//!   * **Optical** (ka > 20): σ → π a² (geometric cross section).

use serde::{Deserialize, Serialize};

use super::{CanonicalTruth, Conditions, Status, Truth};
use crate::bessel::pec_sphere_sigma;
use crate::tolerance::ToleranceBand;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PecSphere {
    pub radius_m: f64,
}

impl PecSphere {
    pub fn new(radius_m: f64) -> Self {
        Self { radius_m }
    }

    fn regime_label(ka: f64) -> &'static str {
        if ka < 0.4 {
            "rayleigh"
        } else if ka > 20.0 {
            "optical"
        } else {
            "resonance"
        }
    }

    /// Closed-form Rayleigh asymptote σ = 9π a² (ka)^4.
    pub fn rayleigh_sigma(&self, ka: f64) -> f64 {
        9.0 * std::f64::consts::PI * self.radius_m * self.radius_m * ka.powi(4)
    }

    /// Optical asymptote σ = π a².
    pub fn optical_sigma(&self) -> f64 {
        std::f64::consts::PI * self.radius_m * self.radius_m
    }
}

impl CanonicalTruth for PecSphere {
    fn sigma_m2(&self, conditions: &Conditions) -> Truth {
        let lambda = conditions.wavelength_m();
        let ka = 2.0 * std::f64::consts::PI * self.radius_m / lambda;
        let sigma = pec_sphere_sigma(self.radius_m, lambda);
        Truth {
            value: sigma,
            regime: Self::regime_label(ka).to_string(),
            status: Status::Pass,
        }
    }

    fn validity_mask(&self, conditions: &Conditions) -> bool {
        conditions.frequency_hz > 0.0 && self.radius_m > 0.0
    }

    fn tolerance(&self, conditions: &Conditions) -> ToleranceBand {
        let lambda = conditions.wavelength_m();
        let ka = 2.0 * std::f64::consts::PI * self.radius_m / lambda;
        let band = if (0.4..=20.0).contains(&ka) {
            0.25
        } else {
            0.10
        };
        ToleranceBand {
            analytic_db: band,
            numeric_db: 0.0,
            method_db: 0.0,
            total_db: band.clamp(0.1, 3.0),
            floor_db: 0.1,
            ceiling_db: 3.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::units::SPEED_OF_LIGHT;

    fn cond(f_hz: f64) -> Conditions {
        Conditions::broadside(f_hz)
    }

    #[test]
    fn rayleigh_matches_asymptote() {
        let s = PecSphere::new(0.01);
        let lambda = 2.0 * std::f64::consts::PI * s.radius_m / 0.1; // ka=0.1
        let f = SPEED_OF_LIGHT / lambda;
        let t = s.sigma_m2(&cond(f));
        let asy = s.rayleigh_sigma(0.1);
        assert!((t.value / asy - 1.0).abs() < 0.05);
        assert_eq!(t.regime, "rayleigh");
    }

    #[test]
    fn optical_approaches_geometric() {
        let s = PecSphere::new(0.5);
        let lambda = 2.0 * std::f64::consts::PI * s.radius_m / 80.0; // ka=80
        let f = SPEED_OF_LIGHT / lambda;
        let t = s.sigma_m2(&cond(f));
        let geo = s.optical_sigma();
        assert!((t.value / geo - 1.0).abs() < 0.25);
        assert_eq!(t.regime, "optical");
    }
}
