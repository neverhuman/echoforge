//! PEC trihedral corner reflector. Boresight σ for square = 12π l⁴ / λ², for
//! triangular = 4π l⁴ / (3 λ²). Ruck Ch. 9.

use serde::{Deserialize, Serialize};

use super::{CanonicalTruth, Conditions, Status, Truth};
use crate::tolerance::ToleranceBand;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TrihedralShape {
    Square,
    Triangular,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PecTrihedral {
    pub edge_m: f64,
    pub shape: TrihedralShape,
}

impl PecTrihedral {
    pub fn new(edge_m: f64, shape: TrihedralShape) -> Self {
        Self { edge_m, shape }
    }

    pub fn peak_sigma(&self, lambda: f64) -> f64 {
        let l4 = self.edge_m.powi(4);
        match self.shape {
            TrihedralShape::Square => 12.0 * std::f64::consts::PI * l4 / (lambda * lambda),
            TrihedralShape::Triangular => 4.0 * std::f64::consts::PI * l4 / (3.0 * lambda * lambda),
        }
    }
}

impl CanonicalTruth for PecTrihedral {
    fn sigma_m2(&self, conditions: &Conditions) -> Truth {
        let lambda = conditions.wavelength_m();
        let peak = self.peak_sigma(lambda);
        let regime = match self.shape {
            TrihedralShape::Square => "square_boresight",
            TrihedralShape::Triangular => "triangular_boresight",
        };
        Truth {
            value: peak,
            regime: regime.to_string(),
            status: Status::Pass,
        }
    }

    fn validity_mask(&self, conditions: &Conditions) -> bool {
        let lambda = conditions.wavelength_m();
        let kl = 2.0 * std::f64::consts::PI * self.edge_m / lambda;
        kl > 3.0
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
    fn ratio_between_shapes() {
        let f = 10e9;
        let lambda = SPEED_OF_LIGHT / f;
        let sq = PecTrihedral::new(0.3, TrihedralShape::Square);
        let tr = PecTrihedral::new(0.3, TrihedralShape::Triangular);
        let ratio = sq.peak_sigma(lambda) / tr.peak_sigma(lambda);
        // 12π / (4π/3) = 9.
        assert!((ratio - 9.0).abs() < 1e-9);
    }
}
