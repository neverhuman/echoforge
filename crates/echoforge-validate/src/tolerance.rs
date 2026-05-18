//! Tolerance budget combination per plan §6.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToleranceBand {
    pub analytic_db: f64,
    pub numeric_db: f64,
    pub method_db: f64,
    pub total_db: f64,
    pub floor_db: f64,
    pub ceiling_db: f64,
}

impl Default for ToleranceBand {
    fn default() -> Self {
        Self {
            analytic_db: 0.0,
            numeric_db: 0.0,
            method_db: 0.0,
            total_db: 0.1,
            floor_db: 0.1,
            ceiling_db: 3.0,
        }
    }
}

/// Combine three orthogonal dB error sources in quadrature, then clamp.
pub fn combine(analytic_db: f64, numeric_db: f64, method_db: f64) -> f64 {
    let raw = (analytic_db.powi(2) + numeric_db.powi(2) + method_db.powi(2)).sqrt();
    raw.clamp(0.1, 3.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn floor_applied() {
        assert!((combine(0.0, 0.0, 0.0) - 0.1).abs() < 1e-12);
    }

    #[test]
    fn ceiling_applied() {
        assert!((combine(10.0, 10.0, 10.0) - 3.0).abs() < 1e-12);
    }

    #[test]
    fn quadrature_combine() {
        let v = combine(0.3, 0.4, 0.0);
        assert!((v - 0.5).abs() < 1e-12);
    }
}
