//! Determinism check utilities.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeterminismReport {
    pub mode: String,
    pub backend: String,
    pub max_abs: f64,
    pub max_rel: f64,
}

pub fn bit_identical_f64(a: &[f64], b: &[f64]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.iter()
        .zip(b.iter())
        .all(|(x, y)| x.to_bits() == y.to_bits())
}

pub fn fp_tolerant_f32(a: &[f32], b: &[f32], rel: f32, abs: f32) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.iter().zip(b.iter()).all(|(x, y)| {
        let diff = (x - y).abs();
        diff <= abs || diff <= rel * x.abs().max(y.abs())
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bit_identical_works() {
        let a = vec![1.0f64, 2.0, 3.0];
        let b = a.clone();
        assert!(bit_identical_f64(&a, &b));
    }

    #[test]
    fn bit_identical_rejects_tiny_diff() {
        let a = vec![1.0f64];
        let b = vec![1.0 + f64::EPSILON];
        assert!(!bit_identical_f64(&a, &b));
    }

    #[test]
    fn fp_tolerant_accepts_within_rel() {
        let a = vec![1.0f32];
        let b = vec![1.0 + 1e-6];
        assert!(fp_tolerant_f32(&a, &b, 1e-5, 0.0));
    }
}
