//! Uncertainty propagation (Jacobian linearization, v0).

/// Linearized output variance: u_out² ≈ Σ (∂f/∂x_i)² σ_{x_i}².
pub fn linearize(jacobian: &[f64], input_sigmas: &[f64]) -> f64 {
    assert_eq!(jacobian.len(), input_sigmas.len());
    jacobian
        .iter()
        .zip(input_sigmas.iter())
        .map(|(j, s)| (j * s).powi(2))
        .sum::<f64>()
        .sqrt()
}

/// Lorentzian confidence from a per-cell σ_db: c = 1 / (1 + (σ/3)²).
pub fn confidence_from_sigma_db(s: f64) -> f64 {
    let r = s / 3.0;
    1.0 / (1.0 + r * r)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn confidence_decreases_with_sigma() {
        assert!(confidence_from_sigma_db(0.0) > confidence_from_sigma_db(3.0));
        assert!((confidence_from_sigma_db(0.0) - 1.0).abs() < 1e-12);
        assert!((confidence_from_sigma_db(3.0) - 0.5).abs() < 1e-12);
    }

    #[test]
    fn linearize_pythagorean() {
        let u = linearize(&[1.0, 1.0], &[3.0, 4.0]);
        assert!((u - 5.0).abs() < 1e-12);
    }
}
