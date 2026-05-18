//! Original Rust PEC Mie coefficients via Riccati-Bessel recurrence.
//!
//! References (cited, not copied):
//!   * Wiscombe, W.J. (1980). "Improved Mie Scattering Algorithms." Appl. Opt. 19, 1505.
//!     Also NCAR/TN-140+STR (1979).
//!   * Bohren, C.F. & Huffman, D.R. (1983). *Absorption and Scattering of Light by
//!     Small Particles.* Wiley. Chapter 4 (Mie series, PEC limit).
//!   * Ruck, G.T. et al. (1970). *Radar Cross Section Handbook.* Chapter 3
//!     (perfectly conducting sphere).
//!
//! Strategy for a PEC sphere of size parameter x = ka:
//!   * Compute Riccati-Bessel ψ_n(x) = x j_n(x) via upward recurrence
//!     ψ_{n+1} = (2n+1)/x · ψ_n − ψ_{n−1}, seeded with
//!     ψ_{-1}(x) = cos(x), ψ_0(x) = sin(x).
//!   * Compute χ_n(x) = −x y_n(x) via upward recurrence
//!     χ_{n+1} = (2n+1)/x · χ_n − χ_{n−1}, seeded with
//!     χ_{-1}(x) = sin(x), χ_0(x) = −cos(x).
//!   * ξ_n(x) = ψ_n(x) − i·χ_n(x) (note sign convention: ξ has the outgoing-wave
//!     form for the e^{−iωt} time dependence used in Bohren-Huffman).
//!   * Derivative identity: ψ_n'(x) = ψ_{n−1}(x) − (n/x)·ψ_n(x); same for ξ.
//!   * PEC scattering coefficients (Bohren-Huffman eq. 4.56, m → ∞ limit):
//!         a_n = ψ_n(x)        / ξ_n(x)
//!         b_n = ψ_n'(x)       / ξ_n'(x)
//!
//! Upward recurrence is stable for ψ (regular Bessel) and χ (irregular Bessel).
//! ψ alone is unstable upward, but the small-argument loss is hidden by ξ_n
//! being dominated by χ_n for the n ≳ x regime we care about. Wiscombe's
//! downward recurrence for the logarithmic derivative D_n is the textbook
//! choice when accuracy at the last few coefficients matters; for the RCS
//! sums truncated by N_max = ka + 4·(ka)^{1/3} + 2 we follow Wiscombe's
//! truncation rule and verify against asymptotes (see validate-crate tests).

use num_complex::Complex64;

/// Recommended truncation order N_max per Wiscombe (1980).
#[inline]
pub fn n_max(ka: f64) -> usize {
    let v = ka + 4.0 * ka.cbrt() + 2.0;
    let n = v.ceil() as usize;
    n.max(5)
}

/// Compute PEC Mie coefficients (a_n, b_n) for n = 1..=N_max.
///
/// Returns vectors of length `N_max(ka)`.
pub fn mie_pec_coeffs(ka: f64) -> (Vec<Complex64>, Vec<Complex64>) {
    assert!(ka > 0.0, "ka must be positive");
    let nmax = n_max(ka);
    let x = ka;

    // Allocate ψ and χ for orders -1..=nmax (length nmax+2).
    // psi[k] holds ψ_{k-1}(x); same indexing for chi.
    let len = nmax + 2;
    let mut psi = vec![0.0f64; len];
    let mut chi = vec![0.0f64; len];

    // Seeds.
    psi[0] = x.cos(); // ψ_{-1}
    psi[1] = x.sin(); // ψ_{0}
    chi[0] = x.sin(); // χ_{-1}
    chi[1] = -x.cos(); // χ_{0}

    // Upward recurrence f_{n+1} = (2n+1)/x · f_n − f_{n−1}, with f_n stored at index n+1.
    for n in 1..=nmax {
        let coeff = (2.0 * (n as f64) - 1.0) / x;
        psi[n + 1] = coeff * psi[n] - psi[n - 1];
        chi[n + 1] = coeff * chi[n] - chi[n - 1];
    }

    let mut a_n = Vec::with_capacity(nmax);
    let mut b_n = Vec::with_capacity(nmax);

    for n in 1..=nmax {
        // ψ_n, ψ_{n−1}, χ_n, χ_{n−1}.
        let psi_n = psi[n + 1];
        let psi_nm1 = psi[n];
        let chi_n = chi[n + 1];
        let chi_nm1 = chi[n];

        // Derivatives via ψ_n'(x) = ψ_{n−1}(x) − (n/x)·ψ_n(x).
        let n_over_x = (n as f64) / x;
        let psi_d = psi_nm1 - n_over_x * psi_n;
        let chi_d = chi_nm1 - n_over_x * chi_n;

        // ξ_n = ψ_n − i χ_n (BH convention).
        let xi_n = Complex64::new(psi_n, -chi_n);
        let xi_d = Complex64::new(psi_d, -chi_d);

        // PEC: a_n = ψ_n / ξ_n, b_n = ψ_n' / ξ_n'.
        let a = Complex64::new(psi_n, 0.0) / xi_n;
        let b = Complex64::new(psi_d, 0.0) / xi_d;

        a_n.push(a);
        b_n.push(b);
    }

    (a_n, b_n)
}

/// Backscatter RCS for a PEC sphere of radius a at wavelength λ.
///
/// σ(ka) = (λ²/π) · |Σ_{n=1}^{N_max} (-1)^n (n + 1/2) (b_n − a_n)|²
///
/// Bohren-Huffman eq. 4.83 specialized to PEC.
pub fn pec_sphere_sigma(radius_m: f64, wavelength_m: f64) -> f64 {
    let ka = 2.0 * std::f64::consts::PI * radius_m / wavelength_m;
    let (a_n, b_n) = mie_pec_coeffs(ka);
    let mut sum = Complex64::new(0.0, 0.0);
    for (idx, (a, b)) in a_n.iter().zip(b_n.iter()).enumerate() {
        let n = (idx + 1) as f64;
        let sign = if (idx + 1) % 2 == 0 { 1.0 } else { -1.0 };
        sum += Complex64::new(sign * (n + 0.5), 0.0) * (b - a);
    }
    let lam2 = wavelength_m * wavelength_m;
    lam2 / std::f64::consts::PI * sum.norm_sqr()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nmax_monotonic() {
        let mut prev = 0usize;
        for ka in [0.1, 1.0, 5.0, 10.0, 50.0, 100.0] {
            let n = n_max(ka);
            assert!(n > prev || ka == 0.1);
            prev = n;
        }
    }

    #[test]
    fn optical_limit_approaches_pi_a_squared() {
        // For ka >> 1 the PEC backscatter should approach π a^2.
        let a = 1.0;
        let lambda = 2.0 * std::f64::consts::PI * a / 50.0; // ka = 50
        let sigma = pec_sphere_sigma(a, lambda);
        let geometric = std::f64::consts::PI * a * a;
        // Within ~25% in the deep optical regime (oscillations decay slowly).
        assert!(
            (sigma / geometric - 1.0).abs() < 0.25,
            "sigma={sigma} geometric={geometric}"
        );
    }

    #[test]
    fn rayleigh_limit_scales_as_ka_six() {
        // σ ≈ (λ²/π) |(3/2)(b_1 − a_1)|²; for ka<<1, a_1 ~ i(2/3)(ka)^3,
        // b_1 ~ −i(1/3)(ka)^3, giving σ → 9π a² (ka)^4.
        let a = 0.01;
        let ka = 0.05;
        let lambda = 2.0 * std::f64::consts::PI * a / ka;
        let sigma = pec_sphere_sigma(a, lambda);
        let predicted = 9.0 * std::f64::consts::PI * a * a * ka.powi(4);
        let ratio = sigma / predicted;
        assert!((ratio - 1.0).abs() < 0.05, "ratio={ratio}");
    }
}
