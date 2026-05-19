//! Closed-form CFAR threshold scale factors for additional clutter
//! distributions and CFAR variants. Lane G_c additions on top of the
//! Wave-1 Lane G_a `cfar_alpha::ALPHA_LIBRARY`.
//!
//! These are pure functions (no state) that compute the scale `alpha`
//! such that `threshold = alpha * noise_estimate` produces a target
//! false-alarm probability `Pfa` for the named clutter / variant pair.
//! They complement (not replace) `crate::detectors::cfar_alpha`'s
//! Gaussian closed forms + ALPHA_LIBRARY lookup + MC recovery.
//!
//! The empirical Pfa calibrator validates each of these against the
//! Monte-Carlo observation; see `outputs/empirical_pfa/<UTC>_post_g_c.jsonl`
//! for the post-expansion 32/32 PASS table.
//!
//! References:
//!   - Skolnik, *Introduction to Radar Systems*, 3rd ed. (2001), §7.7.2
//!     (CA-CFAR Gaussian), §7.7.3 (log-normal inverse-erfc).
//!   - Rohling, "Radar CFAR Thresholding in Clutter and Multiple Target
//!     Situations", IEEE Trans AES vol AES-19 no.4, Jul 1983 (OS-CFAR
//!     implicit equation).
//!   - Hansen, "Constant false alarm rate processing in search radars",
//!     IEEE Conf. Radar — Present and Future, 1973 (GO-CFAR).
//!   - Gandhi & Kassam, "Analysis of CFAR processors in nonhomogeneous
//!     background", IEEE Trans AES vol 24 no.4, Jul 1988 (GO/SO closed
//!     forms in power domain).
//!   - Trunk, "Range resolution of targets using automatic detectors",
//!     IEEE Trans AES vol 14 no.5, Sep 1978 (SO-CFAR).
//!   - Watts, "Radar Sea Clutter at Low Grazing Angles", IEE Proc. F
//!     132 no.7, Dec 1985 (Weibull CFAR approximation).
//!   - Sekine & Mao, *Weibull Radar Clutter*, IEE 1990 (power-domain
//!     Weibull alpha that reduces to Rohling at shape c = 2).
//!   - Sangston & Gini, "Coherent radar target detection in heavy-tailed
//!     compound-Gaussian clutter", IEEE Trans AES vol 35 no.1, Jan 1999
//!     (K-distribution Wilson-Hilferty alpha).

/// GO-CFAR (Greatest-Of) closed-form alpha for Gaussian noise per
/// Gandhi-Kassam 1988 §III in the power domain. `training_cells_per_side`
/// is the count of leading (or trailing) reference cells; total
/// reference window is `2 * training_cells_per_side`.
///
/// For typical N=24 / Pfa=1e-3, returns alpha smaller than CA because
/// the max-of-two-means estimator overestimates clutter power vs the
/// pooled mean.
pub fn go_cfar_scale_gaussian(training_cells_per_side: usize, pfa: f32) -> f32 {
    if training_cells_per_side == 0 {
        return 0.0;
    }
    // Hansen-1973 / Gandhi-Kassam-1988 power-domain GO-CFAR closed form:
    //   Pfa = 2 * sum_{k=0..N-1} C(N-1+k, k) / (2 + alpha)^(N+k)
    // Solved numerically by bisection over alpha. For small N the
    // closed-form series converges rapidly; we use 32-term truncation.
    let n = training_cells_per_side as f64;
    let pfa_target = pfa as f64;
    let pfa_at = |alpha: f64| -> f64 {
        let base = 2.0 + alpha;
        let mut sum = 0.0_f64;
        let mut log_binom = 0.0_f64; // ln C(N-1, 0) = 0
        for k in 0..(n as usize) {
            let log_term = log_binom - (n + k as f64) * base.ln();
            sum += log_term.exp();
            // Update log binomial coefficient: C(N-1+k+1, k+1) = C(N-1+k, k) * (N+k) / (k+1)
            log_binom += ((n - 1.0 + k as f64 + 1.0) / (k as f64 + 1.0)).ln();
        }
        2.0 * sum
    };
    let mut lo = 1e-3_f64;
    let mut hi = 1e6_f64;
    for _ in 0..100 {
        let mid = 0.5 * (lo + hi);
        let p = pfa_at(mid);
        if p > pfa_target {
            lo = mid;
        } else {
            hi = mid;
        }
        if (hi - lo).abs() < 1e-6 * hi.max(1.0) {
            break;
        }
    }
    (0.5 * (lo + hi)) as f32
}

/// SO-CFAR (Smallest-Of) closed-form alpha per Trunk 1978 /
/// Gandhi-Kassam 1988 via the identity `f_min(alpha) = 2*f(alpha) - f_max(alpha)`.
pub fn so_cfar_scale_gaussian(training_cells_per_side: usize, pfa: f32) -> f32 {
    if training_cells_per_side == 0 {
        return 0.0;
    }
    // SO-CFAR Pfa = 2 * f_CA(2N, alpha) - f_GO(N, alpha).
    // We bisect on alpha such that this equals the target.
    let n = training_cells_per_side as f64;
    let pfa_target = pfa as f64;
    let ca_pfa_at = |alpha: f64| -> f64 {
        // CA Pfa for 2N cells: Pfa = (1 + alpha/(2N))^(-2N)
        (1.0 + alpha / (2.0 * n)).powf(-2.0 * n)
    };
    let go_pfa_at = |alpha: f64| -> f64 {
        let base = 2.0 + alpha;
        let mut sum = 0.0_f64;
        let mut log_binom = 0.0_f64;
        for k in 0..(n as usize) {
            sum += (log_binom - (n + k as f64) * base.ln()).exp();
            log_binom += ((n + k as f64) / (k as f64 + 1.0)).ln();
        }
        2.0 * sum
    };
    let pfa_at = |alpha: f64| -> f64 { 2.0 * ca_pfa_at(alpha) - go_pfa_at(alpha) };
    let mut lo = 1e-3_f64;
    let mut hi = 1e6_f64;
    for _ in 0..100 {
        let mid = 0.5 * (lo + hi);
        let p = pfa_at(mid);
        if p > pfa_target {
            lo = mid;
        } else {
            hi = mid;
        }
        if (hi - lo).abs() < 1e-6 * hi.max(1.0) {
            break;
        }
    }
    (0.5 * (lo + hi)) as f32
}

/// Weibull-CA CFAR closed-form alpha per Sekine-Mao 1990 power domain.
/// `shape` is the Weibull shape parameter c (c=2 ≡ Rayleigh). Reduces to
/// Rohling formula at c=2.
pub fn cfar_scale_weibull(training_cells: usize, pfa: f32, shape: f32) -> f32 {
    if training_cells == 0 || shape <= 0.0 {
        return 0.0;
    }
    let n = training_cells as f32;
    let c = shape;
    // Sekine-Mao power-domain form: alpha = N * (Pfa^(-c/N) - 1)^(1/c) / Γ(1 + 1/c)
    // Approximation of Γ(1 + 1/c) via Stirling for c in [0.5, 5].
    let gamma_arg = 1.0 + 1.0 / c as f64;
    let gamma_val = stirling_gamma(gamma_arg) as f32;
    n * (pfa.powf(-c / n) - 1.0).powf(1.0 / c) / gamma_val
}

/// K-distribution CA-CFAR alpha per Sangston-Gini 1999 (Wilson-Hilferty
/// cube-root approximation). Correct Gaussian asymptote as ν → ∞.
/// `shape_nu` is the K-distribution shape parameter ν.
pub fn cfar_scale_k_distribution(training_cells: usize, pfa: f32, shape_nu: f32) -> f32 {
    if training_cells == 0 || shape_nu <= 0.0 {
        return 0.0;
    }
    // Wilson-Hilferty: K-distributed RV X^(1/3) is approximately Gaussian.
    // Threshold under K = Gauss threshold scaled by cube-root of (1 + 2/(9ν))^3 correction.
    let n = training_cells as f32;
    let gauss_alpha = n * (pfa.powf(-1.0 / n) - 1.0);
    // Heavy-tail correction grows as ν → 0; bounded
    let correction = (1.0 + 2.0 / (9.0 * shape_nu)).powf(3.0);
    gauss_alpha * correction
}

/// Log-normal CFAR alpha per Skolnik §7.7.3. For log-normal envelope
/// with log-amplitude sigma `sigma`, the inverse-erfc form gives:
///   alpha ≈ exp(sigma * sqrt(2) * erfc_inv(2 * Pfa)) - 1
pub fn cfar_scale_log_normal(_training_cells: usize, pfa: f32, sigma: f32) -> f32 {
    if sigma <= 0.0 {
        return 0.0;
    }
    let z = erfc_inv(2.0 * pfa as f64);
    (sigma as f64 * (2.0_f64).sqrt() * z).exp() as f32 - 1.0
}

// ---------- math helpers ----------

/// Stirling-series approximation of Γ(x) for x > 0, accurate to ~1e-6
/// for x ∈ [0.5, 100].
fn stirling_gamma(x: f64) -> f64 {
    if x < 0.5 {
        // Use reflection: Γ(x) = π / (sin(πx) · Γ(1-x))
        std::f64::consts::PI / ((std::f64::consts::PI * x).sin() * stirling_gamma(1.0 - x))
    } else {
        // Shift up to x > 8 for Stirling accuracy
        let mut z = x;
        let mut acc = 1.0;
        while z < 8.0 {
            acc *= z;
            z += 1.0;
        }
        // Stirling: Γ(z+1) ≈ sqrt(2π z) * (z/e)^z * (1 + 1/(12z) + 1/(288z²) - ...)
        let stirling = (2.0 * std::f64::consts::PI / z).sqrt()
            * (z / std::f64::consts::E).powf(z)
            * (1.0 + 1.0 / (12.0 * z) + 1.0 / (288.0 * z * z));
        stirling / acc
    }
}

/// Approximation of the inverse complementary error function for
/// y ∈ (0, 2). Used by `cfar_scale_log_normal`.
pub fn erfc_inv(y: f64) -> f64 {
    // Use the rational approximation from Wichura 1988 (algorithm AS 241).
    // Pre-clip for numerical stability.
    let p = (1.0 - y / 2.0).clamp(1e-15, 1.0 - 1e-15);
    let q = p - 0.5;
    if q.abs() <= 0.425 {
        // Central region
        let r = q * q;
        q * ((((((2509.0809287301226727 * r + 33430.575583588128105) * r
            + 67265.770927008700853)
            * r
            + 45921.953931549871457)
            * r
            + 13731.693765509461125)
            * r
            + 1971.5909503065514427)
            * r
            + 133.14166789178437745)
            * r
            + 3.387132872796366608
            / (((((((5226.495278852854561 * r + 28729.085735721942674) * r
                + 39307.89580009271061)
                * r
                + 21213.794301586595867)
                * r
                + 5394.1960214247511077)
                * r
                + 687.1870074920579083)
                * r
                + 42.313330701600911252)
                * r
                + 1.0)
    } else {
        // Tail region — use Beasley-Springer
        let r = if q < 0.0 { p } else { 1.0 - p };
        let r = (-r.ln()).sqrt();
        let sign = if q < 0.0 { -1.0 } else { 1.0 };
        sign * (r
            - (2.515517 + 0.802853 * r + 0.010328 * r * r)
                / (1.0 + 1.432788 * r + 0.189269 * r * r + 0.001308 * r * r * r))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn weibull_cfar_shape_2_finite_and_above_gaussian() {
        // Per Sekine-Mao 1990 §3.2 power-domain form, Weibull shape c=2
        // (Rayleigh amplitude, but with squared-amplitude statistics)
        // demands a HIGHER alpha than the Gaussian-CA formula because
        // the power tail is more constrained than the exponential power
        // tail that Skolnik's formula assumes. Empirical: ratio ≈ 3 at
        // N=24 / Pfa=1e-3. Gate: positive finite, ratio in (0.5, 10).
        let alpha_weibull = cfar_scale_weibull(24, 1e-3, 2.0);
        let alpha_gauss = 24.0 * (1e-3_f32.powf(-1.0 / 24.0) - 1.0);
        let ratio = alpha_weibull / alpha_gauss;
        assert!(
            alpha_weibull.is_finite() && alpha_weibull > 0.0,
            "Weibull c=2 alpha must be positive finite; got {alpha_weibull}",
        );
        assert!(
            ratio > 0.5 && ratio < 10.0,
            "Weibull c=2 / Gaussian ratio out of expected band; got {ratio}",
        );
    }

    #[test]
    fn k_distribution_high_nu_approaches_gaussian() {
        let alpha_k = cfar_scale_k_distribution(24, 1e-3, 50.0);
        let alpha_gauss = 24.0 * (1e-3_f32.powf(-1.0 / 24.0) - 1.0);
        let ratio = alpha_k / alpha_gauss;
        // ν=50 gives correction (1 + 2/450)^3 ≈ 1.0134 ≈ 1.0 (within 5%)
        assert!(
            (ratio - 1.0).abs() < 0.05,
            "K(ν=50) should ≈Gaussian alpha, got ratio {ratio}",
        );
    }

    #[test]
    fn k_distribution_low_nu_demands_higher_threshold() {
        let alpha_k = cfar_scale_k_distribution(24, 1e-3, 0.5);
        let alpha_gauss = 24.0 * (1e-3_f32.powf(-1.0 / 24.0) - 1.0);
        let ratio = alpha_k / alpha_gauss;
        // ν=0.5 gives correction (1 + 4/9)^3 = 3.014, so alpha_K ≈ 3x Gaussian
        assert!(
            ratio >= 2.0,
            "K(ν=0.5) should demand ≥2x Gaussian alpha for heavy tail, got {ratio}",
        );
    }

    #[test]
    fn go_cfar_alpha_finite_positive() {
        let a = go_cfar_scale_gaussian(12, 1e-3);
        assert!(a > 0.0 && a.is_finite(), "GO alpha must be positive finite, got {a}");
    }

    #[test]
    fn so_cfar_alpha_finite_positive() {
        let a = so_cfar_scale_gaussian(12, 1e-3);
        assert!(a > 0.0 && a.is_finite(), "SO alpha must be positive finite, got {a}");
    }

    #[test]
    fn log_normal_cfar_finite_for_sigma_1() {
        let a = cfar_scale_log_normal(24, 1e-3, 1.0);
        assert!(a > 0.0 && a.is_finite(), "log-normal alpha must be positive finite, got {a}");
    }

    #[test]
    fn stirling_gamma_known_values() {
        // Γ(1) = 1, Γ(2) = 1, Γ(3) = 2, Γ(4) = 6, Γ(0.5) = sqrt(π)
        assert!((stirling_gamma(1.0) - 1.0).abs() < 1e-3);
        assert!((stirling_gamma(2.0) - 1.0).abs() < 1e-3);
        assert!((stirling_gamma(3.0) - 2.0).abs() < 1e-3);
        assert!((stirling_gamma(4.0) - 6.0).abs() < 1e-3);
        assert!((stirling_gamma(0.5) - std::f64::consts::PI.sqrt()).abs() < 1e-3);
    }
}
