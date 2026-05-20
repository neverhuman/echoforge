//! CFAR threshold scale factors per (variant, distribution) pair.
//!
//! Lane G_a (Wave 1 Expert Credibility Sweep) — replaces the prior single
//! `crate::cfar::ca_cfar_scale` formula that was being mis-applied to every
//! CFAR variant. The classical CA-CFAR Gaussian formula
//!     alpha_CA = N * (Pfa^(-1/N) - 1)
//! is only correct for Cell-Averaging CFAR on exponentially-distributed
//! power (Rayleigh amplitude / Gaussian I+Q noise). For Ordered-Statistic
//! CFAR (OS-CFAR) the closed form is the Rohling 1983 implicit equation
//!     Pfa = prod_{i=0}^{k-1} (N - i) / (N - i + alpha)
//! solved numerically for alpha. For K-distributed, Weibull and log-normal
//! clutter no closed form exists; we use Monte-Carlo calibration and bake
//! the results into a small `ALPHA_LIBRARY` lookup table.
//!
//! Citations:
//!   * Skolnik, *Introduction to Radar Systems*, 3rd ed. (McGraw-Hill 2001),
//!     section 7.7.2 — CA-CFAR Gaussian formula.
//!   * Rohling, "Radar CFAR Thresholding in Clutter and Multiple Target
//!     Situations", IEEE Trans. AES, vol. AES-19 no. 4, July 1983 — OS-CFAR
//!     implicit Pfa equation.
//!   * Ward, Tough & Watts, *Sea Clutter: Scattering, the K Distribution
//!     and Radar Performance*, 2nd ed. (IET 2013) — K-distribution context
//!     for ALPHA_LIBRARY.
//!
//! Strict-open posture: no measured-truth claims; library values derive
//! from Monte-Carlo calibration on simulated K/Weibull distributions cited
//! to the standard radar handbooks above.

#![allow(clippy::excessive_precision)]

#[path = "cfar_alpha_helpers.rs"]
mod cfar_alpha_helpers;

#[path = "cfar_alpha_library.rs"]
mod cfar_alpha_library;
pub use cfar_alpha_library::{calibrate_alpha_monte_carlo, alpha_library_lookup, alpha_library_len};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CfarVariant {
    /// Cell-Averaging CFAR.
    CellAveraging,
    /// Ordered-Statistic CFAR using the `rank`-th sorted training sample
    /// (1-indexed; valid range `1..=N`).
    OrderedStatistic { rank: usize },
    /// Greatest-Of CFAR (max of leading/lagging window means).
    GreatestOf,
    /// Smallest-Of CFAR (min of leading/lagging window means).
    SmallestOf,
}

/// Noise / clutter amplitude distribution under which the CFAR alpha is
/// calibrated. This is a CFAR-local view of clutter intentionally distinct
/// from `crate::clutter::ClutterDistribution`: it carries only the parameters
/// the alpha calculation actually consumes, and includes the
/// engineering-convenient `Rayleigh` shortcut that is mathematically identical
/// to `Weibull{shape: 2.0}` but lets callers express intent.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NoiseDistribution {
    /// Power-domain exponential (= amplitude Rayleigh = Gaussian I+Q).
    /// The CFAR closed-forms in Skolnik §7.7.2 and Rohling 1983 are
    /// derived under this assumption.
    Gaussian,
    /// Amplitude Rayleigh; in power domain this is identical to Gaussian.
    /// Provided so callers can express "I'm passing amplitude samples" vs
    /// "I'm passing power samples" without changing alpha resolution.
    Rayleigh,
    /// Weibull amplitude with the given shape parameter. `shape == 2.0` is
    /// Rayleigh; `shape == 1.0` is exponential amplitude (heavier tail);
    /// `shape < 1.0` is very spiky.
    Weibull { shape: f32 },
    /// K-distribution amplitude with the given shape parameter `nu`.
    /// `nu < 1` is very spiky sea clutter; `nu >> 10` approaches Rayleigh.
    KDistribution { shape: f32 },
    /// Log-normal amplitude; `sigma` is the standard deviation of the
    /// underlying normal (mean of the normal is taken as zero so the median
    /// of the amplitude is 1).
    LogNormal { sigma: f32 },
}

// =========================================================================
// Closed-form alphas.
// =========================================================================

/// Closed-form CA-CFAR threshold scale for Gaussian / Rayleigh-amplitude
/// noise.
///
/// `alpha = N * (Pfa^(-1/N) - 1)` (Skolnik §7.7.2). Returns `0.0` when
/// `training_cells == 0` so callers can keep their existing guard branches.
pub fn ca_cfar_scale_gaussian(training_cells: usize, pfa: f32) -> f32 {
    if training_cells == 0 {
        return 0.0;
    }
    let n = training_cells as f32;
    n * (pfa.powf(-1.0 / n) - 1.0)
}

/// Closed-form OS-CFAR threshold scale for Gaussian / Rayleigh-amplitude
/// noise. Solves the Rohling 1983 implicit equation
///
/// ```text
/// Pfa = prod_{i=0}^{k-1} (N - i) / (N - i + alpha)
/// ```
///
/// by bisection on `alpha` in `[1e-3, 1e6]`. Pfa is monotonically decreasing
/// in `alpha`, so bisection is well-defined. Returns `0.0` when the input
/// `(N, k)` pair is invalid (`N == 0`, `k == 0`, or `k > N`).
pub fn os_cfar_scale_gaussian(training_cells: usize, rank: usize, pfa: f32) -> f32 {
    if training_cells == 0 || rank == 0 || rank > training_cells {
        return 0.0;
    }
    let pfa_target = pfa as f64;
    let n = training_cells as f64;
    let pfa_at = |alpha: f64| -> f64 {
        let mut p = 1.0f64;
        for i in 0..rank {
            let num = n - i as f64;
            let den = num + alpha;
            p *= num / den;
        }
        p
    };
    let mut lo = 1e-3f64;
    let mut hi = 1e6f64;
    // Pfa is monotonically decreasing in alpha. Bisect: if Pfa(mid) >
    // target then alpha needs to grow (move lo up); else shrink (move hi
    // down). Cap at 100 iterations for safety; convergence is ~50 iters.
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

/// Best-effort alpha resolution:
///   1. Closed-form if Gaussian / Rayleigh + (CA or OS).
///   2. Library lookup for K-distribution / Weibull / log-normal at
///      canonical configurations.
///   3. Monte-Carlo recovery otherwise (slow; 100k trials).
///
/// The MC recovery uses 100k trials and a fixed seed
/// (`0x0CFA_A001`) so repeated calls are bit-stable. Callers that need
/// faster resolution should precompute and add a row to `ALPHA_LIBRARY`.
pub fn resolve_alpha(
    variant: CfarVariant,
    distribution: NoiseDistribution,
    training_cells: usize,
    pfa: f32,
) -> f32 {
    // 1. Closed-form fast path: Gaussian / Rayleigh + (CA or OS).
    match (variant, distribution) {
        (
            CfarVariant::CellAveraging,
            NoiseDistribution::Gaussian | NoiseDistribution::Rayleigh,
        ) => return ca_cfar_scale_gaussian(training_cells, pfa),
        (
            CfarVariant::OrderedStatistic { rank },
            NoiseDistribution::Gaussian | NoiseDistribution::Rayleigh,
        ) => return os_cfar_scale_gaussian(training_cells, rank, pfa),
        _ => {}
    }
    // 2. Library lookup for K/Weibull/log-normal at canonical configs.
    if let Some(alpha) = alpha_library_lookup(variant, distribution, training_cells, pfa) {
        return alpha;
    }
    // 3. Monte-Carlo recovery at modest trial count. Slow.
    calibrate_alpha_monte_carlo(
        variant,
        distribution,
        training_cells,
        4, // guard cells; matches the OS-CFAR Wave 1 default
        pfa,
        100_000,
        0x0CFA_A001,
    )
}

#[cfg(test)]
#[path = "cfar_alpha_tests.rs"]
mod tests;
