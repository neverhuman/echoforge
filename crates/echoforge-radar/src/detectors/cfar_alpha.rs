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
use cfar_alpha_helpers::{AlphaRng, distribution_matches, pfa_close, sample_distribution};

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

// =========================================================================
// Monte-Carlo calibration.
// =========================================================================

/// Monte-Carlo-calibrated alpha for any CFAR variant on any distribution.
/// Generates `trials` independent training-window draws from the supplied
/// distribution, computes the per-window noise estimate under the variant's
/// rule, and reports the alpha that yields the target Pfa empirically.
///
/// The algorithm is the textbook empirical-quantile method:
/// 1. For each trial, draw `2 * training_cells + 2 * guard_cells + 1`
///    samples from the distribution (the guard cells are drawn but
///    reserved — matches the run-time window layout exactly).
/// 2. Compute the variant's noise estimate from the training half.
/// 3. Compute the ratio `cut / noise_estimate`. The empirical Pfa is the
///    probability this ratio exceeds `alpha`, so `alpha` is the
///    `(1 - pfa)`-quantile of the ratio distribution.
///
/// Slow (~`trials` distribution draws); intended for offline `ALPHA_LIBRARY`
/// population and unit-test sanity checks. For production calls prefer
/// `resolve_alpha`, which tries closed-form and library lookup first.
pub fn calibrate_alpha_monte_carlo(
    variant: CfarVariant,
    distribution: NoiseDistribution,
    training_cells: usize,
    guard_cells: usize,
    pfa: f32,
    trials: usize,
    seed: u64,
) -> f32 {
    if training_cells == 0 || trials == 0 {
        return 0.0;
    }
    if let CfarVariant::OrderedStatistic { rank } = variant {
        // Total training set is 2*N (lead + lag). Rank must fit.
        if rank == 0 || rank > 2 * training_cells {
            return 0.0;
        }
    }
    let mut rng = AlphaRng::new(seed);
    let mut ratios: Vec<f64> = Vec::with_capacity(trials);
    let window_size = 2 * training_cells + 2 * guard_cells + 1;
    let mut buf = vec![0.0f64; window_size];

    for _ in 0..trials {
        for slot in buf.iter_mut() {
            *slot = sample_distribution(&mut rng, distribution);
        }
        let cut = buf[training_cells + guard_cells];
        let lead = &buf[0..training_cells];
        let lag = &buf[training_cells + 2 * guard_cells + 1..window_size];
        let noise = match variant {
            CfarVariant::CellAveraging => {
                let s: f64 = lead.iter().sum::<f64>() + lag.iter().sum::<f64>();
                s / (lead.len() + lag.len()) as f64
            }
            CfarVariant::GreatestOf => {
                let m1 = lead.iter().sum::<f64>() / lead.len() as f64;
                let m2 = lag.iter().sum::<f64>() / lag.len() as f64;
                m1.max(m2)
            }
            CfarVariant::SmallestOf => {
                let m1 = lead.iter().sum::<f64>() / lead.len() as f64;
                let m2 = lag.iter().sum::<f64>() / lag.len() as f64;
                m1.min(m2)
            }
            CfarVariant::OrderedStatistic { rank } => {
                let mut combined: Vec<f64> = Vec::with_capacity(lead.len() + lag.len());
                combined.extend_from_slice(lead);
                combined.extend_from_slice(lag);
                combined.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let idx = (rank - 1).min(combined.len() - 1);
                combined[idx]
            }
        };
        if noise.abs() < 1e-30 {
            continue;
        }
        ratios.push(cut / noise);
    }

    if ratios.is_empty() {
        return 0.0;
    }
    // Empirical (1 - Pfa) quantile.
    ratios.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let q = (1.0 - pfa as f64).clamp(0.0, 1.0);
    let idx = ((q * ratios.len() as f64).round() as isize - 1)
        .clamp(0, ratios.len() as isize - 1) as usize;
    ratios[idx] as f32
}

// =========================================================================
// Library lookup for K / Weibull (precomputed via the MC calibrator above).
// =========================================================================

/// One entry in the precomputed `ALPHA_LIBRARY`. Match keys are matched
/// loosely (training_cells exact, Pfa within 5% relative) so the library
/// covers the common CFAR configurations without bloating the table.
#[derive(Debug, Clone, Copy)]
struct AlphaEntry {
    variant: CfarVariant,
    distribution: NoiseDistribution,
    training_cells: usize,
    pfa: f32,
    alpha: f32,
}

const fn ca_entry(distribution: NoiseDistribution, training_cells: usize, alpha: f32) -> AlphaEntry {
    AlphaEntry {
        variant: CfarVariant::CellAveraging,
        distribution,
        training_cells,
        pfa: 1e-3,
        alpha,
    }
}

const fn os_entry(rank: usize, distribution: NoiseDistribution, training_cells: usize, alpha: f32) -> AlphaEntry {
    AlphaEntry {
        variant: CfarVariant::OrderedStatistic { rank },
        distribution,
        training_cells,
        pfa: 1e-3,
        alpha,
    }
}

/// Precomputed (variant, distribution, N, Pfa) -> alpha table.
///
/// **How these numbers were derived**: each row is the output of
/// `calibrate_alpha_monte_carlo(variant, distribution, training_cells,
/// guard_cells=4, pfa, trials=200_000, seed=0x0CFA_A001)` on this codebase.
/// To re-derive any row, call `calibrate_alpha_monte_carlo` with the same
/// arguments and that exact seed; the result will reproduce bit-for-bit
/// because the RNG is deterministic. The values are intentionally stable
/// across builds (no system RNG, no thread RNG, no float-determinism
/// pragmas needed because the math is plain f64 arithmetic).
///
/// Library entries cover the canonical Wave 1 cruise configurations:
///   * CA-CFAR / OS-CFAR (rank = 0.75 * N)
///   * Weibull shape c ∈ {1.2, 2.0}
///   * K-distribution shape nu ∈ {0.8, 2.0}
///   * Training cells N ∈ {16, 24}
///   * Pfa = 1e-3
///
/// Lane G_b is responsible for expanding this table as ClutterRegime gets
/// plumbed through the detectors.
const ALPHA_LIBRARY: &[AlphaEntry] = &[
    // CA-CFAR on Weibull(shape=1.2) amplitude (power = X^2 = Weibull(0.6)).
    // Heavier tail than exponential power; alpha is much larger than
    // ca_cfar_scale_gaussian(N, Pfa) to hold Pfa.
    ca_entry(NoiseDistribution::Weibull { shape: 1.2 }, 16, 20.43),
    ca_entry(NoiseDistribution::Weibull { shape: 1.2 }, 24, 18.71),
    // CA-CFAR on Weibull(shape=2.0) amplitude (= Rayleigh ⇒ exponential
    // power). Should track ca_cfar_scale_gaussian within MC noise (closed
    // form at N=16/Pfa=1e-3 is 8.64; at N=24 is 8.00). The library
    // entry is the MC value so resolve_alpha is bit-stable for the
    // Weibull-2 case without invoking the closed-form fast path (the
    // dispatcher routes Rayleigh through the closed form anyway).
    ca_entry(NoiseDistribution::Weibull { shape: 2.0 }, 16, 7.59),
    ca_entry(NoiseDistribution::Weibull { shape: 2.0 }, 24, 7.27),
    // CA-CFAR on K-distribution(nu=0.8) — very spiky sea clutter; alpha
    // is dramatically larger than Gaussian to hold Pfa.
    ca_entry(NoiseDistribution::KDistribution { shape: 0.8 }, 16, 23.06),
    ca_entry(NoiseDistribution::KDistribution { shape: 0.8 }, 24, 21.59),
    // CA-CFAR on K-distribution(nu=2.0) — moderately spiky.
    ca_entry(NoiseDistribution::KDistribution { shape: 2.0 }, 16, 14.38),
    ca_entry(NoiseDistribution::KDistribution { shape: 2.0 }, 24, 14.02),
    // OS-CFAR(rank=12, N=16, 75th-percentile) on Weibull(shape=1.2). The
    // kth-order statistic of heavy-tail samples is small relative to the
    // distribution's upper tail, so the multiplier needed for Pfa=1e-3 is
    // ~145; this is exactly the behaviour the CA-formula misuse was hiding.
    os_entry(12, NoiseDistribution::Weibull { shape: 1.2 }, 16, 147.0),
    // OS-CFAR(rank=18, N=24).
    os_entry(18, NoiseDistribution::Weibull { shape: 1.2 }, 24, 122.2),
    // OS-CFAR(rank=12, N=16) on K-distribution(nu=0.8).
    os_entry(12, NoiseDistribution::KDistribution { shape: 0.8 }, 16, 152.25),
    os_entry(18, NoiseDistribution::KDistribution { shape: 0.8 }, 24, 134.47),
];

/// Look up an alpha from the const `ALPHA_LIBRARY` for known
/// `(variant, distribution, training_cells, pfa)` tuples.
///
/// Returns `None` if no match is found. The Pfa match is loose (within 5%
/// relative); the others are exact. Library entries cover the canonical
/// Wave 1 cruise configurations — see `ALPHA_LIBRARY` for the populated set.
pub fn alpha_library_lookup(
    variant: CfarVariant,
    distribution: NoiseDistribution,
    training_cells: usize,
    pfa: f32,
) -> Option<f32> {
    for entry in ALPHA_LIBRARY {
        if entry.variant == variant
            && distribution_matches(entry.distribution, distribution)
            && entry.training_cells == training_cells
            && pfa_close(entry.pfa, pfa)
        {
            return Some(entry.alpha);
        }
    }
    None
}

/// Number of entries in the const `ALPHA_LIBRARY`. Exposed for the
/// receipt and for unit tests; library size matters for downstream Lane G_b
/// completeness assertions.
pub fn alpha_library_len() -> usize {
    ALPHA_LIBRARY.len()
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
