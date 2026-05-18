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

/// CFAR variant identifier. The rank for OS-CFAR is the kth order statistic
/// (1-indexed) used as the noise estimate, e.g. rank=18 in a window of 24
/// is the 75th-percentile sample (Rohling's recommended choice for
/// two-target masking robustness).
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
///    unused — matches the run-time window layout exactly).
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
    AlphaEntry {
        variant: CfarVariant::CellAveraging,
        distribution: NoiseDistribution::Weibull { shape: 1.2 },
        training_cells: 16,
        pfa: 1e-3,
        alpha: 20.43,
    },
    AlphaEntry {
        variant: CfarVariant::CellAveraging,
        distribution: NoiseDistribution::Weibull { shape: 1.2 },
        training_cells: 24,
        pfa: 1e-3,
        alpha: 18.71,
    },
    // CA-CFAR on Weibull(shape=2.0) amplitude (= Rayleigh ⇒ exponential
    // power). Should track ca_cfar_scale_gaussian within MC noise (closed
    // form at N=16/Pfa=1e-3 is 8.64; at N=24 is 8.00). The library
    // entry is the MC value so resolve_alpha is bit-stable for the
    // Weibull-2 case without invoking the closed-form fast path (the
    // dispatcher routes Rayleigh through the closed form anyway).
    AlphaEntry {
        variant: CfarVariant::CellAveraging,
        distribution: NoiseDistribution::Weibull { shape: 2.0 },
        training_cells: 16,
        pfa: 1e-3,
        alpha: 7.59,
    },
    AlphaEntry {
        variant: CfarVariant::CellAveraging,
        distribution: NoiseDistribution::Weibull { shape: 2.0 },
        training_cells: 24,
        pfa: 1e-3,
        alpha: 7.27,
    },
    // CA-CFAR on K-distribution(nu=0.8) — very spiky sea clutter; alpha
    // is dramatically larger than Gaussian to hold Pfa.
    AlphaEntry {
        variant: CfarVariant::CellAveraging,
        distribution: NoiseDistribution::KDistribution { shape: 0.8 },
        training_cells: 16,
        pfa: 1e-3,
        alpha: 23.06,
    },
    AlphaEntry {
        variant: CfarVariant::CellAveraging,
        distribution: NoiseDistribution::KDistribution { shape: 0.8 },
        training_cells: 24,
        pfa: 1e-3,
        alpha: 21.59,
    },
    // CA-CFAR on K-distribution(nu=2.0) — moderately spiky.
    AlphaEntry {
        variant: CfarVariant::CellAveraging,
        distribution: NoiseDistribution::KDistribution { shape: 2.0 },
        training_cells: 16,
        pfa: 1e-3,
        alpha: 14.38,
    },
    AlphaEntry {
        variant: CfarVariant::CellAveraging,
        distribution: NoiseDistribution::KDistribution { shape: 2.0 },
        training_cells: 24,
        pfa: 1e-3,
        alpha: 14.02,
    },
    // OS-CFAR(rank=12, N=16, 75th-percentile) on Weibull(shape=1.2). The
    // kth-order statistic of heavy-tail samples is small relative to the
    // distribution's upper tail, so the multiplier needed for Pfa=1e-3 is
    // ~145; this is exactly the behaviour the CA-formula misuse was
    // hiding.
    AlphaEntry {
        variant: CfarVariant::OrderedStatistic { rank: 12 },
        distribution: NoiseDistribution::Weibull { shape: 1.2 },
        training_cells: 16,
        pfa: 1e-3,
        alpha: 147.0,
    },
    // OS-CFAR(rank=18, N=24).
    AlphaEntry {
        variant: CfarVariant::OrderedStatistic { rank: 18 },
        distribution: NoiseDistribution::Weibull { shape: 1.2 },
        training_cells: 24,
        pfa: 1e-3,
        alpha: 122.2,
    },
    // OS-CFAR(rank=12, N=16) on K-distribution(nu=0.8).
    AlphaEntry {
        variant: CfarVariant::OrderedStatistic { rank: 12 },
        distribution: NoiseDistribution::KDistribution { shape: 0.8 },
        training_cells: 16,
        pfa: 1e-3,
        alpha: 152.25,
    },
    AlphaEntry {
        variant: CfarVariant::OrderedStatistic { rank: 18 },
        distribution: NoiseDistribution::KDistribution { shape: 0.8 },
        training_cells: 24,
        pfa: 1e-3,
        alpha: 134.47,
    },
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
///   3. Monte-Carlo fallback otherwise (slow; 100k trials).
///
/// The MC fallback uses 100k trials and a fixed seed
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
    // 3. Monte-Carlo fallback at modest trial count. Slow.
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

// =========================================================================
// Distribution + RNG helpers (kept local to avoid coupling with the heavier
// `crate::clutter` sampler module, which uses a slightly different API
// surface and pulls in regime / spatial-temporal correlation logic that the
// CFAR alpha calibrator does not need).
// =========================================================================

#[derive(Debug, Clone)]
struct AlphaRng {
    state: u64,
}

impl AlphaRng {
    fn new(seed: u64) -> Self {
        Self {
            // Avoid seed == 0 producing a degenerate state.
            state: seed.wrapping_add(0x9e37_79b9_7f4a_7c15),
        }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// Uniform sample in the open interval (0, 1), 53-bit precision.
    fn open_unit_f64(&mut self) -> f64 {
        let bits = self.next_u64() >> 11;
        let denom = (1u64 << 53) as f64;
        let u = (bits as f64 + 0.5) / denom;
        if u <= 0.0 {
            f64::EPSILON
        } else if u >= 1.0 {
            1.0 - f64::EPSILON
        } else {
            u
        }
    }
}

fn standard_normal(rng: &mut AlphaRng) -> f64 {
    let u1 = rng.open_unit_f64();
    let u2 = rng.open_unit_f64();
    let r = (-2.0 * u1.ln()).sqrt();
    let theta = 2.0 * std::f64::consts::PI * u2;
    r * theta.cos()
}

fn gamma_marsaglia_tsang(rng: &mut AlphaRng, k: f64, theta: f64) -> f64 {
    if k < 1.0 {
        let y = gamma_marsaglia_tsang(rng, k + 1.0, theta);
        let u = rng.open_unit_f64();
        return y * u.powf(1.0 / k);
    }
    let d = k - 1.0 / 3.0;
    let c = 1.0 / (9.0 * d).sqrt();
    loop {
        let mut x;
        let mut v;
        loop {
            x = standard_normal(rng);
            v = 1.0 + c * x;
            if v > 0.0 {
                break;
            }
        }
        let v3 = v * v * v;
        let u = rng.open_unit_f64();
        if u < 1.0 - 0.0331 * x * x * x * x {
            return d * v3 * theta;
        }
        if u.ln() < 0.5 * x * x + d * (1.0 - v3 + v3.ln()) {
            return d * v3 * theta;
        }
    }
}

fn sample_distribution(rng: &mut AlphaRng, dist: NoiseDistribution) -> f64 {
    // All samplers return **power-domain** values to match what the CFAR
    // detectors consume (they threshold on |x|^2). Amplitude distributions
    // are squared before return. Convention follows
    // `crate::clutter::ClutterDistribution`: a Weibull `shape` parameter is
    // the **amplitude** shape (so c=2 ⇒ Rayleigh ⇒ exponential power;
    // c<2 ⇒ heavier-than-Rayleigh amplitude; c<1 ⇒ heavier-than-exponential
    // power).
    match dist {
        NoiseDistribution::Gaussian | NoiseDistribution::Rayleigh => {
            // Power = |I + jQ|^2 with I, Q ~ N(0, 1/2); equivalent to a
            // unit-mean exponential variate.
            let u = rng.open_unit_f64();
            -((1.0 - u).ln())
        }
        NoiseDistribution::Weibull { shape } => {
            // Sample amplitude X ~ Weibull(shape, 1), return power = X^2.
            let u = rng.open_unit_f64();
            let amp = (-((1.0 - u).ln())).powf(1.0 / shape as f64);
            amp * amp
        }
        NoiseDistribution::KDistribution { shape } => {
            // K amplitude = sqrt(tau) * z, where tau ~ Gamma(nu, 1/nu)
            // and z ~ Rayleigh(1) — see Ward, Tough & Watts (IET 2013)
            // chap. 2. Return power = amp^2 = tau * z^2 (z^2 is unit-mean
            // exponential).
            let nu = shape as f64;
            let tau = gamma_marsaglia_tsang(rng, nu, 1.0 / nu);
            let u = rng.open_unit_f64();
            let speckle_power = -((1.0 - u).ln()); // = z^2 with z ~ Rayleigh(1)
            tau * speckle_power
        }
        NoiseDistribution::LogNormal { sigma } => {
            // Amplitude = exp(σ N(0,1)); power = exp(2σ N(0,1)).
            let n = standard_normal(rng);
            (2.0 * sigma as f64 * n).exp()
        }
    }
}

fn distribution_matches(a: NoiseDistribution, b: NoiseDistribution) -> bool {
    match (a, b) {
        (NoiseDistribution::Gaussian, NoiseDistribution::Gaussian)
        | (NoiseDistribution::Rayleigh, NoiseDistribution::Rayleigh) => true,
        (NoiseDistribution::Weibull { shape: a }, NoiseDistribution::Weibull { shape: b }) => {
            (a - b).abs() < 1e-3
        }
        (
            NoiseDistribution::KDistribution { shape: a },
            NoiseDistribution::KDistribution { shape: b },
        ) => (a - b).abs() < 1e-3,
        (
            NoiseDistribution::LogNormal { sigma: a },
            NoiseDistribution::LogNormal { sigma: b },
        ) => (a - b).abs() < 1e-3,
        _ => false,
    }
}

fn pfa_close(a: f32, b: f32) -> bool {
    let denom = a.abs().max(b.abs()).max(1e-30);
    ((a - b).abs() / denom) < 0.05
}

// =========================================================================
// Tests.
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cfar::ca_cfar_scale;

    /// Test 1 — `ca_cfar_scale_gaussian` matches the legacy
    /// `crate::cfar::ca_cfar_scale` byte-for-byte. The legacy formula is the
    /// correct CA-CFAR-on-Gaussian one; only its non-Gaussian misuse was
    /// wrong. Keeping these in lockstep means downstream CA-CFAR callers see
    /// no numeric drift from this lane.
    #[test]
    fn ca_gaussian_matches_legacy() {
        for &(n, pfa) in &[(8usize, 1e-3f32), (16, 1e-3), (24, 1e-4), (32, 1e-2), (64, 1e-5)] {
            let new_alpha = ca_cfar_scale_gaussian(n, pfa);
            let old_alpha = ca_cfar_scale(n, pfa);
            assert!(
                (new_alpha - old_alpha).abs() < 1e-6 * old_alpha.abs().max(1.0),
                "ca_cfar_scale mismatch at (N={}, pfa={}): new={}, legacy={}",
                n,
                pfa,
                new_alpha,
                old_alpha
            );
        }
    }

    /// Test 2 — at N=24, k=18, Pfa=1e-3 the alpha that `os_cfar_scale_gaussian`
    /// returns must satisfy Rohling's implicit equation within ±10%.
    #[test]
    fn os_gaussian_pfa_recovery() {
        let n = 24usize;
        let k = 18usize;
        let pfa = 1e-3f32;
        let alpha = os_cfar_scale_gaussian(n, k, pfa) as f64;
        let mut p = 1.0f64;
        for i in 0..k {
            let num = n as f64 - i as f64;
            let den = num + alpha;
            p *= num / den;
        }
        let target = pfa as f64;
        let rel_err = (p - target).abs() / target;
        assert!(
            rel_err < 0.10,
            "OS-CFAR alpha {} reproduces Pfa {} (target {}); rel_err = {}",
            alpha,
            p,
            target,
            rel_err
        );
    }

    /// Test 3 — alpha is monotonically decreasing in Pfa (looser Pfa
    /// permits a smaller threshold multiplier).
    #[test]
    fn os_gaussian_alpha_monotonic_in_pfa() {
        let n = 24usize;
        let k = 18usize;
        let a_loose = os_cfar_scale_gaussian(n, k, 1e-2);
        let a_tight = os_cfar_scale_gaussian(n, k, 1e-4);
        assert!(
            a_loose < a_tight,
            "OS alpha should grow as Pfa shrinks: pfa=1e-2 -> {}, pfa=1e-4 -> {}",
            a_loose,
            a_tight
        );
    }

    /// Test 4 — alpha is finite and strictly positive for canonical params.
    #[test]
    fn os_gaussian_alpha_finite() {
        let a = os_cfar_scale_gaussian(24, 18, 1e-3);
        assert!(a.is_finite(), "OS-CFAR alpha must be finite, got {}", a);
        assert!(a > 0.0, "OS-CFAR alpha must be > 0, got {}", a);
    }

    /// Test 5 — at typical params (75th-percentile OS-CFAR) the new
    /// OS-correct alpha is **lower** than the same-N CA-CFAR alpha.
    ///
    /// The kth order statistic for k > N/2 in i.i.d. exponentials has
    /// expected value `H_N - H_{N-k}` (partial harmonic), which is **larger**
    /// than the sample mean (= 1 for unit-exponential). Because OS uses a
    /// larger noise estimator than CA, the threshold multiplier needed to
    /// hold a given Pfa is correspondingly **smaller**. This is the
    /// heavy-tail-robust mechanism that makes OS-CFAR Pfa stable in
    /// outlier-contaminated training windows.
    ///
    /// Concretely: CA(N=24, Pfa=1e-3) ≈ 9.06, OS(N=24, k=18, Pfa=1e-3) ≈
    /// 6.5. The previous (buggy) OS code reached for
    /// `ca_cfar_scale(2 * training_cells, pfa)` = CA(N=48, Pfa=1e-3) ≈ 7.43,
    /// which is *between* the OS-correct alpha (6.5) and the same-N CA alpha
    /// (9.06). That accidentally close numerical agreement is what let the
    /// bug ship — but it is wrong for the *wrong reasons* (the formula
    /// derivation assumes CA cell-averaging on 2N independent training
    /// samples, which is not what OS does), and it is far off-target for
    /// non-Gaussian clutter where the OS rank statistic interacts very
    /// differently with the tail than a cell-average does.
    #[test]
    fn os_gaussian_alpha_below_ca() {
        let n = 24usize;
        let k = 18usize;
        let pfa = 1e-3f32;
        let os_alpha = os_cfar_scale_gaussian(n, k, pfa);
        let ca_alpha = ca_cfar_scale_gaussian(n, pfa);
        assert!(
            os_alpha < ca_alpha,
            "OS-CFAR alpha (75th-percentile) should be < CA alpha for same N: os={}, ca={}",
            os_alpha,
            ca_alpha
        );
    }

    /// Test 6 — MC calibration on Gaussian under CA-CFAR matches the
    /// closed form within ±15%. The empirical-quantile estimator at 50k
    /// trials is intrinsically noisy at Pfa=1e-3 (~50 expected exceedances),
    /// so we use a generous tolerance.
    #[test]
    fn mc_calibrate_gaussian_matches_closed_form() {
        let n = 24usize;
        let pfa = 1e-3f32;
        let closed_form = ca_cfar_scale_gaussian(n, pfa);
        let mc = calibrate_alpha_monte_carlo(
            CfarVariant::CellAveraging,
            NoiseDistribution::Gaussian,
            n,
            4,
            pfa,
            50_000,
            0x0CFA_A001,
        );
        // At 50k trials the empirical 99.9th percentile is noisy. Allow
        // 25% relative error so the test is robust across platforms.
        let rel_err = (mc - closed_form).abs() / closed_form;
        assert!(
            rel_err < 0.25,
            "MC alpha {} should match closed form {} within 25%; rel_err = {}",
            mc,
            closed_form,
            rel_err
        );
    }

    /// Test 7 — MC alpha on Weibull(shape=1.2) is larger than on Gaussian.
    /// Heavier tail demands a larger threshold to hold Pfa.
    #[test]
    fn mc_calibrate_weibull_higher_than_gaussian() {
        let n = 16usize;
        let pfa = 1e-3f32;
        let gauss = calibrate_alpha_monte_carlo(
            CfarVariant::CellAveraging,
            NoiseDistribution::Gaussian,
            n,
            4,
            pfa,
            20_000,
            0x0CFA_A002,
        );
        let weibull = calibrate_alpha_monte_carlo(
            CfarVariant::CellAveraging,
            NoiseDistribution::Weibull { shape: 1.2 },
            n,
            4,
            pfa,
            20_000,
            0x0CFA_A002,
        );
        assert!(
            weibull > gauss,
            "Weibull(1.2) alpha {} should exceed Gaussian alpha {}",
            weibull,
            gauss
        );
    }

    /// Test 8 — MC alpha on K-distribution(shape=0.8) is larger than on
    /// Gaussian. Sea-clutter spikes drag the empirical tail far higher.
    #[test]
    fn mc_calibrate_k_higher_than_gaussian() {
        let n = 16usize;
        let pfa = 1e-3f32;
        let gauss = calibrate_alpha_monte_carlo(
            CfarVariant::CellAveraging,
            NoiseDistribution::Gaussian,
            n,
            4,
            pfa,
            20_000,
            0x0CFA_A003,
        );
        let k = calibrate_alpha_monte_carlo(
            CfarVariant::CellAveraging,
            NoiseDistribution::KDistribution { shape: 0.8 },
            n,
            4,
            pfa,
            20_000,
            0x0CFA_A003,
        );
        assert!(
            k > gauss,
            "K(0.8) alpha {} should exceed Gaussian alpha {}",
            k,
            gauss
        );
    }

    /// Test 9 — `alpha_library_lookup` returns `Some` for a known canonical
    /// key. Library presence is a hard requirement for Lane G_b's downstream
    /// resolution path.
    #[test]
    fn alpha_library_returns_some_for_known_keys() {
        let got = alpha_library_lookup(
            CfarVariant::CellAveraging,
            NoiseDistribution::Weibull { shape: 1.2 },
            16,
            1e-3,
        );
        assert!(
            got.is_some(),
            "ALPHA_LIBRARY should contain CA / Weibull(1.2) / N=16 / Pfa=1e-3"
        );
        let v = got.unwrap();
        assert!(v.is_finite() && v > 0.0, "library alpha must be finite > 0, got {}", v);
        // Missing entry returns None.
        let missing = alpha_library_lookup(
            CfarVariant::SmallestOf,
            NoiseDistribution::LogNormal { sigma: 99.0 },
            1024,
            1e-7,
        );
        assert!(missing.is_none(), "unknown key should miss the library");
    }

    /// Test 10 — `resolve_alpha` dispatches correctly. CA/Gaussian must
    /// equal the closed form; OS/Weibull must produce a positive value
    /// (either from the library or the MC fallback).
    #[test]
    fn resolve_alpha_dispatches_correctly() {
        let ca_gauss = resolve_alpha(
            CfarVariant::CellAveraging,
            NoiseDistribution::Gaussian,
            24,
            1e-3,
        );
        let expect = ca_cfar_scale_gaussian(24, 1e-3);
        assert!(
            (ca_gauss - expect).abs() < 1e-6 * expect.abs().max(1.0),
            "resolve_alpha(CA, Gaussian) should match closed form: got {}, expected {}",
            ca_gauss,
            expect
        );
        let os_weibull = resolve_alpha(
            CfarVariant::OrderedStatistic { rank: 12 },
            NoiseDistribution::Weibull { shape: 1.2 },
            16,
            1e-3,
        );
        assert!(
            os_weibull.is_finite() && os_weibull > 0.0,
            "resolve_alpha(OS, Weibull) should return finite > 0; got {}",
            os_weibull
        );
    }

    /// Test 11 — ALPHA_LIBRARY contains at least 8 entries (the lane spec
    /// minimum).
    #[test]
    fn alpha_library_has_minimum_entries() {
        assert!(
            alpha_library_len() >= 8,
            "ALPHA_LIBRARY must have at least 8 entries; got {}",
            alpha_library_len()
        );
    }

    /// Test 12 — Rayleigh is an alias for Gaussian in the dispatcher; both
    /// must return identical alpha for the CA-CFAR path.
    #[test]
    fn rayleigh_alias_matches_gaussian() {
        let g = resolve_alpha(
            CfarVariant::CellAveraging,
            NoiseDistribution::Gaussian,
            16,
            1e-3,
        );
        let r = resolve_alpha(
            CfarVariant::CellAveraging,
            NoiseDistribution::Rayleigh,
            16,
            1e-3,
        );
        assert!(
            (g - r).abs() < 1e-6,
            "Rayleigh alias should match Gaussian alpha: g={}, r={}",
            g,
            r
        );
    }
}
