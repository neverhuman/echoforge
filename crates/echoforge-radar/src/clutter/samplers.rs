use super::regime::{ClutterDistribution, ClutterRegime};
use super::SplitMix64;

/// Sample one Weibull-distributed value. Inverse-CDF method:
/// `x = scale * (-ln(1 - u))^(1/shape)` for `u ~ U(0, 1)`. This is the
/// standard inverse-CDF Weibull sampler; see e.g. Knuth TAOCP vol 2
/// sec 3.4.1 or Devroye, "Non-Uniform Random Variate Generation" (1986).
///
/// Determinism: the supplied `seed` drives a `SplitMix64` instance owned
/// by this call; no thread_rng / system entropy is consulted.
pub fn sample_weibull(shape: f64, scale: f64, seed: u64) -> f64 {
    debug_assert!(shape > 0.0, "Weibull shape must be > 0");
    debug_assert!(scale > 0.0, "Weibull scale must be > 0");
    let mut rng = SplitMix64::new(seed);
    weibull_from_rng(&mut rng, shape, scale)
}

/// Sample one K-distributed value using the textbook product form
/// `K = sqrt(tau) * z`, where `tau ~ Gamma(nu, 1/nu)` is the slowly
/// varying texture and `z ~ Rayleigh(1)` is the fast speckle.
/// See Ward, Tough & Watts (IET 2013) chap. 2 for the derivation.
///
/// Determinism: as for `sample_weibull`.
pub fn sample_k_distribution(shape: f64, scale: f64, seed: u64) -> f64 {
    debug_assert!(shape > 0.0, "K-distribution shape (nu) must be > 0");
    debug_assert!(scale > 0.0, "K-distribution scale must be > 0");
    let mut rng = SplitMix64::new(seed);
    k_distribution_from_rng(&mut rng, shape, scale)
}

/// Sample one log-normal value. Returns `exp(N(mean_log, std_log))`.
/// Determinism: as for `sample_weibull`.
pub fn sample_log_normal(mean_log: f64, std_log: f64, seed: u64) -> f64 {
    debug_assert!(std_log >= 0.0, "log-normal std_log must be >= 0");
    let mut rng = SplitMix64::new(seed);
    log_normal_from_rng(&mut rng, mean_log, std_log)
}

/// Dispatch sampler — returns one amplitude sample drawn from the supplied
/// `ClutterDistribution`. Determinism: as for `sample_weibull`.
pub fn sample_clutter_amplitude(dist: &ClutterDistribution, seed: u64) -> f64 {
    let mut rng = SplitMix64::new(seed);
    sample_amplitude_from_rng(&mut rng, dist)
}

/// Generate a sequence of clutter samples with spatial AR(1) correlation
/// across range bins (inner axis) and temporal AR(1) correlation across
/// pulses (outer axis). Returns a row-major flat vec of length
/// `n_pulses * n_range_bins`, with the row index = pulse and column
/// index = range bin.
///
/// The mixing rule for AR(1) is `x_t = rho * x_{t-1} + sqrt(1 - rho^2) * w_t`
/// where `w_t` is a fresh draw from the regime's amplitude distribution.
/// This is the textbook AR(1) one-step recursion; it preserves the
/// stationary variance of the input innovations (Box, Jenkins, Reinsel,
/// "Time Series Analysis: Forecasting and Control", chap. 3).
///
/// Determinism: a single `SplitMix64` instance is seeded by `seed` and
/// drives every draw, so the output is bit-stable for a given
/// (regime, n_range_bins, n_pulses, seed) tuple.
pub fn generate_clutter_sequence(
    regime: &ClutterRegime,
    n_range_bins: usize,
    n_pulses: usize,
    seed: u64,
) -> Vec<f32> {
    let total = n_pulses.saturating_mul(n_range_bins);
    let mut out = vec![0.0f32; total];
    if total == 0 {
        return out;
    }
    let rho_s = regime.spatial_correlation.clamp(0.0, 1.0);
    let rho_t = regime.temporal_correlation.clamp(0.0, 1.0);
    let beta_s = (1.0 - rho_s * rho_s).max(0.0).sqrt();
    let beta_t = (1.0 - rho_t * rho_t).max(0.0).sqrt();
    let mut rng = SplitMix64::new(seed);

    // Previous-pulse buffer for the temporal AR(1) step. None on the first pulse.
    let mut prev_pulse: Option<Vec<f64>> = None;

    for p in 0..n_pulses {
        let mut row = vec![0.0f64; n_range_bins];
        let mut prev_bin = 0.0f64;
        for r in 0..n_range_bins {
            // Fresh innovation drawn from the regime's amplitude distribution.
            let w = sample_amplitude_from_rng(&mut rng, &regime.distribution);
            // Spatial AR(1) across range bins (within this pulse).
            let spatial = if r == 0 {
                w
            } else {
                rho_s * prev_bin + beta_s * w
            };
            // Temporal AR(1) across pulses (within this range bin).
            let mixed = if let Some(ref prev) = prev_pulse {
                rho_t * prev[r] + beta_t * spatial
            } else {
                spatial
            };
            row[r] = mixed;
            prev_bin = mixed;
            out[p * n_range_bins + r] = mixed as f32;
        }
        prev_pulse = Some(row);
    }
    out
}

// -------------------------------------------------------------------------
// Internal RNG-driven samplers. Sharing one RNG instance across many
// draws (as `generate_clutter_sequence` does) keeps the sequence
// deterministic and statistically independent without needing per-call
// re-seeding.
// -------------------------------------------------------------------------

fn weibull_from_rng(rng: &mut SplitMix64, shape: f64, scale: f64) -> f64 {
    // `open_unit_f64` returns u in (0, 1), so (1 - u) is also in (0, 1)
    // and ln(1 - u) < 0 is finite. Negating gives a strictly positive
    // argument for the fractional power.
    let u = rng.open_unit_f64();
    scale * (-((1.0 - u).ln())).powf(1.0 / shape)
}

/// Standard normal via Box-Muller. Two uniforms in, one standard normal
/// out (the second normal is discarded to keep call counts predictable
/// and the sequence reproducible for a fixed seed regardless of how
/// callers consume the stream).
fn standard_normal(rng: &mut SplitMix64) -> f64 {
    let u1 = rng.open_unit_f64();
    let u2 = rng.open_unit_f64();
    let r = (-2.0 * u1.ln()).sqrt();
    let theta = 2.0 * std::f64::consts::PI * u2;
    r * theta.cos()
}

/// Gamma(shape=k, scale=theta) via Marsaglia & Tsang's method
/// ("A Simple Method for Generating Gamma Variables", ACM TOMS 26(3),
/// 2000). Handles k >= 1 directly; for k < 1 uses the Boost-equivalent
/// boost trick `Gamma(k) = Gamma(k+1) * U^(1/k)`.
fn gamma_marsaglia_tsang(rng: &mut SplitMix64, k: f64, theta: f64) -> f64 {
    if k < 1.0 {
        // Use the k -> k+1 boost: X = Y * U^(1/k), Y ~ Gamma(k+1, theta).
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
        // Squeeze test, then full acceptance test (Marsaglia & Tsang).
        if u < 1.0 - 0.0331 * x * x * x * x {
            return d * v3 * theta;
        }
        if u.ln() < 0.5 * x * x + d * (1.0 - v3 + v3.ln()) {
            return d * v3 * theta;
        }
    }
}

fn k_distribution_from_rng(rng: &mut SplitMix64, shape: f64, scale: f64) -> f64 {
    // Texture: Gamma(nu, 1/nu) -> unit-mean Gamma. Speckle: Rayleigh(1)
    // = Weibull(shape=2, scale=1). Product form per Ward, Tough &
    // Watts (IET 2013) chap. 2.
    let tau = gamma_marsaglia_tsang(rng, shape, 1.0 / shape);
    let z = weibull_from_rng(rng, 2.0, 1.0);
    scale * tau.sqrt() * z
}

fn log_normal_from_rng(rng: &mut SplitMix64, mean_log: f64, std_log: f64) -> f64 {
    let n = standard_normal(rng);
    (mean_log + std_log * n).exp()
}

pub(super) fn sample_amplitude_from_rng(rng: &mut SplitMix64, dist: &ClutterDistribution) -> f64 {
    match *dist {
        ClutterDistribution::Rayleigh => weibull_from_rng(rng, 2.0, 1.0),
        ClutterDistribution::Weibull { shape, scale } => weibull_from_rng(rng, shape, scale),
        ClutterDistribution::KDistribution { shape, scale } => {
            k_distribution_from_rng(rng, shape, scale)
        }
        ClutterDistribution::LogNormal { mean_log, std_log } => {
            log_normal_from_rng(rng, mean_log, std_log)
        }
    }
}
