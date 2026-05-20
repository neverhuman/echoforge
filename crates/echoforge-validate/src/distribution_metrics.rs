//! 1-D distribution distance metrics used by the V4 gate.
//!
//! Both functions operate on SORTED ascending sample slices. The caller
//! is responsible for sorting; [`crate::tier_measured_anchored::evaluate_v4_gate`]
//! sorts through its private `compute_distance` helper.

/// 1-D Wasserstein-1 (Earth Mover's) distance between two empirical
/// distributions given as sorted sample vectors.
///
/// Standard rectangular-CDF formulation: for two ECDFs `F` and `G`,
/// `W_1(F, G) = ∫ |F(x) - G(x)| dx`. The implementation here uses the
/// equivalent merged-quantile form which is `O(n log n)` rather than
/// requiring an integration grid.
///
/// Both inputs MUST be sorted ascending. Empty inputs return `f64::INFINITY`
/// (treated as "no comparison possible"); the gate evaluator filters
/// these out via `V4MetricThresholds::min_samples_per_distribution`.
pub fn wasserstein_1d_sorted(a: &[f64], b: &[f64]) -> f64 {
    if a.is_empty() || b.is_empty() {
        return f64::INFINITY;
    }
    let na = a.len();
    let nb = b.len();
    let mut i = 0usize;
    let mut j = 0usize;
    let mut total = 0.0f64;
    let mut prev = if a[0] < b[0] { a[0] } else { b[0] };
    let mut cdf_a;
    let mut cdf_b;
    loop {
        let (next, advance_a, advance_b) = match (a.get(i), b.get(j)) {
            (Some(&av), Some(&bv)) => {
                if av < bv {
                    (av, true, false)
                } else if bv < av {
                    (bv, false, true)
                } else {
                    (av, true, true)
                }
            }
            (Some(&av), None) => (av, true, false),
            (None, Some(&bv)) => (bv, false, true),
            (None, None) => break,
        };
        let width = next - prev;
        cdf_a = i as f64 / na as f64;
        cdf_b = j as f64 / nb as f64;
        total += (cdf_a - cdf_b).abs() * width;
        if advance_a {
            i += 1;
        }
        if advance_b {
            j += 1;
        }
        prev = next;
    }
    total
}

/// 1-D Kolmogorov-Smirnov distance between two empirical distributions
/// given as sorted sample vectors.
///
/// Returns `sup |F(x) - G(x)|` over the merged sample set, in `[0, 1]`.
/// Both inputs MUST be sorted ascending. Empty inputs return `1.0`
/// (maximum possible KS distance) so the gate's tolerance check fails
/// gracefully rather than silently accepting an empty observation.
pub fn ks_distance_1d_sorted(a: &[f64], b: &[f64]) -> f64 {
    if a.is_empty() || b.is_empty() {
        return 1.0;
    }
    let na = a.len();
    let nb = b.len();
    let mut i = 0usize;
    let mut j = 0usize;
    let mut sup = 0.0f64;
    while i < na || j < nb {
        let (advance_a, advance_b) = match (a.get(i), b.get(j)) {
            (Some(&av), Some(&bv)) => {
                if av < bv { (true, false) }
                else if bv < av { (false, true) }
                else { (true, true) }
            }
            (Some(_), None) => (true, false),
            (None, Some(_)) => (false, true),
            (None, None) => break,
        };
        if advance_a { i += 1; }
        if advance_b { j += 1; }
        let diff = (i as f64 / na as f64 - j as f64 / nb as f64).abs();
        if diff > sup { sup = diff; }
    }
    sup
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wasserstein_zero_for_identical() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        assert!(wasserstein_1d_sorted(&a, &a).abs() < 1e-12);
    }

    #[test]
    fn wasserstein_shift_equals_translation() {
        let a = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let b = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert!((wasserstein_1d_sorted(&a, &b) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn wasserstein_empty_returns_infinity() {
        assert_eq!(wasserstein_1d_sorted(&[], &[1.0, 2.0]), f64::INFINITY);
    }

    #[test]
    fn ks_zero_for_identical() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        assert!(ks_distance_1d_sorted(&a, &a).abs() < 1e-12);
    }

    #[test]
    fn ks_one_for_disjoint() {
        let a = vec![0.0, 0.0, 0.0, 0.0];
        let b = vec![10.0, 10.0, 10.0, 10.0];
        assert!((ks_distance_1d_sorted(&a, &b) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn ks_empty_returns_one() {
        assert_eq!(ks_distance_1d_sorted(&[], &[1.0, 2.0]), 1.0);
    }
}
