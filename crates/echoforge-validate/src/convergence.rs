//! Richardson convergence analysis.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConvergenceReport {
    pub levels: Vec<(f64, f64)>,
    pub p_observed: f64,
    pub extrapolated: f64,
    pub error_bar_db: f64,
    pub pass: bool,
}

/// Run `run` at h0, h0/2, h0/4, ... for `levels` steps, then report observed
/// order of accuracy and Richardson-extrapolated value.
///
/// Pass criterion: `p_observed >= 0.5 * p_expected` AND monotonic |delta|
/// decrease across the three finest levels. `p_expected` must be supplied by
/// the caller via the `expected_order` parameter (added vs. the plan signature
/// so the pass rule is enforceable inside this function).
pub fn richardson<F: Fn(f64) -> f64>(
    run: F,
    h0: f64,
    levels: usize,
    expected_order: f64,
) -> ConvergenceReport {
    assert!(levels >= 3, "richardson requires at least 3 levels");
    let mut series = Vec::with_capacity(levels);
    let mut h = h0;
    for _ in 0..levels {
        let v = run(h);
        series.push((h, v));
        h *= 0.5;
    }
    let (_, f1) = series[levels - 3];
    let (_, f2) = series[levels - 2];
    let (_, f3) = series[levels - 1];
    let num = f1 - f2;
    let den = f2 - f3;
    let p_observed = if den.abs() > 1e-30 && num.abs() > 1e-30 {
        (num / den).abs().log2()
    } else {
        0.0
    };
    let monotonic = num.abs() >= den.abs();
    let extrapolated = f3 + (f3 - f2) / ((2f64).powf(p_observed) - 1.0).max(1e-9);
    let error_bar = (extrapolated - f3).abs();
    let error_bar_db = if extrapolated.abs() > 1e-30 {
        10.0 * (1.0 + error_bar / extrapolated.abs()).log10()
    } else {
        0.0
    };
    let pass = p_observed >= 0.5 * expected_order && monotonic;
    ConvergenceReport {
        levels: series,
        p_observed,
        extrapolated,
        error_bar_db,
        pass,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn synthetic_second_order() {
        // f(h) = 1 + h^2 → p_observed = 2.
        let r = richardson(|h| 1.0 + h.powi(2), 0.1, 4, 2.0);
        assert!((r.p_observed - 2.0).abs() < 0.1, "p={}", r.p_observed);
        assert!(r.pass);
        assert!((r.extrapolated - 1.0).abs() < 1e-3);
    }
}
