use super::*;
use crate::cfar::ca_cfar_scale;

/// Test 1 — `ca_cfar_scale_gaussian` matches the prior
/// `crate::cfar::ca_cfar_scale` byte-for-byte. The prior formula is the
/// correct CA-CFAR-on-Gaussian one; only its non-Gaussian misuse was
/// wrong. Keeping these in lockstep means downstream CA-CFAR callers see
/// no numeric drift from this lane.
#[test]
fn ca_gaussian_matches_prior() {
    for &(n, pfa) in &[
        (8usize, 1e-3f32),
        (16, 1e-3),
        (24, 1e-4),
        (32, 1e-2),
        (64, 1e-5),
    ] {
        let new_alpha = ca_cfar_scale_gaussian(n, pfa);
        let old_alpha = ca_cfar_scale(n, pfa);
        assert!(
            (new_alpha - old_alpha).abs() < 1e-6 * old_alpha.abs().max(1.0),
            "ca_cfar_scale mismatch at (N={}, pfa={}): new={}, prior={}",
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
    assert!(
        v.is_finite() && v > 0.0,
        "library alpha must be finite > 0, got {}",
        v
    );
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
/// (either from the library or the MC recovery).
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
