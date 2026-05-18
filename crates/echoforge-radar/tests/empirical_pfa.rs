//! Empirical Pfa calibration test -- the credibility artifact for Lane G_b
//! (Wave 3, item C12) of the Radar Expert Credibility Sweep.
//!
//! The full C12 gate is IGNORED BY DEFAULT (long-running, >=10^7 trials per
//! combo, ~30 min on single thread). Run explicitly with:
//!     rtk cargo test -p echoforge-radar --release --test empirical_pfa -- --ignored --nocapture
//!
//! The fast smoke test (`empirical_pfa_smoke_gaussian_ca_cfar`) is wired into
//! the default test run and verifies the calibrator end-to-end at modest
//! trial count: Gaussian-amplitude clutter + CA-CFAR @ Pfa=1e-2 must produce
//! an observed Pfa within ~2x of nominal at 1e5 trials.

use echoforge_radar::clutter::{ClutterRegime, TerrainClass};
use echoforge_radar::detectors::cfar_alpha::CfarVariant;
use echoforge_radar::empirical_pfa::{calibrate_standard_table, measure_pfa, PfaTrial};

/// Fast sanity check: the Gaussian / Rayleigh-amplitude regime plus CA-CFAR
/// at a generous Pfa (1e-2) must give an observed Pfa in the same order of
/// magnitude as nominal at 1e5 trials. This is the short-circuit test that
/// verifies the calibrator wiring works without the 30-minute full run.
#[test]
fn empirical_pfa_smoke_gaussian_ca_cfar() {
    let regime = ClutterRegime::for_terrain(TerrainClass::OpenSky, 30.0);
    let trial = PfaTrial {
        clutter_regime: regime,
        cfar_variant: CfarVariant::CellAveraging,
        training_cells: 24,
        guard_cells: 4,
        // Generous so 1e5 trials see ~1000 false alarms; with that count the
        // Wilson half-width is ~6% of the mean, so a 2x band easily contains
        // the truth.
        nominal_pfa: 1e-2,
        trials: 100_000,
        seed: 0x12345,
    };
    let obs = measure_pfa(&trial);
    assert!(
        obs.ratio_observed_to_nominal > 0.5 && obs.ratio_observed_to_nominal < 2.0,
        "expected observed/nominal in [0.5, 2.0], got {} (observed {}, nominal {})",
        obs.ratio_observed_to_nominal,
        obs.observed_pfa,
        trial.nominal_pfa,
    );
    // Sanity: the 95% Wilson CI should bracket the observed value by
    // definition; just check it has non-zero width.
    assert!(obs.wilson_ci_high > obs.wilson_ci_low);
    assert!(obs.trials == 100_000);
}

/// Same smoke check but with the OrderedStatistic variant, to exercise the
/// k-th order statistic code path under the same Gaussian / Rayleigh regime
/// (where cfar_alpha has the Rohling 1983 closed-form alpha; observed should
/// closely track nominal).
#[test]
fn empirical_pfa_smoke_gaussian_os_cfar() {
    let regime = ClutterRegime::for_terrain(TerrainClass::OpenSky, 30.0);
    let trial = PfaTrial {
        clutter_regime: regime,
        cfar_variant: CfarVariant::OrderedStatistic { rank: 18 },
        training_cells: 24,
        guard_cells: 4,
        nominal_pfa: 1e-2,
        trials: 100_000,
        seed: 0x67890,
    };
    let obs = measure_pfa(&trial);
    assert!(
        obs.ratio_observed_to_nominal > 0.5 && obs.ratio_observed_to_nominal < 2.0,
        "OS-CFAR expected observed/nominal in [0.5, 2.0], got {} (observed {}, nominal {})",
        obs.ratio_observed_to_nominal,
        obs.observed_pfa,
        trial.nominal_pfa,
    );
}

/// C12 production gate. Generates 10^7 cells per combo (32 combos total),
/// counts false alarms, compares nominal vs observed via 95% Wilson CI.
/// Any combo whose nominal lies outside the observed CI fails the test.
///
/// IGNORED BY DEFAULT (long-running). Run with:
///   rtk cargo test -p echoforge-radar --release --test empirical_pfa -- --ignored --nocapture
#[ignore = "long-running C12 gate; run explicitly with --ignored"]
#[test]
fn c12_empirical_pfa_within_95ci_for_standard_table() {
    let observations = calibrate_standard_table(10_000_000, 0xC12_A001);
    let failures: Vec<_> = observations.iter().filter(|o| !o.passes).collect();
    for obs in &failures {
        eprintln!(
            "C12 FAIL: {} / {:?} / N={} / nominal {:.2e} observed {:.2e} CI [{:.2e}, {:.2e}] ratio {:.2}x",
            obs.clutter_regime_name,
            obs.cfar_variant,
            obs.training_cells,
            obs.nominal_pfa,
            obs.observed_pfa,
            obs.wilson_ci_low,
            obs.wilson_ci_high,
            obs.ratio_observed_to_nominal,
        );
    }
    assert!(
        failures.is_empty(),
        "{} of {} CFAR/regime combos lie outside 95% CI of nominal Pfa",
        failures.len(),
        observations.len()
    );
}
