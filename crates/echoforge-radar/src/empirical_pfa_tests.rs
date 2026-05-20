use super::*;

#[test]
fn wilson_ci_centred_proportion() {
    // 500 / 1000 = 0.5 should give a CI roughly centred on 0.5 with
    // ~3% half-width.
    let (lo, hi) = wilson_ci_95(500, 1000);
    assert!(lo < 0.5 && hi > 0.5, "CI should bracket 0.5: [{lo}, {hi}]");
    assert!(
        hi - lo < 0.10,
        "CI half-width should be small: [{lo}, {hi}]"
    );
}

#[test]
fn wilson_ci_zero_trials_is_uninformative() {
    let (lo, hi) = wilson_ci_95(0, 0);
    assert_eq!(lo, 0.0);
    assert_eq!(hi, 1.0);
}

#[test]
fn wilson_ci_zero_successes_lower_bound_zero() {
    // Wilson CI on (0, N) gives a centre + half-width that subtract to
    // ~ -eps due to floating-point round-off; we clamp the lower bound
    // to >= 0 in wilson_ci_95, so the returned value should be either
    // exactly 0.0 or a non-negative ~ eps value.
    let (lo, _hi) = wilson_ci_95(0, 1000);
    assert!(
        lo >= 0.0 && lo < 1e-12,
        "lower bound should clamp to ~0, got {lo}"
    );
}

#[test]
fn render_markdown_has_header_and_one_row_per_observation() {
    // Synthetic minimal observation set; just verify table structure.
    let observations = calibrate_standard_table(1_000, 0xABCD);
    let md = render_markdown(&observations);
    assert!(
        md.starts_with("| Regime"),
        "markdown should start with header"
    );
    // 2 header rows + N data rows.
    let line_count = md.lines().count();
    assert_eq!(line_count, observations.len() + 2);
}

#[test]
fn render_jsonl_one_line_per_observation() {
    let observations = calibrate_standard_table(1_000, 0xABCD);
    let jsonl = render_jsonl(&observations);
    let line_count = jsonl.lines().count();
    assert_eq!(line_count, observations.len());
}

#[test]
fn standard_table_has_thirty_two_combos() {
    let table = standard_table();
    assert_eq!(table.len(), 32);
}

#[test]
fn measure_pfa_deterministic_for_fixed_seed() {
    let regime = ClutterRegime::for_terrain(TerrainClass::OpenSky, 30.0);
    let trial = PfaTrial {
        clutter_regime: regime,
        cfar_variant: CfarVariant::CellAveraging,
        training_cells: 24,
        guard_cells: 4,
        nominal_pfa: 1e-2,
        trials: 10_000,
        seed: 0xDEAD_BEEF,
    };
    let a = measure_pfa(&trial);
    let b = measure_pfa(&trial);
    assert_eq!(a.observed_count, b.observed_count);
    assert_eq!(a.trials, b.trials);
}
