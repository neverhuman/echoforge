use super::*;

/// Standard atmosphere, radar 10 m / target 100 m: standard
/// horizon ≈ 54 km. The Skolnik formula gives
/// `4.12 · (√10 + √100) = 4.12 · (3.162 + 10) = 4.12 · 13.162 ≈
/// 54.23 km`. Tolerance 1 km.
#[test]
fn standard_horizon_10m_100m() {
    let d = standard_horizon_range_m(10.0, 100.0);
    let expected_km = 54.23;
    let d_km = d / 1000.0;
    assert!(
        (d_km - expected_km).abs() < 1.0,
        "standard horizon {d_km:.2} km off from {expected_km} km"
    );
}

/// MIT Lincoln Lab radar course ch. 4 worked example: radar 100 m /
/// target 100 m, standard atmosphere ⇒ horizon ≈ 82 km
/// (= 4.12 · (10 + 10)). Tolerance 1 km.
#[test]
fn lincoln_lab_100m_100m_anchor() {
    let d = standard_horizon_range_m(100.0, 100.0);
    let expected_km = 82.4;
    let d_km = d / 1000.0;
    assert!(
        (d_km - expected_km).abs() < 1.0,
        "Lincoln Lab anchor: horizon {d_km:.2} km off from {expected_km} km"
    );
}

/// Surface duct height 50 m, radar 10 m / target 100 m: horizon
/// extends by ≥ 30 % per Lincoln Lab worked examples; target range
/// ≥ ~70 km (54 km × 1.5 = 81 km, well into the published 60-100
/// km surface-duct envelope).
#[test]
fn surface_duct_50m_extends_at_least_30pct() {
    let standard = standard_horizon_range_m(10.0, 100.0);
    let ducted = ducting_horizon_range_m(
        10.0,
        100.0,
        DuctRegime::Surface { height_m: 50.0 },
        3.0,
    );
    let extension = (ducted - standard) / standard;
    assert!(
        extension >= 0.30,
        "expected ≥ 30 % extension under 50 m surface duct, got {:.1} %",
        extension * 100.0
    );
}

/// Spec acceptance gate (test 2 in packet): surface duct extends
/// 3 GHz detection range over sea by ≥ 30 % vs standard atmosphere
/// in moderate-duct conditions (gradient -200 N/km → duct height
/// ~55 m via the classifier, radar 10 m, target 100 m).
#[test]
fn surface_duct_3ghz_30pct_extension_over_sea() {
    use super::super::duct_detection::classify_duct_regime;
    let regime = classify_duct_regime(-200.0, 2.5, 0.5);
    match regime {
        DuctRegime::Surface { .. } => {}
        other => panic!("expected Surface duct for -200 N/km, got {other:?}"),
    }
    let pct = horizon_extension_pct(10.0, 100.0, regime, 3.0);
    assert!(
        pct >= 30.0,
        "moderate-duct horizon extension {pct:.1} % below 30 % gate"
    );
}

/// Surface-duct multiplier saturates at 3.5× even for very thick
/// (200 m+) ducts.
#[test]
fn surface_duct_multiplier_saturates() {
    let standard = standard_horizon_range_m(10.0, 100.0);
    let ducted = ducting_horizon_range_m(
        10.0,
        100.0,
        DuctRegime::Surface { height_m: 500.0 },
        3.0,
    );
    let ratio = ducted / standard;
    assert!(
        (ratio - SURFACE_DUCT_MAX_MULTIPLIER).abs() < 0.01,
        "surface duct multiplier should saturate at {SURFACE_DUCT_MAX_MULTIPLIER}, got {ratio}"
    );
}

/// Evaporation duct at 3 GHz (S-band) — marginal boost (~0 %, no
/// effect because the duct is too thin for the wavelength).
#[test]
fn evaporation_duct_sband_marginal() {
    let pct = horizon_extension_pct(
        10.0,
        100.0,
        DuctRegime::Evaporation { height_m: 18.0 },
        3.0,
    );
    assert!(
        pct < 5.0,
        "expected ~0 % S-band evaporation-duct boost, got {pct:.1} %"
    );
}

/// Evaporation duct at 10 GHz (X-band) — 5-20 % marginal boost.
/// This is the "X-band marginal" half of the spec acceptance gate.
#[test]
fn evaporation_duct_xband_marginal_boost() {
    let pct = horizon_extension_pct(
        10.0,
        100.0,
        DuctRegime::Evaporation { height_m: 18.0 },
        10.0,
    );
    assert!(
        (5.0..=25.0).contains(&pct),
        "X-band evaporation-duct boost {pct:.1} % outside 5-25 % marginal envelope"
    );
}

/// Evaporation duct at 16 GHz (Ku-band) — substantial 30-80 % boost.
/// This is the spec acceptance gate's "Ku substantial" half.
#[test]
fn evaporation_duct_kuband_substantial_boost() {
    let pct = horizon_extension_pct(
        10.0,
        100.0,
        DuctRegime::Evaporation { height_m: 18.0 },
        16.0,
    );
    assert!(
        (30.0..=80.0).contains(&pct),
        "Ku-band evaporation-duct boost {pct:.1} % outside 30-80 % substantial envelope"
    );
}

/// Evaporation duct boost at Ku-band must be strictly larger than
/// at X-band, which in turn must exceed S-band — the frequency
/// dependence is the diagnostic signature.
#[test]
fn evaporation_duct_frequency_ordering() {
    let s = horizon_extension_pct(
        10.0,
        100.0,
        DuctRegime::Evaporation { height_m: 18.0 },
        3.0,
    );
    let x = horizon_extension_pct(
        10.0,
        100.0,
        DuctRegime::Evaporation { height_m: 18.0 },
        10.0,
    );
    let ku = horizon_extension_pct(
        10.0,
        100.0,
        DuctRegime::Evaporation { height_m: 18.0 },
        16.0,
    );
    assert!(
        ku > x && x > s,
        "frequency ordering broken: S={s:.1} X={x:.1} Ku={ku:.1}"
    );
}

/// Elevated duct: above-duct target gets flagged as blocked.
#[test]
fn elevated_duct_blocks_above_duct_target() {
    // Elevated trapping layer at 500-700 m. Surface radar tries to
    // see a target at 1500 m — geometrically above the duct,
    // should be blocked.
    let result = ducting_horizon_with_blocking(
        10.0,
        1500.0,
        DuctRegime::Elevated {
            base_m: 500.0,
            top_m: 700.0,
        },
        3.0,
    );
    assert_eq!(result.visibility, AboveDuctVisibility::AboveDuctBlocked);
    assert!(
        result.range_m > 0.0,
        "horizon range should still be finite under elevated duct"
    );
}

/// Elevated duct: below-duct target is visible (and gets the small
/// below-duct boost).
#[test]
fn elevated_duct_below_duct_target_visible() {
    let result = ducting_horizon_with_blocking(
        10.0,
        100.0,
        DuctRegime::Elevated {
            base_m: 500.0,
            top_m: 700.0,
        },
        3.0,
    );
    assert_eq!(result.visibility, AboveDuctVisibility::Visible);
    // The 1.2× below-duct boost should be visible too.
    let standard = standard_horizon_range_m(10.0, 100.0);
    assert!(
        result.range_m > standard,
        "below-duct elevated case should extend horizon"
    );
}

/// Standard regime: extension percentage is exactly 0.
#[test]
fn standard_regime_zero_extension() {
    let pct = horizon_extension_pct(10.0, 100.0, DuctRegime::None, 3.0);
    assert!(
        pct.abs() < 1e-9,
        "standard regime should give 0 % extension, got {pct}"
    );
}

/// Zero-height radar and target ⇒ zero horizon.
#[test]
fn zero_geometry_zero_horizon() {
    let d = standard_horizon_range_m(0.0, 0.0);
    assert_eq!(d, 0.0);
}

/// Horizon range monotone-increasing in radar height (target fixed).
#[test]
fn horizon_monotone_in_radar_height() {
    let mut prev = -1.0;
    for h_r in [1.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0] {
        let d = standard_horizon_range_m(h_r, 100.0);
        assert!(d > prev, "horizon not monotone at h_r={h_r}");
        prev = d;
    }
}
