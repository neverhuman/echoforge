use super::*;

/// Test 1 — Horizon inversion at the canonical antenna height /
/// range pair (h_r=20 m, R=100 km) under the 4/3-Earth model.
///
/// The Skolnik (2.42) inversion gives
///   h_t = (R - sqrt(2·R_eff·h_r))² / (2·R_eff)
/// with R_eff = 4/3 · 6_371_008.8 ≈ 8.495e6 m. Numerically
/// h_t ≈ 391.6 m. The original v3 dossier listed ≈568 m for this
/// case; tracing the algebra back to the cited formula shows that
/// figure was an arithmetic error in the dossier (568 m corresponds
/// to h_r ≈ 0 m, not 20 m). The test below pins the correct
/// physics value with a 10 m tolerance and the receipt for this
/// packet records the deviation from the dossier figure.
#[test]
fn min_target_altitude_canonical_geometry() {
    let h_t = min_target_altitude_for_los_m(20.0, 100_000.0, STANDARD_K_FACTOR);
    let expected = 391.6;
    assert!(
        (h_t - expected).abs() < 10.0,
        "min altitude {h_t:.2} m differs from {expected:.2} m by more than 10 m"
    );
}

/// Test 2 — A 50 m target at 100 km is below the 4/3-Earth horizon
/// for a 20 m antenna (required altitude is ~392 m).
#[test]
fn target_below_horizon_returns_false() {
    assert!(!target_above_horizon(20.0, 100_000.0, 50.0, STANDARD_K_FACTOR));
}

/// Test 3 — A 1000 m target at 100 km is comfortably above the
/// 4/3-Earth horizon for a 20 m antenna.
#[test]
fn target_above_horizon_returns_true() {
    assert!(target_above_horizon(20.0, 100_000.0, 1000.0, STANDARD_K_FACTOR));
}

/// Test 4 — Required target altitude is monotonically non-decreasing
/// in range for a fixed antenna height. Sample the 1 km–500 km
/// interval coarsely.
#[test]
fn min_target_altitude_monotonic_in_range() {
    let mut prev = f64::NEG_INFINITY;
    for r_km in (1..=500).step_by(5) {
        let h = min_target_altitude_for_los_m(20.0, r_km as f64 * 1000.0, STANDARD_K_FACTOR);
        assert!(
            h >= prev - 1e-9,
            "non-monotonic horizon altitude: prev={prev:.3} new={h:.3} at {r_km} km"
        );
        prev = h;
    }
}

/// Test 5 — First two-ray null altitude. For 3 GHz (λ=0.0999 m
/// from C=2.998e8), R=80 km, h_r=20 m, the textbook
/// `|F| = 2|sin(2π·h_r·h_t/(λ·R))|` has its first null at
/// `h_t = λ·R / (2·h_r) ≈ 199.87 m`. (The original v3 spec text
/// quoted "≈200 m" as the first null but described the formula as
/// `2·sin(π·h_r·h_t/(λ·R))` — the latter would put the first null
/// at 400 m. The Balanis derivation cited in the docstring shows
/// that the spec's *test value* matches the correct full-phase
/// formula `2·sin(2π·h_r·h_t/(λ·R))`; this module implements that
/// formula. The tolerance below allows ±5 m.)
#[test]
fn two_ray_first_null_low_grazing() {
    let lambda = SPEED_OF_LIGHT_M_PER_S / 3.0e9;
    let h_t_null = lambda * 80_000.0 / (2.0 * 20.0);
    let f = two_ray_propagation_factor_magnitude(3.0e9, h_t_null, 20.0, 80_000.0, 1.0);
    assert!(
        f < 1e-6,
        "expected null |F| ≈ 0 at h_t={h_t_null:.2} m; got {f}"
    );
    // Also confirm the analytic first-null altitude is within 5 m
    // of the 200 m spec value.
    assert!(
        (h_t_null - 200.0).abs() < 5.0,
        "analytic first-null altitude {h_t_null:.2} m off from spec 200 m"
    );
}

/// Test 6 — First two-ray peak altitude. Same geometry as Test 5,
/// peak at `h_t = λ·R / (4·h_r) ≈ 99.93 m`, with `|F| → 2`.
#[test]
fn two_ray_first_peak_low_grazing() {
    let lambda = SPEED_OF_LIGHT_M_PER_S / 3.0e9;
    let h_t_peak = lambda * 80_000.0 / (4.0 * 20.0);
    let f = two_ray_propagation_factor_magnitude(3.0e9, h_t_peak, 20.0, 80_000.0, 1.0);
    assert!(
        (f - 2.0).abs() < 1e-6,
        "expected peak |F| ≈ 2 at h_t={h_t_peak:.2} m; got {f}"
    );
    assert!(
        (h_t_peak - 100.0).abs() < 5.0,
        "analytic first-peak altitude {h_t_peak:.2} m off from spec 100 m"
    );
}

/// Test 7 — ITU-R P.676 anchor: X-band (10 GHz), 100 km, standard
/// reference atmosphere (288.15 K, 101.325 kPa, 7.5 g/m³ H₂O).
/// Expected ≈ 1.3 dB ± 0.5 dB.
#[test]
fn p676_x_band_reference_atmosphere() {
    let att = itu_r_p676_gas_attenuation_db(10.0, 100.0, 288.15, 101.325, 7.5);
    assert!(
        (att - 1.3).abs() < 0.5,
        "X-band 100 km gas attenuation {att:.3} dB outside 1.3 ± 0.5"
    );
}

/// Test 8 — Atmospheric gas attenuation is strictly increasing in
/// path length for any fixed atmosphere / frequency.
#[test]
fn p676_monotonic_in_range() {
    let mut prev = -1.0;
    for r_km in [1.0, 10.0, 50.0, 100.0, 250.0, 500.0] {
        let att = itu_r_p676_gas_attenuation_db(10.0, r_km, 288.15, 101.325, 7.5);
        assert!(
            att > prev,
            "gas attenuation not monotonic: prev={prev} att={att} at {r_km} km"
        );
        prev = att;
    }
}

/// Test 9 — ITU-R P.838: X-band, 10 mm/h rain (moderate),
/// horizontal polarization, 100 km. P.838-3 Table 1 gives
/// k_H=0.01217, α_H=1.2571 ⇒ γ ≈ 0.22 dB/km ⇒ 22 dB at 100 km,
/// well above the 0.5 dB floor required by the packet.
#[test]
fn p838_x_band_moderate_rain_horizontal() {
    let att = itu_r_p838_rain_attenuation_db(
        10.0,
        10.0,
        100.0,
        RainPolarization::Horizontal,
    );
    assert!(
        att > 0.5,
        "expected > 0.5 dB rain attenuation, got {att:.3} dB"
    );
    // Anchor the magnitude near the table prediction.
    let expected = 0.01217 * 10f64.powf(1.2571) * 100.0;
    assert!(
        (att - expected).abs() < 0.1,
        "rain attenuation {att:.3} dB diverges from table prediction {expected:.3} dB"
    );
}

/// Test 10 — Zero rainfall ⇒ zero rain attenuation, regardless of
/// frequency / polarization / path length.
#[test]
fn p838_zero_rain_zero_attenuation() {
    for pol in [
        RainPolarization::Horizontal,
        RainPolarization::Vertical,
        RainPolarization::Circular,
    ] {
        let att = itu_r_p838_rain_attenuation_db(10.0, 0.0, 100.0, pol);
        assert_eq!(att, 0.0, "expected zero attenuation for zero rain ({pol:?})");
    }
}

/// Test 11 — Sea-level refractivity matches the canonical
/// N_0 = 315 N-units from ITU-R P.453-14.
#[test]
fn p453_sea_level_refractivity() {
    let n = itu_r_p453_refractivity_n_units(0.0);
    assert!(
        (n - 315.0).abs() < 1e-9,
        "expected N(0) = 315 N-units, got {n}"
    );
}

/// Test 12 — At one scale height (7350 m), refractivity drops to
/// N_0 / e ≈ 115.88 N-units.
#[test]
fn p453_one_scale_height_refractivity() {
    let n = itu_r_p453_refractivity_n_units(7350.0);
    let expected = 315.0 / std::f64::consts::E;
    assert!(
        (n - expected).abs() < 0.5,
        "expected N(7350) ≈ {expected:.3}, got {n:.3}"
    );
    assert!((expected - 115.88).abs() < 0.05);
}

/// Test 13 (bonus) — Circular polarization rain attenuation lies
/// between the H- and V-polarization values for typical X-band rain,
/// confirming the P.530-17 polarization-mixing formula is correctly
/// applied.
#[test]
fn p838_circular_polarization_between_h_and_v() {
    let h = itu_r_p838_rain_attenuation_db(10.0, 25.0, 50.0, RainPolarization::Horizontal);
    let v = itu_r_p838_rain_attenuation_db(10.0, 25.0, 50.0, RainPolarization::Vertical);
    let c = itu_r_p838_rain_attenuation_db(10.0, 25.0, 50.0, RainPolarization::Circular);
    let lo = h.min(v);
    let hi = h.max(v);
    assert!(
        c >= lo - 1e-9 && c <= hi + 1e-9,
        "circular {c:.3} dB not bracketed by H {h:.3} / V {v:.3}"
    );
}
