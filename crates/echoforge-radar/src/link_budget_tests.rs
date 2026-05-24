use super::*;
use crate::propagation::itu_r_p676_gas_attenuation_db;

/// Test 1 — Skolnik textbook reproduction.
///
/// Closed-form per Skolnik *Introduction to Radar Systems* 3rd ed.
/// §2.5, eq. (2.6), with `P_t=1 MW`, `G_t=G_r=35 dBi`,
/// `λ=c/3 GHz ≈ 0.0999 m`, `σ=1 m²`, `R=50 km`, `B=1 MHz`,
/// `F_n=3 dB`, `T_sys=290 K`, no atmospheric / rain / system /
/// processing losses, no two-ray multipath, single pulse.
///
/// Hand calculation:
///   numerator   = 1e6 · 10^3.5 · 10^3.5 · (0.0999)² · 1 · 1
///               = 1e6 · 3162.28 · 3162.28 · 9.987e-3
///               ≈ 9.988e10
///   denominator = (4π)³ · (50_000)⁴
///               = 1984.40 · 6.25e18
///               ≈ 1.240e22
///   P_r         = numerator / denominator ≈ 8.054e-12 W
///   P_n         = k · T_sys · B · F_n
///               = 1.381e-23 · 290 · 1e6 · 1.995
///               ≈ 7.99e-15 W
///   SNR         = 8.054e-12 / 7.99e-15 ≈ 1008
///   SNR_dB      = 10·log10(1008) ≈ 30.03 dB
///
/// Tolerance: ±0.5 dB.
#[test]
fn skolnik_textbook_reproduction() {
    let budget = LinkBudget {
        carrier_hz: 3.0e9,
        noise_figure_db: 3.0,
        system_loss_db: 0.0,
        processing_loss_db: 0.0,
        ..LinkBudget::default()
    };
    let prop = PropagationContext {
        range_m: 50_000.0,
        target_altitude_agl_m: 10_000.0,
        radar_altitude_agl_m: 100.0,
        atmospheric_one_way_db_per_km: 0.0,
        rain_rate_mm_per_h: 0.0,
        ground_reflection_coefficient_magnitude: 0.0,
    };
    let result = evaluate_link_budget(&budget, &prop, 1.0);
    let expected = 30.03_f64;
    assert!(
        result.above_horizon,
        "test geometry should be above the horizon"
    );
    assert!(
        (result.snr_db - expected).abs() < 0.5,
        "Skolnik §2.5 closed form expected {expected:.2} dB ± 0.5; got {:.3}",
        result.snr_db,
    );
}

/// Test 2 — R⁴ law: doubling range drops SNR by 12.04 dB.
///
/// `20·log10(2⁴) = 40·log10(2) ≈ 12.041 dB`. Tolerance 0.05 dB
/// to absorb numerical noise only.
#[test]
fn r_to_the_fourth_law() {
    let budget = LinkBudget {
        carrier_hz: 3.0e9,
        noise_figure_db: 3.0,
        system_loss_db: 0.0,
        processing_loss_db: 0.0,
        ..LinkBudget::default()
    };
    let prop_near = PropagationContext {
        range_m: 25_000.0,
        target_altitude_agl_m: 10_000.0,
        radar_altitude_agl_m: 100.0,
        atmospheric_one_way_db_per_km: 0.0,
        rain_rate_mm_per_h: 0.0,
        ground_reflection_coefficient_magnitude: 0.0,
    };
    let prop_far = PropagationContext {
        range_m: 50_000.0,
        ..prop_near
    };
    let snr_near = evaluate_link_budget(&budget, &prop_near, 1.0).snr_db;
    let snr_far = evaluate_link_budget(&budget, &prop_far, 1.0).snr_db;
    let drop = snr_near - snr_far;
    let expected = 40.0 * 2f64.log10();
    assert!(
        (drop - expected).abs() < 0.05,
        "R⁴ drop expected {expected:.4} dB; observed {drop:.4} dB",
    );
}

/// Test 3 — RCS proportional: 10× RCS lifts SNR by exactly 10 dB.
#[test]
fn rcs_proportional_to_snr() {
    let budget = LinkBudget::default();
    let prop = PropagationContext {
        range_m: 50_000.0,
        target_altitude_agl_m: 1000.0,
        radar_altitude_agl_m: 20.0,
        atmospheric_one_way_db_per_km: 0.0,
        rain_rate_mm_per_h: 0.0,
        ground_reflection_coefficient_magnitude: 0.0,
    };
    let snr_small = evaluate_link_budget(&budget, &prop, 1.0).snr_db;
    let snr_large = evaluate_link_budget(&budget, &prop, 10.0).snr_db;
    let delta = snr_large - snr_small;
    assert!(
        (delta - 10.0).abs() < 0.05,
        "expected +10 dB SNR for 10× RCS; observed {delta:.4} dB",
    );
}

/// Test 4 — ITU-R P.676 atmospheric loss at X-band.
///
/// 10 GHz, 100 km path, standard reference atmosphere
/// (T = 288.15 K, P = 101.325 kPa, 7.5 g/m³ H₂O). The propagation
/// primitive returns ≈ 1.3 dB one-way → ≈ 2.6 dB two-way. Skolnik
/// §2.10 and ITU-R P.676 Annex 2 figures both sit in the 2-5 dB
/// two-way band for this geometry. Tolerance: 2.0–5.0 dB two-way.
#[test]
fn p676_x_band_two_way_loss() {
    let one_way_db = itu_r_p676_gas_attenuation_db(10.0, 100.0, 288.15, 101.325, 7.5);
    let one_way_per_km = one_way_db / 100.0;
    let budget = LinkBudget {
        transmit_power_w: 1.0e6,
        tx_gain_dbi: 35.0,
        rx_gain_dbi: 35.0,
        carrier_hz: 10.0e9,
        noise_figure_db: 3.0,
        noise_bandwidth_hz: 1.0e6,
        system_temperature_k: REFERENCE_NOISE_TEMPERATURE_K,
        system_loss_db: 0.0,
        processing_loss_db: 0.0,
        coherent_integration_pulses: 1,
    };
    let prop = PropagationContext {
        range_m: 100_000.0,
        target_altitude_agl_m: 5_000.0,
        radar_altitude_agl_m: 20.0,
        atmospheric_one_way_db_per_km: one_way_per_km,
        rain_rate_mm_per_h: 0.0,
        ground_reflection_coefficient_magnitude: 0.0,
    };
    let result = evaluate_link_budget(&budget, &prop, 1.0);
    let two_way = result.atmospheric_loss_db;
    assert!(
        (2.0..=5.0).contains(&two_way),
        "X-band 100 km two-way atmospheric loss {two_way:.3} dB out of 2-5 dB band",
    );
}

/// Test 5 — Coherent integration gain: N=32 pulses → +15.05 dB.
///
/// `10·log10(32) ≈ 15.051 dB`. Tolerance 0.01 dB (analytic).
#[test]
fn coherent_integration_gain_matches_log10_n() {
    let mut budget = LinkBudget {
        coherent_integration_pulses: 1,
        processing_loss_db: 0.0,
        ..Default::default()
    };
    let prop = PropagationContext {
        range_m: 50_000.0,
        target_altitude_agl_m: 1000.0,
        radar_altitude_agl_m: 20.0,
        atmospheric_one_way_db_per_km: 0.0,
        rain_rate_mm_per_h: 0.0,
        ground_reflection_coefficient_magnitude: 0.0,
    };
    let snr_single = evaluate_link_budget(&budget, &prop, 1.0).snr_db;

    budget.coherent_integration_pulses = 32;
    let result_n = evaluate_link_budget(&budget, &prop, 1.0);
    let delta = result_n.snr_db - snr_single;
    let expected = 10.0 * 32f64.log10();
    assert!(
        (delta - expected).abs() < 0.01,
        "coherent integration gain expected {expected:.4} dB; got {delta:.4} dB",
    );
    assert!(
        (result_n.coherent_integration_gain_db - expected).abs() < 0.01,
        "coherent_integration_gain_db field {} differs from analytic {expected:.4}",
        result_n.coherent_integration_gain_db,
    );
}

/// Test 6 — Above-horizon check.
///
/// Geometry: radar antenna at 20 m AGL, target at R=100 km.
/// 4/3-Earth horizon requires h_t ≈ 391.6 m (see
/// `propagation::min_target_altitude_canonical_geometry` test).
/// A 10 m target is below the horizon; a 1000 m target is above.
#[test]
fn above_horizon_check_at_100km() {
    let budget = LinkBudget::default();
    let prop_low = PropagationContext {
        range_m: 100_000.0,
        target_altitude_agl_m: 10.0,
        radar_altitude_agl_m: 20.0,
        atmospheric_one_way_db_per_km: 0.0,
        rain_rate_mm_per_h: 0.0,
        ground_reflection_coefficient_magnitude: 0.0,
    };
    let prop_high = PropagationContext {
        target_altitude_agl_m: 1000.0,
        ..prop_low
    };
    let low = evaluate_link_budget(&budget, &prop_low, 1.0);
    let high = evaluate_link_budget(&budget, &prop_high, 1.0);
    assert!(
        !low.above_horizon,
        "10 m target at 100 km should be below horizon for 20 m antenna",
    );
    assert!(
        high.above_horizon,
        "1000 m target at 100 km should be above horizon for 20 m antenna",
    );
}

/// `snr_to_target_amplitude` round-trips: power(amp) / σ² = 10^(snr/10).
#[test]
fn snr_to_amplitude_round_trip() {
    let sigma = 0.05_f32;
    for snr_db in [-10.0_f64, 0.0, 10.0, 20.0, 30.0] {
        let amp = snr_to_target_amplitude(snr_db, sigma);
        let observed_snr_linear = (amp as f64 / sigma as f64).powi(2);
        let expected = 10f64.powf(snr_db / 10.0);
        let rel = (observed_snr_linear - expected).abs() / expected;
        assert!(
            rel < 1e-5,
            "snr_to_target_amplitude(snr={snr_db}, σ={sigma}) gave amp={amp} → \
             SNR_linear={observed_snr_linear:.6}; expected {expected:.6} (rel={rel})",
        );
    }
}

/// `snr_to_target_amplitude` returns 0 for non-finite SNR (e.g.
/// sub-horizon target with `snr_db == -inf`).
#[test]
fn snr_to_amplitude_handles_non_finite() {
    assert_eq!(snr_to_target_amplitude(f64::NEG_INFINITY, 0.05), 0.0);
    assert_eq!(snr_to_target_amplitude(f64::NAN, 0.05), 0.0);
}
