//! Radar-expert credibility smoke checklist.
//!
//! Each test below corresponds to a "5-minute sanity check" a
//! professional counter-UAS / AESA radar engineer would run before
//! trusting the simulator's output. If any of these fail, the
//! simulator is misrepresenting basic physics.
//!
//! Strict-open posture: each assertion is anchored to a published
//! textbook expectation. No measured-truth claims. Sensor defaults
//! match the public-proxy UAE coastal scenario at
//! `configs/scenarios/uae-coastal-surveillance-v1.json` (2.9 GHz S-band,
//! 1 MW transmit, 35 dBi antennas, 1 MHz matched-filter bandwidth, 4 dB
//! noise figure, 4 dB system loss, 2 dB processing loss, single pulse).
//!
//! References:
//! - Skolnik, *Introduction to Radar Systems*, 3rd ed. (2001),
//!   chap. 2 — radar equation, noise model, 4th-root law, horizon
//!   geometry.
//! - ITU-R Recommendation P.838-3 (2005), Table 1 — rain attenuation
//!   coefficients `k`, `α` versus frequency and polarization.
//! - Chen, V. C., *The Micro-Doppler Effect in Radar* (Artech House,
//!   2011), chap. 5 — rotating-blade kinematics, tip-speed kinematic
//!   relation.

use echoforge_radar::{
    evaluate_link_budget, itu_r_p838_rain_attenuation_db, LinkBudget, Polarization,
    PropagationContext, PropellerGenerator, RainPolarization, Rcs, REFERENCE_NOISE_TEMPERATURE_K,
};

// ---------------------------------------------------------------------------
// Shared defaults — mirror the public-proxy S-band scenario so each smoke
// test starts from the same canonical baseline before the one parameter
// the test is exercising is varied.
// ---------------------------------------------------------------------------

fn canonical_s_band_budget() -> LinkBudget {
    LinkBudget {
        transmit_power_w: 1.0e6,
        tx_gain_dbi: 35.0,
        rx_gain_dbi: 35.0,
        carrier_hz: 2.9e9,
        noise_figure_db: 4.0,
        noise_bandwidth_hz: 1.0e6,
        system_temperature_k: REFERENCE_NOISE_TEMPERATURE_K,
        system_loss_db: 4.0,
        processing_loss_db: 2.0,
        coherent_integration_pulses: 1,
    }
}

fn canonical_clear_air_prop(range_m: f64, target_altitude_agl_m: f64) -> PropagationContext {
    PropagationContext {
        range_m,
        target_altitude_agl_m,
        radar_altitude_agl_m: 20.0,
        atmospheric_one_way_db_per_km: 0.0,
        rain_rate_mm_per_h: 0.0,
        ground_reflection_coefficient_magnitude: 0.0,
    }
}

// ===========================================================================
// Smoke 1 — 4th-root law (Skolnik §2.5).
// ===========================================================================

/// **Smoke 1: 4th-root law (Skolnik §2.5).** Doubling target range
/// drops single-pulse SNR by exactly `40·log10(2) ≈ 12.04 dB` with all
/// other parameters held constant.
///
/// A real radar engineer's first sanity check on any radar simulator is
/// whether the received signal falls as `1/R^4`. Anything else means
/// the equation is wired wrong (e.g. `1/R^2` would be a one-way
/// communications link, `1/R^3` would mean the bistatic / spherical
/// scatterer model leaked into the monostatic path).
#[test]
fn smoke_1_fourth_root_law() {
    let budget = canonical_s_band_budget();
    // Pick target altitude generously above the horizon so the
    // `above_horizon` check does not short-circuit either side of the
    // comparison.
    let prop_near = canonical_clear_air_prop(25_000.0, 10_000.0);
    let prop_far = canonical_clear_air_prop(50_000.0, 10_000.0);
    let r_near = evaluate_link_budget(&budget, &prop_near, 0.1);
    let r_far = evaluate_link_budget(&budget, &prop_far, 0.1);
    assert!(
        r_near.above_horizon && r_far.above_horizon,
        "test geometry should be above horizon at both ranges",
    );
    let delta_db = r_near.snr_db - r_far.snr_db;
    let expected = 40.0 * 2f64.log10();
    assert!(
        (delta_db - expected).abs() < 0.5,
        "4th-root law: 2x range should drop SNR by {expected:.2} dB \
         (Skolnik §2.5 eq. 2.6); got {delta_db:.2} dB",
    );
}

// ===========================================================================
// Smoke 2 — 10 dB / 1.778x RCS-range trade (Skolnik §2.5).
// ===========================================================================

/// **Smoke 2: RCS 10 dB swap (Skolnik §2.5).** A 10x larger linear RCS
/// must raise SNR by exactly 10 dB at fixed range, and equivalently
/// the detection range must extend by `10^(10/40) = 1.7783x` to recover
/// the SNR baseline of the smaller target. The two halves of this test
/// pin both sides of the `SNR ∝ σ · R^-4` proportionality.
#[test]
fn smoke_2_rcs_10db_swap_range_177x() {
    let budget = canonical_s_band_budget();
    let prop = canonical_clear_air_prop(50_000.0, 10_000.0);

    // Half (a) — SNR rises by exactly 10 dB for 10x larger RCS.
    let snr_small = evaluate_link_budget(&budget, &prop, 0.1).snr_db;
    let snr_big = evaluate_link_budget(&budget, &prop, 1.0).snr_db;
    let delta_db = snr_big - snr_small;
    assert!(
        (delta_db - 10.0).abs() < 0.5,
        "10x larger RCS should give +10 dB SNR (Skolnik §2.5); \
         observed {delta_db:.2} dB",
    );

    // Half (b) — range-scaling form. Find the range at which the
    // larger RCS reproduces the SNR of the smaller RCS at 50 km. By
    // R^-4 inversion, R_match = 50_000 · 10^(10/40) = 50_000 · 1.7783.
    let expected_match_range = 50_000.0 * 10f64.powf(10.0 / 40.0);
    let prop_extended = canonical_clear_air_prop(expected_match_range, 10_000.0);
    let snr_big_at_extended = evaluate_link_budget(&budget, &prop_extended, 1.0).snr_db;
    let residual_db = snr_big_at_extended - snr_small;
    assert!(
        residual_db.abs() < 0.05,
        "range scaling: stretching range by 1.7783x for a 10 dB stronger \
         target should recover the same SNR; residual {residual_db:.3} dB",
    );
}

// ===========================================================================
// Smoke 3 — sub-horizon target reports `above_horizon=false`.
// ===========================================================================

/// **Smoke 3: Sub-horizon target reports `above_horizon=false`
/// (Skolnik §2.10).** Geometry: 20 m radar antenna AGL, 50 km range to
/// a 50 m AGL target. Using the standard 4/3-Earth model
/// (`R_eff = (4/3)·R_e ≈ 8.495e6 m`), the antenna's own horizon is at
/// `sqrt(2·R_eff·20) ≈ 18.43 km`, so the target must be at least
/// `(50e3 - 18.43e3)² / (2·R_eff) ≈ 58.6 m` AGL to clear the radar
/// horizon. A 50 m target is therefore sub-horizon and the
/// `LinkBudgetResult::above_horizon` field must report `false`. A
/// silent zero-SNR miss would let the simulator pretend horizon-blocked
/// targets are simply non-detections, which is a classic silent-failure
/// mode.
#[test]
fn smoke_3_sub_horizon_target_reports_blocked() {
    let budget = canonical_s_band_budget();
    let prop = canonical_clear_air_prop(50_000.0, 50.0);
    let result = evaluate_link_budget(&budget, &prop, 0.1);
    assert!(
        !result.above_horizon,
        "target at 50 m AGL / radar at 20 m AGL / range 50 km is sub-horizon \
         under the 4/3-Earth model (min altitude ~58.6 m per Skolnik §2.10); \
         link budget must report above_horizon=false but reported true",
    );
}

// ===========================================================================
// Smoke 4 — RCS aspect dependence >= 10 dB.
// ===========================================================================

/// **Smoke 4: RCS aspect dependence >= 10 dB
/// (`Rcs::seeded_public_proxy_v1`).** Real targets are not isotropic —
/// nose-on vs broadside aspect must change RCS by at least 10 dB for at
/// least one seeded reference table. The strict-open seeded library
/// covers fixed-wing UAS, large birds, and quadrotors; the fixed-wing
/// planform alone should easily exceed 10 dB nose-vs-broadside (~15 dB
/// in the bundled table).
#[test]
fn smoke_4_rcs_aspect_dependence_ten_db() {
    let rcs = Rcs::seeded_public_proxy_v1();
    let candidates: [(&str, Polarization); 3] = [
        ("fixed-wing-uas-small", Polarization::Vv),
        ("bird-large-single", Polarization::Hh),
        ("quadrotor", Polarization::Vv),
    ];

    let mut max_delta_db = 0.0_f64;
    let mut max_delta_class: &str = "<none>";
    for (class, pol) in candidates {
        let v_nose = rcs.evaluate_static(class, 0.0, 0.0, 10.0, pol);
        let v_broadside = rcs.evaluate_static(class, 90.0, 0.0, 10.0, pol);
        assert!(
            v_nose.is_finite() && v_broadside.is_finite(),
            "{class} returned non-finite dBsm (nose={v_nose}, broadside={v_broadside})",
        );
        let delta = (v_nose - v_broadside).abs();
        if delta > max_delta_db {
            max_delta_db = delta;
            max_delta_class = class;
        }
    }
    assert!(
        max_delta_db >= 10.0,
        "expected at least one seeded RCS table to differ by >=10 dB \
         between 0deg and 90deg aspect; observed max delta = \
         {max_delta_db:.2} dB on class '{max_delta_class}'",
    );
}

// ===========================================================================
// Smoke 5 — ITU-R P.838 rain attenuation at X-band.
// ===========================================================================

/// **Smoke 5: ITU-R P.838 rain attenuation at X-band.** 50 mm/h
/// (heavy rain) at 10 GHz over a 50 km horizontal path must produce
/// at least 20 dB of TWO-WAY additional loss versus clear air. With
/// the vertical-polarization coefficients from P.838-3 Table 1
/// (`k_V = 0.01129`, `α_V = 1.2156`) the specific attenuation is
/// `γ_R = 0.01129 · 50^1.2156 ≈ 1.24 dB/km`, giving a one-way path
/// loss of ~62 dB and a two-way loss of ~124 dB — comfortably above
/// the 20 dB floor. The function returns total dB across the supplied
/// `range_km`, so a single call yields the one-way path loss and we
/// double it for the round-trip.
#[test]
fn smoke_5_heavy_rain_xband_loss_ge_20db() {
    let one_way_db = itu_r_p838_rain_attenuation_db(
        10.0, // GHz
        50.0, // mm/h (heavy rain)
        50.0, // km path
        RainPolarization::Vertical,
    );
    let total_two_way_db = 2.0 * one_way_db;
    assert!(
        total_two_way_db >= 20.0,
        "ITU-R P.838: 50 mm/h rain at 10 GHz over 50 km should give >=20 dB \
         two-way loss; got {total_two_way_db:.2} dB (one-way {one_way_db:.2} dB)",
    );
}

// ===========================================================================
// Smoke 6 — blade-tip kinematics for a Shahed-class proxy (Chen 2011).
// ===========================================================================

/// **Smoke 6: Blade-tip kinematics for a Shahed-class propeller proxy
/// (Chen 2011 §5).** A 2-blade × 95 Hz × 0.6 m `PropellerGenerator`
/// must have tip speed `v_tip = 2π · f · L = 2π · 95 · 0.6 ≈ 358.14
/// m/s`. At S-band 2.9 GHz this in turn produces a blade-tip Doppler
/// of `±2 · v_tip · f_c / c = ±2 · 358.14 · 2.9e9 / 3e8 ≈ ±6.92 kHz`,
/// and the dominant-blade AM line lands at `rotation_hz = 95 Hz` per
/// the N=2 physics convention documented in Lane D. This test pins the
/// kinematic side of the chain — if the analytic tip-speed identity
/// does not hold, every downstream micro-Doppler spectral assertion
/// is hollow.
#[test]
fn smoke_6_blade_tip_speed_analytical() {
    let prop = PropellerGenerator::new(2, 95.0, 0.6, 0.0);
    let v_tip = prop.tip_speed_mps();
    let expected = 2.0 * std::f64::consts::PI * 95.0 * 0.6;
    assert!(
        (v_tip - expected).abs() < 1e-3,
        "tip speed {v_tip} m/s should equal 2π·f·L = {expected} m/s \
         (Chen 2011 §5 rotating-blade kinematics)",
    );

    // Sanity-check the derived blade-tip Doppler at S-band 2.9 GHz.
    // The kinematic identity is `f_D = 2·v_tip·f_c / c`; we assert it
    // here to keep the analytical chain (mechanics → electromagnetics)
    // tied to a single sub-Hz tolerance.
    let c_m_per_s = 2.998e8_f64;
    let carrier_hz = 2.9e9_f64;
    let expected_blade_tip_doppler_hz = 2.0 * expected * carrier_hz / c_m_per_s;
    let observed_blade_tip_doppler_hz = 2.0 * v_tip * carrier_hz / c_m_per_s;
    assert!(
        (observed_blade_tip_doppler_hz - expected_blade_tip_doppler_hz).abs() < 1e-3,
        "blade-tip Doppler at S-band derived from observed v_tip ({observed_blade_tip_doppler_hz:.4} Hz) \
         should match analytic ({expected_blade_tip_doppler_hz:.4} Hz)",
    );
    // The analytical value should land near the documented ±6.92 kHz.
    assert!(
        (expected_blade_tip_doppler_hz - 6920.0).abs() < 100.0,
        "analytic blade-tip Doppler {expected_blade_tip_doppler_hz:.1} Hz \
         should be near 6.92 kHz at S-band 2.9 GHz",
    );
}
