//! Implementation helpers for `link_budget.rs` — extracted for LOC compliance.

use super::{LinkBudget, LinkBudgetResult, PropagationContext, BOLTZMANN_J_PER_K};
use crate::propagation::{
    itu_r_p838_rain_attenuation_db, min_target_altitude_for_los_m,
    two_ray_propagation_factor_magnitude, RainPolarization, SPEED_OF_LIGHT_M_PER_S,
    STANDARD_K_FACTOR,
};

#[inline]
pub(super) fn db_to_linear(db: f64) -> f64 {
    10f64.powf(db / 10.0)
}

/// Core monostatic radar equation evaluation. Called from
/// `evaluate_link_budget` in `link_budget.rs`.
pub(super) fn evaluate_link_budget_impl(
    budget: &LinkBudget,
    prop: &PropagationContext,
    target_rcs_m2: f64,
) -> LinkBudgetResult {
    // ------------------------------------------------------------------
    // Horizon check (4/3-Earth model).
    // ------------------------------------------------------------------
    let required_alt = min_target_altitude_for_los_m(
        prop.radar_altitude_agl_m.max(0.0),
        prop.range_m.max(0.0),
        STANDARD_K_FACTOR,
    );
    let above_horizon = prop.target_altitude_agl_m > required_alt;

    // ------------------------------------------------------------------
    // Wavelength.
    // ------------------------------------------------------------------
    let lambda_m = if budget.carrier_hz > 0.0 {
        SPEED_OF_LIGHT_M_PER_S / budget.carrier_hz
    } else {
        0.0
    };

    // ------------------------------------------------------------------
    // Antenna gains and system losses, dB -> linear.
    // ------------------------------------------------------------------
    let g_t = db_to_linear(budget.tx_gain_dbi);
    let g_r = db_to_linear(budget.rx_gain_dbi);
    let f_n = db_to_linear(budget.noise_figure_db);
    let l_sys = db_to_linear(budget.system_loss_db.max(0.0));
    let l_proc = db_to_linear(budget.processing_loss_db.max(0.0));

    // ------------------------------------------------------------------
    // Propagation factor |F|.
    // ------------------------------------------------------------------
    let f_mag = if prop.ground_reflection_coefficient_magnitude > 0.0 {
        two_ray_propagation_factor_magnitude(
            budget.carrier_hz,
            prop.target_altitude_agl_m.max(0.0),
            prop.radar_altitude_agl_m.max(0.0),
            prop.range_m.max(1e-3),
            prop.ground_reflection_coefficient_magnitude,
        )
    } else {
        1.0
    };
    let f_sq = f_mag * f_mag;
    let propagation_factor_db = if f_mag > 0.0 {
        20.0 * f_mag.log10()
    } else {
        f64::NEG_INFINITY
    };

    // ------------------------------------------------------------------
    // Atmospheric and rain attenuation (one-way -> two-way for the
    // round-trip radar geometry).
    // ------------------------------------------------------------------
    let range_km = (prop.range_m.max(0.0)) / 1000.0;
    let atm_one_way_db = prop.atmospheric_one_way_db_per_km.max(0.0) * range_km;
    let atmospheric_loss_db = 2.0 * atm_one_way_db;
    let l_atmos = db_to_linear(atmospheric_loss_db);

    let rain_one_way_db = if prop.rain_rate_mm_per_h > 0.0 && range_km > 0.0 {
        itu_r_p838_rain_attenuation_db(
            budget.carrier_hz / 1.0e9,
            prop.rain_rate_mm_per_h,
            range_km,
            RainPolarization::Circular,
        )
    } else {
        0.0
    };
    let rain_loss_db = 2.0 * rain_one_way_db;
    let l_rain = db_to_linear(rain_loss_db);

    // ------------------------------------------------------------------
    // Radar equation in linear power.
    //
    //   P_r = (P_t G_t G_r λ² σ |F|²)
    //          / ((4π)³ R⁴ L_atmos L_rain L_sys)
    // ------------------------------------------------------------------
    let four_pi_cubed = (4.0 * std::f64::consts::PI).powi(3);
    let r = prop.range_m.max(0.0);
    let r4 = r.powi(4);
    let numerator =
        budget.transmit_power_w * g_t * g_r * lambda_m * lambda_m * target_rcs_m2.max(0.0) * f_sq;
    let denominator = four_pi_cubed * r4 * l_atmos * l_rain * l_sys;

    let received_power_w = if denominator > 0.0 && r4 > 0.0 {
        numerator / denominator
    } else {
        0.0
    };

    // ------------------------------------------------------------------
    // Noise power and single-pulse SNR.
    // ------------------------------------------------------------------
    let noise_power_w = BOLTZMANN_J_PER_K
        * budget.system_temperature_k.max(0.0)
        * budget.noise_bandwidth_hz.max(0.0)
        * f_n;

    let single_pulse_snr_linear = if noise_power_w > 0.0 {
        received_power_w / (noise_power_w * l_proc.max(1e-30))
    } else {
        0.0
    };

    // ------------------------------------------------------------------
    // Coherent integration gain.
    // ------------------------------------------------------------------
    let n = budget.coherent_integration_pulses.max(1) as f64;
    let coherent_integration_gain_db = 10.0 * n.log10();
    let integration_gain_linear = n;

    let snr_linear = single_pulse_snr_linear * integration_gain_linear;
    let snr_db = if snr_linear > 0.0 {
        10.0 * snr_linear.log10()
    } else {
        f64::NEG_INFINITY
    };

    // ------------------------------------------------------------------
    // Free-space path-loss diagnostic (two-way).
    // ------------------------------------------------------------------
    let free_space_path_loss_db = if r > 0.0 && lambda_m > 0.0 {
        2.0 * 20.0 * ((4.0 * std::f64::consts::PI * r) / lambda_m).log10()
    } else {
        f64::INFINITY
    };

    LinkBudgetResult {
        received_power_w,
        noise_power_w,
        snr_linear,
        snr_db,
        propagation_factor_db,
        atmospheric_loss_db,
        rain_loss_db,
        free_space_path_loss_db,
        coherent_integration_gain_db,
        above_horizon,
    }
}
