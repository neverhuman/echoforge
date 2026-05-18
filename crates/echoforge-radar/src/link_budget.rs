//! Monostatic radar-equation link budget.
//!
//! This module computes received signal power, noise power, and the
//! resulting per-pulse signal-to-noise ratio from a transparent set of
//! sensor / propagation / target parameters. It replaces the
//! `target_snr_db` knob previously hardcoded in
//! `crates/echoforge-radar/src/sim.rs::synthesize_takeoff_episode` so
//! that SNR EMERGES from physics rather than being an input.
//!
//! # Radar equation
//!
//! For a monostatic pulse-doppler radar tracking a point target of
//! radar cross section `σ` at slant range `R`, the per-pulse received
//! power is (Skolnik, *Introduction to Radar Systems*, 3rd ed. (2001),
//! §2.5, eq. 2.6):
//!
//! ```text
//!         P_t · G_t · G_r · λ² · σ · |F|²
//! P_r  =  -----------------------------------
//!         (4π)³ · R⁴ · L_atmos · L_rain · L_sys
//! ```
//!
//! where
//! - `P_t` is the transmit power (W),
//! - `G_t`, `G_r` are the transmit / receive antenna power gains
//!   (linear, not dB),
//! - `λ = c / f_c` is the carrier wavelength (m),
//! - `σ` is the target radar cross section (m²),
//! - `|F|` is the magnitude of the propagation factor accounting for
//!   ground-reflected multipath (Balanis §4.7, supplied by
//!   `crate::propagation::two_ray_propagation_factor_magnitude`),
//! - `R` is the slant range to the target (m),
//! - `L_atmos`, `L_rain`, `L_sys` are linear (not dB) loss factors for
//!   the atmospheric / rain / system contributions, all >= 1.
//!
//! # Noise model
//!
//! The receiver thermal noise power is (Skolnik §2.6, eq. 2.10):
//!
//! ```text
//! P_n  =  k · T_sys · B · F_n
//! ```
//!
//! where `k` is the Boltzmann constant, `T_sys` is the system noise
//! temperature (K), `B` is the matched-filter noise bandwidth (Hz),
//! and `F_n` is the receiver noise figure (linear).
//!
//! # Coherent integration gain
//!
//! For `N` pulses processed coherently (slow-time FFT prior to
//! detection), the post-integration SNR rises by `10·log10(N)`
//! relative to the single-pulse SNR (Skolnik §2.8, eq. 2.46 in the
//! limit of perfect coherent integration). Non-coherent integration
//! losses (Marcum, Swerling) and processing loss (CFAR, straddle,
//! window) are folded into `processing_loss_db` instead of being
//! modeled separately here.
//!
//! # References
//!
//! - Skolnik, *Introduction to Radar Systems*, 3rd ed. (2001),
//!   chap. 2 — radar equation, noise model, integration gain.
//! - Balanis, *Antenna Theory: Analysis and Design*, 4th ed. (2016),
//!   §4.7 — two-ray ground reflection (consumed via
//!   `crate::propagation::two_ray_propagation_factor_magnitude`).
//! - ITU-R Recommendation P.676-13 (2022), Annex 2 — atmospheric gas
//!   attenuation (consumed via
//!   `crate::propagation::itu_r_p676_gas_attenuation_db`).
//! - ITU-R Recommendation P.838-3 (2005), Table 1 — rain attenuation
//!   (consumed via `crate::propagation::itu_r_p838_rain_attenuation_db`).
//! - CODATA 2018 — exact Boltzmann constant
//!   `k = 1.380649e-23 J/K` (SI redefinition).
//!
//! # Strict-open posture
//!
//! All parameters consumed by this module are explicit textbook /
//! ITU-R quantities. No measured platform parameters are embedded and
//! no operational radar tunings are baked in. Default sensor numbers
//! used by `RadarSimConfig` are sourced from the public-proxy
//! UAE-coastal scenario (`configs/scenarios/uae-coastal-surveillance-v1.json`)
//! which is itself documented as illustrative public-proxy data.

use crate::propagation::{
    itu_r_p838_rain_attenuation_db, min_target_altitude_for_los_m,
    two_ray_propagation_factor_magnitude, RainPolarization, SPEED_OF_LIGHT_M_PER_S,
    STANDARD_K_FACTOR,
};

/// Boltzmann constant `k` (J/K). CODATA 2018 / SI 2019 redefinition,
/// exact value 1.380649e-23 J/K.
pub const BOLTZMANN_J_PER_K: f64 = 1.380649e-23;

/// Reference noise temperature T₀ = 290 K (IEEE / Skolnik convention).
/// Used as the default for `LinkBudget::system_temperature_k` when the
/// caller does not override it.
pub const REFERENCE_NOISE_TEMPERATURE_K: f64 = 290.0;

/// Sensor parameters that feed the link budget. All fields are SI
/// engineering units; gains and losses are expressed as decibels so
/// callers can populate them directly from radar engineering specs
/// (sensor data sheets, scenario files).
///
/// Convert linear values to dB via `10·log10(linear)`; the module
/// handles the conversion internally.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LinkBudget {
    /// Transmit power at the antenna terminals (W).
    pub transmit_power_w: f64,
    /// Transmit antenna power gain (dBi).
    pub tx_gain_dbi: f64,
    /// Receive antenna power gain (dBi). For a monostatic radar this
    /// is typically equal to `tx_gain_dbi`.
    pub rx_gain_dbi: f64,
    /// Carrier frequency (Hz).
    pub carrier_hz: f64,
    /// Receiver noise figure (dB).
    pub noise_figure_db: f64,
    /// Matched-filter noise bandwidth (Hz). Typically the LFM chirp
    /// bandwidth for a pulse-compression radar.
    pub noise_bandwidth_hz: f64,
    /// System noise temperature (K). For a sky-pointing radar this
    /// should be `T0 + T_antenna + T_line`; for a typical surface
    /// surveillance radar the convention is to absorb the antenna /
    /// feedline contributions into `noise_figure_db` and leave this
    /// at `REFERENCE_NOISE_TEMPERATURE_K = 290 K`.
    pub system_temperature_k: f64,
    /// System / plumbing losses on transmit + receive paths (dB).
    pub system_loss_db: f64,
    /// Signal-processing losses (dB): CFAR loss, straddle loss,
    /// windowing loss, etc. Skolnik §2.8.
    pub processing_loss_db: f64,
    /// Number of pulses integrated coherently per CPI. Setting this
    /// to 1 disables the integration gain term.
    pub coherent_integration_pulses: u32,
}

impl Default for LinkBudget {
    /// Defaults sourced from the public-proxy S-band scenario
    /// `configs/scenarios/uae-coastal-surveillance-v1.json` (2.9 GHz,
    /// 1 MHz bandwidth, 1 MW transmit, 35 dBi antenna, 4 dB noise
    /// figure, 4 dB system loss, 100-pulse CPI).
    fn default() -> Self {
        Self {
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
}

/// Propagation geometry and atmosphere needed to compute the
/// loss / propagation-factor terms of the radar equation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PropagationContext {
    /// Slant range to the target (m).
    pub range_m: f64,
    /// Target altitude above ground level (m).
    pub target_altitude_agl_m: f64,
    /// Radar antenna height above local ground level (m).
    pub radar_altitude_agl_m: f64,
    /// One-way atmospheric specific attenuation at carrier
    /// (dB/km). Callers can pre-compute this from
    /// `crate::propagation::itu_r_p676_gas_attenuation_db` divided by
    /// range; supplied here as a context parameter so a single
    /// per-scenario value can be reused across many target ranges
    /// without recomputing the table interpolation.
    pub atmospheric_one_way_db_per_km: f64,
    /// Rain rate along the path (mm/h). Zero disables the rain term.
    pub rain_rate_mm_per_h: f64,
    /// Magnitude of the ground reflection coefficient `|Γ|`.
    /// Typically ~1 for low-grazing-angle geometries over a smooth
    /// reflector (sea, salt flat, runway tarmac). Setting to 0
    /// disables the two-ray multipath term entirely (no ground
    /// reflection assumed).
    pub ground_reflection_coefficient_magnitude: f64,
}

impl Default for PropagationContext {
    fn default() -> Self {
        Self {
            range_m: 50_000.0,
            target_altitude_agl_m: 100.0,
            radar_altitude_agl_m: 20.0,
            atmospheric_one_way_db_per_km: 0.0,
            rain_rate_mm_per_h: 0.0,
            ground_reflection_coefficient_magnitude: 0.0,
        }
    }
}

/// Result of a single-pulse + coherent-integration link-budget
/// evaluation.
///
/// All linear quantities are SI (W, dimensionless). All dB quantities
/// are referenced as marked in the field docstring (e.g. two-way for
/// atmospheric and rain losses, since the wave traverses the path
/// twice).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LinkBudgetResult {
    /// Per-pulse received signal power at the receiver input (W).
    pub received_power_w: f64,
    /// Thermal noise power referred to the same point (W).
    pub noise_power_w: f64,
    /// Post-integration SNR (linear).
    pub snr_linear: f64,
    /// Post-integration SNR (dB).
    pub snr_db: f64,
    /// Two-ray propagation factor in dB, `20·log10(|F|)`. Equal to 0
    /// when the ground reflection coefficient is zero (no multipath
    /// modeled).
    pub propagation_factor_db: f64,
    /// Atmospheric gas attenuation, TWO-WAY (dB). The radar wave
    /// traverses the path on transmit AND receive, so this is twice
    /// the one-way value from ITU-R P.676.
    pub atmospheric_loss_db: f64,
    /// Rain attenuation, TWO-WAY (dB). Same two-pass convention as
    /// the atmospheric term.
    pub rain_loss_db: f64,
    /// Free-space path loss `20·log10((4π·R)/λ)·2` (two-way, dB).
    /// Diagnostic only — the link-budget calculation works directly
    /// in linear power and does NOT pass through this dB term.
    pub free_space_path_loss_db: f64,
    /// Coherent-integration SNR gain (dB), `10·log10(N)` for `N >= 1`.
    pub coherent_integration_gain_db: f64,
    /// True if the target sits above the 4/3-Earth radar horizon
    /// from the radar antenna. False targets get zero return amplitude
    /// in the simulator chain.
    pub above_horizon: bool,
}

/// Evaluate the monostatic radar equation for the given sensor /
/// propagation / target geometry.
///
/// `target_rcs_m2` is the linear RCS in m² (NOT dBsm). Callers that
/// have a dBsm value should convert via `10f64.powf(dbsm / 10.0)`.
///
/// Returns a `LinkBudgetResult` whose `snr_db` field is the final
/// post-integration, post-processing-loss SNR.
pub fn evaluate_link_budget(
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
        // The propagation primitive currently only exposes a
        // horizontal/vertical/circular selector; pick circular as a
        // neutral default at the link-budget surface (callers can
        // override by pre-computing one-way attenuation themselves
        // and adding it to atmospheric_one_way_db_per_km).
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
    let numerator = budget.transmit_power_w
        * g_t
        * g_r
        * lambda_m * lambda_m
        * target_rcs_m2.max(0.0)
        * f_sq;
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

/// Convert a target SNR (dB) to a real-valued amplitude scalar such
/// that the resulting target sample power equals `snr_linear · σ²`
/// where `σ` is the noise standard deviation per receiver channel.
///
/// Specifically `target_amp = σ · sqrt(10^(snr_db/10))`, so the
/// per-sample power of the target return is `target_amp² = σ² ·
/// 10^(snr_db/10)`.
///
/// `noise_sigma` is the per-quadrature noise standard deviation in
/// the simulator's matched-filter input units (see
/// `NoiseProfile::awgn_sigma`). The function clamps the sigma to a
/// small positive floor so a vanishing noise budget does not blow
/// the amplitude up to infinity.
pub fn snr_to_target_amplitude(snr_db: f64, noise_sigma: f32) -> f32 {
    if !snr_db.is_finite() {
        return 0.0;
    }
    let sigma = (noise_sigma as f64).max(1e-9);
    let ratio = 10f64.powf(snr_db / 10.0).max(0.0);
    (sigma * ratio.sqrt()) as f32
}

#[inline]
fn db_to_linear(db: f64) -> f64 {
    10f64.powf(db / 10.0)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
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
            transmit_power_w: 1.0e6,
            tx_gain_dbi: 35.0,
            rx_gain_dbi: 35.0,
            carrier_hz: 3.0e9,
            noise_figure_db: 3.0,
            noise_bandwidth_hz: 1.0e6,
            system_temperature_k: REFERENCE_NOISE_TEMPERATURE_K,
            system_loss_db: 0.0,
            processing_loss_db: 0.0,
            coherent_integration_pulses: 1,
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
            transmit_power_w: 1.0e6,
            tx_gain_dbi: 35.0,
            rx_gain_dbi: 35.0,
            carrier_hz: 3.0e9,
            noise_figure_db: 3.0,
            noise_bandwidth_hz: 1.0e6,
            system_temperature_k: REFERENCE_NOISE_TEMPERATURE_K,
            system_loss_db: 0.0,
            processing_loss_db: 0.0,
            coherent_integration_pulses: 1,
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
        let mut budget = LinkBudget::default();
        budget.coherent_integration_pulses = 1;
        budget.processing_loss_db = 0.0;
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
}
