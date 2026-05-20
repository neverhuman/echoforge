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
//! where each symbol maps to an SI field in [`LinkBudget`] and [`PropagationContext`]:
//! `P_t`=`transmit_power_w`, `G_t`/`G_r`=`tx/rx_gain_dbi` (converted to linear),
//! `λ`=`c/carrier_hz`, `σ`=target RCS (caller-supplied), `|F|`=two-ray factor
//! (from `crate::propagation`), `R`=`range_m`, `L_*`=system/atmospheric loss factors.
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

#[path = "link_budget_impl.rs"]
mod link_budget_impl;

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

impl LinkBudget {
    /// Public-proxy S-band defaults sourced from the UAE coastal surveillance scenario
    /// `configs/scenarios/uae-coastal-surveillance-v1.json` (2.9 GHz, 1 MHz bandwidth,
    /// 1 MW transmit, 35 dBi antenna, 4 dB noise figure, 4 dB system loss, 100-pulse CPI).
    pub fn uae_coastal_proxy() -> Self {
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

impl Default for LinkBudget {
    fn default() -> Self {
        Self::uae_coastal_proxy()
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

impl PropagationContext {
    /// Default propagation context for medium-range C-UAS scenarios:
    /// 50 km range, target at 100 m AGL, radar at 20 m AGL, clear-air with no rain.
    pub fn medium_range_clear_air() -> Self {
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

impl Default for PropagationContext {
    fn default() -> Self {
        Self::medium_range_clear_air()
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
    link_budget_impl::evaluate_link_budget_impl(budget, prop, target_rcs_m2)
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
/// small positive floor so zero-noise fixture paths stay numerically
/// visible without blowing the amplitude up to infinity.
pub fn snr_to_target_amplitude(snr_db: f64, noise_sigma: f32) -> f32 {
    if !snr_db.is_finite() {
        return 0.0;
    }
    let sigma = (noise_sigma as f64).max(1e-3);
    let ratio = 10f64.powf(snr_db / 10.0).max(0.0);
    (sigma * ratio.sqrt()) as f32
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
#[path = "link_budget_tests.rs"]
mod tests;
