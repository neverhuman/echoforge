//! Self-contained monostatic radar-equation link budget used by Wave 9.
//!
//! Implements the textbook Skolnik §2.5 form:
//!
//!   P_r = (P_t · G_t · G_r · λ² · σ) / ((4π)³ · R⁴ · L)
//!
//! plus thermal noise via Skolnik §2.6 eq 2.10:
//!
//!   P_n = k · T_sys · B · F
//!
//! and coherent integration gain `10·log10(N)` for N pulses in a CPI.
//!
//! Detection threshold uses the Albersheim approximation (Albersheim 1981)
//! for required single-look SNR at declared (Pd, Pfa); the simulator's
//! per-card declared envelope picks the (Pd, Pfa) operating point.
//!
//! Inputs are deliberately scalar (one frequency, one polarization, one
//! aspect) — this is the *Wave 9 case study* reproduction surface, not a
//! replacement for the existing `echoforge-radar::link_budget` chain
//! that the full simulator uses.

/// CODATA 2018 Boltzmann constant (exact).
pub const K_BOLTZMANN_J_PER_K: f64 = 1.380_649e-23;
/// Speed of light in vacuum (m/s).
pub const C_M_PER_S: f64 = 2.997_924_58e8;

/// Inputs needed to evaluate the link budget at one range / one RCS.
#[derive(Debug, Clone, Copy)]
pub struct LinkBudgetInputs {
    /// Peak transmit power (W).
    pub peak_power_w: f64,
    /// Transmit antenna gain (linear, not dB).
    pub g_t_linear: f64,
    /// Receive antenna gain (linear). Monostatic: usually g_t_linear.
    pub g_r_linear: f64,
    /// Wavelength (m). Defaults via c / freq.
    pub wavelength_m: f64,
    /// System noise temperature (K).
    pub system_temperature_k: f64,
    /// Noise figure (linear, not dB). Default: 10^(NF_dB/10).
    pub noise_figure_linear: f64,
    /// Matched-filter effective bandwidth (Hz). For LFM pulse compression
    /// this is 1 / (compressed pulse width) ≈ 1 / (pulse_width_s /
    /// compression_ratio).
    pub effective_bandwidth_hz: f64,
    /// Cumulative system losses (linear). Default: 10^(L_dB/10).
    pub system_losses_linear: f64,
    /// Number of pulses coherently integrated in one CPI.
    pub coherent_integration_pulses: usize,
}

impl LinkBudgetInputs {
    /// Build from a radar_platform_card-style parameter set.
    #[allow(clippy::too_many_arguments)]
    pub fn from_card_fields(
        peak_power_w: f64,
        gain_dbi: f64,
        center_frequency_hz: f64,
        system_temperature_k: f64,
        noise_figure_db: f64,
        pulse_width_s: f64,
        pulse_compression_ratio: f64,
        system_losses_db: f64,
        prf_hz: f64,
        dwell_time_s: f64,
    ) -> Self {
        let g_linear = 10f64.powf(gain_dbi / 10.0);
        let wavelength_m = C_M_PER_S / center_frequency_hz;
        let nf_linear = 10f64.powf(noise_figure_db / 10.0);
        let loss_linear = 10f64.powf(system_losses_db / 10.0);
        // Matched-filter equivalent noise bandwidth = 1 / τ_TX (transmit
        // pulse width), per Skolnik §6.2 ("the matched filter output
        // noise bandwidth equals the reciprocal of the transmit pulse
        // duration"). For LFM pulse compression, the SNR improvement
        // factor over an un-compressed pulse of the same transmit
        // duration is the compression ratio; that improvement is
        // already captured by using τ_TX (not τ_compressed) here.
        let _ = pulse_compression_ratio; // declared in the card for display; not needed here.
        let eff_bw_hz = 1.0 / pulse_width_s;
        // Coherent integration count = PRF · dwell.
        let n_pulses = (prf_hz * dwell_time_s).floor().max(1.0) as usize;
        Self {
            peak_power_w,
            g_t_linear: g_linear,
            g_r_linear: g_linear,
            wavelength_m,
            system_temperature_k,
            noise_figure_linear: nf_linear,
            effective_bandwidth_hz: eff_bw_hz,
            system_losses_linear: loss_linear,
            coherent_integration_pulses: n_pulses,
        }
    }

    /// Received target power at the radar receiver (W).
    pub fn received_power_w(&self, rcs_m2: f64, range_m: f64) -> f64 {
        let num = self.peak_power_w
            * self.g_t_linear
            * self.g_r_linear
            * self.wavelength_m.powi(2)
            * rcs_m2;
        let four_pi = 4.0 * std::f64::consts::PI;
        let den = four_pi.powi(3) * range_m.powi(4) * self.system_losses_linear;
        num / den
    }

    /// Receiver thermal noise power (W).
    pub fn noise_power_w(&self) -> f64 {
        K_BOLTZMANN_J_PER_K
            * self.system_temperature_k
            * self.effective_bandwidth_hz
            * self.noise_figure_linear
    }

    /// Single-pulse SNR (linear).
    pub fn single_pulse_snr_linear(&self, rcs_m2: f64, range_m: f64) -> f64 {
        self.received_power_w(rcs_m2, range_m) / self.noise_power_w()
    }

    /// Coherently-integrated SNR (linear).
    pub fn integrated_snr_linear(&self, rcs_m2: f64, range_m: f64) -> f64 {
        self.single_pulse_snr_linear(rcs_m2, range_m)
            * self.coherent_integration_pulses as f64
    }

    /// Integrated SNR in dB.
    pub fn integrated_snr_db(&self, rcs_m2: f64, range_m: f64) -> f64 {
        10.0 * self.integrated_snr_linear(rcs_m2, range_m).log10()
    }

    /// Solve for the range at which the integrated SNR equals
    /// `required_snr_db`. Closed-form because the radar equation is R^4.
    pub fn predicted_detection_range_m(
        &self,
        rcs_m2: f64,
        required_snr_db: f64,
    ) -> f64 {
        let snr_req_linear = 10f64.powf(required_snr_db / 10.0);
        // From: SNR_int = (Pt · Gt · Gr · λ² · σ · N) / ((4π)³ · R⁴ · L · N0)
        // R^4 = (Pt · Gt · Gr · λ² · σ · N) / ((4π)³ · L · N0 · SNR_req)
        let four_pi = 4.0 * std::f64::consts::PI;
        let num = self.peak_power_w
            * self.g_t_linear
            * self.g_r_linear
            * self.wavelength_m.powi(2)
            * rcs_m2
            * self.coherent_integration_pulses as f64;
        let den = four_pi.powi(3)
            * self.system_losses_linear
            * self.noise_power_w()
            * snr_req_linear;
        (num / den).powf(0.25)
    }
}

/// Albersheim 1981 approximation: required single-look SNR (dB) for a
/// non-fluctuating target at given Pd / Pfa. Coherent integration is
/// folded in by the caller via `integrated_snr_db`.
///
/// Reference: Albersheim, "Closed-Form Approximation to Robertson's
/// Detection Characteristics," Proc. IEEE, 1981.
pub fn albersheim_required_snr_db(pd: f64, pfa: f64) -> f64 {
    assert!((0.0..1.0).contains(&pd), "pd must be in [0, 1)");
    assert!((0.0..1.0).contains(&pfa), "pfa must be in [0, 1)");
    let a = (0.62 / pfa).ln();
    let b = (pd / (1.0 - pd)).ln();
    let snr_linear = a + 0.12 * a * b + 1.7 * b;
    10.0 * snr_linear.log10()
}

/// Swerling-target fluctuation penalty (dB) added on top of the
/// non-fluctuating Albersheim SNR floor at high Pd (≥ 0.8).
///
/// Values from Skolnik 3rd ed. Table 2.3 + Figure 2.8 evaluated at
/// Pd = 0.85, Pfa = 1e-4 (approximate; published tables span 0.5 dB
/// ranges depending on the specific approximation):
///
/// - Swerling 0 (non-fluctuating, single scatterer): 0 dB
/// - Swerling 1 (scan-to-scan, many equal scatterers): +5.7 dB
/// - Swerling 2 (pulse-to-pulse, many equal scatterers): +1.5 dB
/// - Swerling 3 (scan-to-scan, one dominant + many small): +2.0 dB
/// - Swerling 4 (pulse-to-pulse, one dominant + many small): +0.5 dB
///
/// The `fluctuation_model` field of `source_platform_card.rcs_signature`
/// selects the penalty.
pub fn swerling_fluctuation_penalty_db(fluctuation_model: &str) -> f64 {
    match fluctuation_model {
        "swerling_0" | "non_fluctuating" => 0.0,
        "swerling_1" => 5.7,
        "swerling_2" => 1.5,
        "swerling_3" => 2.0,
        "swerling_4" => 0.5,
        // Conservative default for unknown / mixed models.
        _ => 3.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Sanity: doubling range drops single-pulse received power by 12 dB
    /// (R^-4 → 4 doublings of inverse-power = -12.041 dB).
    #[test]
    fn r_fourth_power_law() {
        let inp = LinkBudgetInputs::from_card_fields(
            1500.0, 31.0, 3.0e9, 290.0, 3.0, 50.0e-6, 100.0, 4.0, 2000.0, 0.020,
        );
        let p_10km = inp.received_power_w(1.0, 10_000.0);
        let p_20km = inp.received_power_w(1.0, 20_000.0);
        let ratio_db = 10.0 * (p_10km / p_20km).log10();
        assert!(
            (ratio_db - 12.041).abs() < 0.01,
            "expected 12.041 dB drop for 2x range, got {ratio_db}"
        );
    }

    /// Sanity: doubling RCS gives 3 dB more SNR.
    #[test]
    fn rcs_linearity() {
        let inp = LinkBudgetInputs::from_card_fields(
            1500.0, 31.0, 3.0e9, 290.0, 3.0, 50.0e-6, 100.0, 4.0, 2000.0, 0.020,
        );
        let snr1 = inp.integrated_snr_db(1.0, 50_000.0);
        let snr2 = inp.integrated_snr_db(2.0, 50_000.0);
        assert!(
            (snr2 - snr1 - 3.010).abs() < 0.01,
            "expected 3.010 dB increase, got {}",
            snr2 - snr1
        );
    }

    /// Albersheim: published table example: Pd=0.9, Pfa=1e-6 → SNR ≈ 13.1 dB.
    #[test]
    fn albersheim_standard_point() {
        let snr = albersheim_required_snr_db(0.9, 1.0e-6);
        assert!(
            (snr - 13.1).abs() < 0.5,
            "Albersheim Pd=0.9 Pfa=1e-6 expected ~13.1 dB, got {snr}"
        );
    }
}
