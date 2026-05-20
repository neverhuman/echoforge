//! Passive RF / electronic-support sensor archetype.
//!
//! Models a passive receiver (or a network of receivers) that detects
//! and bears emitters such as drone control links, video downlinks,
//! telemetry, or GNSS-aided autopilots. The sensor never radiates.
//!
//! Three primitive sub-models are bundled:
//!
//! 1. **Single-site interferometric AoA** — a one- or two-baseline
//!    phase-interferometer measures the angle of arrival from the
//!    phase difference across the baseline. Bearing accuracy is fixed
//!    by the baseline-in-wavelengths and the SNR; we treat it as a
//!    declared parameter on the card.
//! 2. **TDoA network (3+ sites)** — multiple synchronised receivers
//!    cross-correlate the emitter waveform across pairs to estimate
//!    time-difference-of-arrival, then triangulate. Cross-range
//!    accuracy at slant-range `R` is approximately
//!    `Δx ≈ c · σ_t · R / baseline` (Stein 1981 §IV), where `σ_t` is
//!    the cross-correlation timing 1-σ.
//! 3. **Free-space link budget** — the receiver detects the emitter at
//!    the range where the received power equals the receiver MDS,
//!    `P_rx = EIRP - L_fs(R)` with `L_fs = 20·log10(4πR / λ)`.
//!
//! **Null-detection branch.** The single most important strict-open
//! property of this module is the explicit `detects_rf_silent_target`
//! helper: if the target is a pre-programmed RF-silent OWA drone
//! (Shahed-136 / Geran-2 class, GNSS+INS only, no telemetry uplink),
//! this passive-RF sensor cannot detect it at any range. The function
//! returns `None` to make that null answer first-class and prevent
//! callers from silently falling back to a fictitious range. The
//! limitation is documented in `tips/detectors/tip1.txt` §4.6 and in
//! the Robin Radar 2023 C-UAS sensor survey.
//!
//! # References
//!
//! - Stein S., *Algorithms for ambiguity function processing*, IEEE
//!   Trans. Acoust. Speech Signal Process. **ASSP-29**(3):588-599,
//!   1981 — TDoA / FDoA cross-correlation estimator and its accuracy
//!   bound `σ_t ≥ 1 / (B · √SNR)`.
//! - Robin Radar Systems, *Counter-UAS Sensor Architecture Survey*,
//!   2023 — passive-RF capability table, especially the
//!   "RF-silent OWA drone" gap.
//! - Adamy D. L., *EW 101: A First Course in Electronic Warfare*,
//!   Artech House 2001 — link-budget conventions used here.

/// Angle-of-arrival measurement method discriminator. Matches the
/// `aoa_method` enum (subset) of
/// `schemas/passive_rf_sensor_card.schema.json`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AoaMethod {
    /// Single-baseline phase interferometer.
    SingleBaselineInterferometry,
    /// TDoA network of three or more synchronised sites.
    TdoaNetworkThreeOrMore,
    /// Multi-baseline phase interferometry (ambiguity-resolution
    /// baselines).
    PhaseInterferometryMultiBaseline,
    /// Monopulse amplitude-comparison panels.
    MonopulseAmplitudeComparison,
    /// Monopulse phase-comparison panels.
    MonopulsePhaseComparison,
}

/// Passive-RF sensor parameters used by the link-budget / TDoA
/// helpers.
#[derive(Debug, Clone, Copy)]
pub struct PassiveRfSensorParams {
    /// Tunable RF range `(f_lo, f_hi)` (Hz).
    pub frequency_range_hz: (f64, f64),
    /// Instantaneous bandwidth at any one tune (Hz).
    pub instantaneous_bandwidth_hz: f64,
    /// Receiver minimum-detectable-signal at declared bandwidth (dBm).
    pub sensitivity_dbm: f64,
    /// AoA estimator method.
    pub aoa_method: AoaMethod,
    /// 1-σ bearing accuracy (deg).
    pub bearing_accuracy_deg: f64,
    /// Number of receiver sites in a deployed network (1 for
    /// single-site interferometry, ≥ 3 for TDoA 2D, ≥ 4 for TDoA 3D).
    pub network_count: u32,
    /// Inter-site baseline (km). Zero for single-site sensors.
    pub baseline_km: f64,
    /// Cross-correlation timing 1-σ (ns).
    pub timing_accuracy_ns: f64,
}

/// Cross-range location accuracy (m) for a TDoA network at slant-range
/// `range_m`. Stein 1981 §IV gives the canonical first-order
/// approximation
///
/// ```text
/// Δx ≈ c · σ_t · R / baseline
/// ```
///
/// (i.e. the cross-range error is the timing error multiplied by the
/// lever-arm ratio `R / baseline`). Returns 0 for non-TDoA sensors or
/// invalid geometry.
pub fn tdoa_cross_range_accuracy_m(
    params: &PassiveRfSensorParams,
    range_m: f64,
) -> f64 {
    if params.aoa_method != AoaMethod::TdoaNetworkThreeOrMore {
        return 0.0;
    }
    if params.baseline_km <= 0.0 || params.timing_accuracy_ns <= 0.0 || range_m <= 0.0 {
        return 0.0;
    }
    let c_m_per_s = 2.998_792_458e8_f64;
    let sigma_t_s = params.timing_accuracy_ns * 1e-9;
    let baseline_m = params.baseline_km * 1000.0;
    c_m_per_s * sigma_t_s * range_m / baseline_m
}

/// Free-space passive-RF detection range (m) for an emitter at EIRP
/// `emitter_eirp_dbm` (dBm) on frequency `frequency_ghz` (GHz). Solves
///
/// ```text
/// P_rx = EIRP - 20·log10(4π R / λ)  ≥  MDS
/// ```
///
/// for `R`. With wavelength `λ = c / f`, the budget at the threshold is
///
/// ```text
/// EIRP - MDS = 20·log10(4π R / λ)
///     ⇒  R = λ / (4π) · 10^((EIRP - MDS) / 20)
/// ```
///
/// Returns 0 if the receiver MDS already exceeds the emitter EIRP at
/// any positive range.
pub fn detection_range_m(
    params: &PassiveRfSensorParams,
    emitter_eirp_dbm: f64,
    frequency_ghz: f64,
) -> f64 {
    if frequency_ghz <= 0.0 {
        return 0.0;
    }
    let path_loss_budget_db = emitter_eirp_dbm - params.sensitivity_dbm;
    if path_loss_budget_db <= 0.0 {
        return 0.0;
    }
    let c_m_per_s = 2.998_792_458e8_f64;
    let lambda_m = c_m_per_s / (frequency_ghz * 1e9);
    let coeff = lambda_m / (4.0 * std::f64::consts::PI);
    coeff * 10.0_f64.powf(path_loss_budget_db / 20.0)
}

/// First-class null-detection helper. Returns `None` when the target
/// is declared RF-silent (Shahed-class OWA drones with GNSS+INS only
/// and no command/control link); otherwise the caller is responsible
/// for invoking [`detection_range_m`] with an appropriate emitter
/// EIRP. This explicit `Option<f64>` signature exists to prevent
/// downstream chains from silently substituting a fictional non-zero
/// range when the platform legitimately does not emit. The behaviour
/// is documented in `tips/detectors/tip1.txt` §4.6 and the Robin
/// Radar 2023 C-UAS sensor survey.
pub fn detects_rf_silent_target(silent_default: bool) -> Option<f64> {
    if silent_default {
        None
    } else {
        Some(0.0)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn typical_tdoa_network() -> PassiveRfSensorParams {
        PassiveRfSensorParams {
            frequency_range_hz: (70.0e6, 6.0e9),
            instantaneous_bandwidth_hz: 80.0e6,
            sensitivity_dbm: -110.0,
            aoa_method: AoaMethod::TdoaNetworkThreeOrMore,
            bearing_accuracy_deg: 1.5,
            network_count: 3,
            baseline_km: 20.0,
            timing_accuracy_ns: 30.0,
        }
    }

    /// Test 1 — TDoA acceptance gate (Stein 1981 §IV). At 10 km range
    /// with 30 ns timing accuracy and 20 km baseline,
    ///
    /// ```text
    /// Δx ≈ 3e8 · 30e-9 · 10_000 / 20_000 ≈ 4.5 m
    /// ```
    ///
    /// Acceptance window: 3-6 m cross-range accuracy.
    #[test]
    fn tdoa_cross_range_10km_30ns_20km_baseline() {
        let params = typical_tdoa_network();
        let dx = tdoa_cross_range_accuracy_m(&params, 10_000.0);
        assert!(
            (3.0..6.0).contains(&dx),
            "TDoA cross-range accuracy at 10 km = {dx:.3} m, expected 3-6 m"
        );
    }

    /// Test 2 — Free-space link-budget acceptance gate. A
    /// +20 dBm EIRP commercial control link at 5 GHz received by a
    /// -110 dBm passive RF receiver must give a detection range in
    /// the 10-25 km bracket:
    ///
    /// ```text
    /// λ        = 3e8 / 5e9 = 0.06 m
    /// budget   = 20 - (-110) = 130 dB
    /// R        = 0.06 / (4π) · 10^(130/20)
    ///         ≈ 4.77e-3 · 3.16e6 ≈ 15 km
    /// ```
    ///
    /// Acceptance window: 10-25 km.
    #[test]
    fn passive_rf_detection_5ghz_neg110_plus20_eirp() {
        let params = typical_tdoa_network();
        let r = detection_range_m(&params, 20.0, 5.0);
        assert!(
            (10_000.0..25_000.0).contains(&r),
            "passive-RF detection range was {r:.0} m, expected 10-25 km"
        );
    }

    /// Test 3 — RF-silent target null-detection guard. A
    /// pre-programmed OWA drone with `silent_default = true`
    /// must return `None`, never a non-zero detection range. This is
    /// the explicit Shahed-class limitation declared on every passive-
    /// RF sensor card per `tips/detectors/tip1.txt` §4.6.
    #[test]
    fn rf_silent_target_returns_null() {
        assert!(detects_rf_silent_target(true).is_none());
        assert!(detects_rf_silent_target(false).is_some());
    }

    /// Test 4 — Non-TDoA sensor returns 0 cross-range accuracy from
    /// the TDoA helper (the geometry simply does not apply). Guards
    /// against accidental use of TDoA accuracy formulae for
    /// interferometric single-site sensors.
    #[test]
    fn tdoa_helper_returns_zero_for_non_tdoa_sensor() {
        let mut params = typical_tdoa_network();
        params.aoa_method = AoaMethod::SingleBaselineInterferometry;
        params.baseline_km = 0.0;
        let dx = tdoa_cross_range_accuracy_m(&params, 10_000.0);
        assert_eq!(dx, 0.0);
    }

    /// Test 5 — Detection range scales as `10^(ΔBudget / 20)`: every
    /// extra 6 dB of link budget should approximately double the
    /// detection range. Comparing +20 dBm vs +26 dBm at 5 GHz must
    /// give a 1.95-2.05× ratio.
    #[test]
    fn detection_range_doubles_per_6db_budget_increase() {
        let params = typical_tdoa_network();
        let r_low = detection_range_m(&params, 20.0, 5.0);
        let r_high = detection_range_m(&params, 26.0, 5.0);
        let ratio = r_high / r_low;
        assert!(
            (1.95..2.05).contains(&ratio),
            "expected ~2× range per +6 dB, got ratio {ratio:.3}"
        );
    }
}
