use serde::{Deserialize, Serialize};

use crate::cfar::CfarParams;
use crate::link_budget::{LinkBudget, PropagationContext, REFERENCE_NOISE_TEMPERATURE_K};
use crate::rcs::Polarization;
use crate::waveform::LfmChirp;
// NoiseProfile re-exported from config_noise submodule.

#[path = "config_noise.rs"]
mod config_noise;
pub use config_noise::NoiseProfile;

use super::episode::TargetState;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TakeoffProfile {
    pub initial_range_m: f64,
    pub runway_heading_deg: f64,
    pub ground_speed_mps: f64,
    pub acceleration_mps2: f64,
    pub climb_rate_mps: f64,
    pub max_altitude_m: f64,
    pub radial_velocity_bias_mps: f64,
    pub pitch_jitter_deg: f64,
    pub yaw_jitter_deg: f64,
    pub propulsor_hz: f64,
    pub micro_doppler_hz: f64,
    pub rcs_scalar: f64,
    /// Number of rotor/propeller blades. Default `None` preserves the
    /// prior single-sinusoid micro-Doppler used by existing reproduction
    /// fixtures. When `Some(n)` (with `blade_length_m` also `Some(_)`),
    /// the synthesis loop dispatches to
    /// [`crate::micro_doppler_gen::PropellerGenerator`], which models a
    /// multi-blade rotor with blade-flash convention. A `n = 2`
    /// configuration matches the Shahed-class public-proxy pusher prop.
    #[serde(default)]
    pub blade_count: Option<usize>,
    /// Blade length in metres (tip radius). Default `None` falls back
    /// to the prior single-sinusoid micro-Doppler. Shahed-class
    /// public-proxy proxy is roughly 0.6 m (see Wave-A
    /// `shahed-public-proxy-flight-envelope-v2` dossier).
    #[serde(default)]
    pub blade_length_m: Option<f64>,
}

impl Default for TakeoffProfile {
    fn default() -> Self {
        Self {
            initial_range_m: 1_450.0,
            runway_heading_deg: 18.0,
            ground_speed_mps: 31.0,
            acceleration_mps2: 1.2,
            climb_rate_mps: 4.5,
            max_altitude_m: 380.0,
            radial_velocity_bias_mps: -18.0,
            pitch_jitter_deg: 1.2,
            yaw_jitter_deg: 1.7,
            propulsor_hz: 95.0,
            micro_doppler_hz: 42.0,
            rcs_scalar: 1.0,
            blade_count: None,
            blade_length_m: None,
        }
    }
}

impl TakeoffProfile {
    pub fn state_at(&self, t_s: f64) -> TargetState {
        let speed = self.ground_speed_mps + self.acceleration_mps2 * t_s;
        let along_runway_m = self.ground_speed_mps * t_s + 0.5 * self.acceleration_mps2 * t_s * t_s;
        let heading_rad = self.runway_heading_deg.to_radians();
        let cross_range_m = along_runway_m * heading_rad.sin();
        let down_range_m = self.initial_range_m + along_runway_m * heading_rad.cos();
        let altitude_m = (self.climb_rate_mps * t_s).min(self.max_altitude_m);
        let slant_range_m =
            (down_range_m * down_range_m + cross_range_m * cross_range_m + altitude_m * altitude_m)
                .sqrt();
        let radial_velocity_mps = self.radial_velocity_bias_mps + speed * heading_rad.cos() * 0.35;
        let pitch_deg = (altitude_m / self.max_altitude_m.max(1.0) * 7.0)
            + self.pitch_jitter_deg * (2.0 * std::f64::consts::PI * 0.31 * t_s).sin();
        let yaw_deg = self.yaw_jitter_deg * (2.0 * std::f64::consts::PI * 0.17 * t_s + 0.4).sin();
        let propulsor_phase_rad = 2.0 * std::f64::consts::PI * self.propulsor_hz * t_s;

        TargetState {
            time_s: t_s,
            range_m: slant_range_m,
            altitude_m,
            radial_velocity_mps,
            pitch_deg,
            yaw_deg,
            propulsor_phase_rad,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RadarSimConfig {
    pub sample_rate_hz: f64,
    pub pulse_width_s: f64,
    pub bandwidth_hz: f64,
    pub carrier_hz: f64,
    pub pulse_count: usize,
    pub pri_s: f64,
    /// Diagnostic SNR knob. NO LONGER an input to the simulator chain
    /// — `synthesize_takeoff_episode` derives target amplitude from a
    /// full monostatic radar-equation link budget (see
    /// `crate::link_budget`). This field is retained so existing
    /// fixtures continue to deserialize, but its value is reserved at
    /// the simulator surface. Reproduction-byte fixtures may still
    /// populate it for backwards-compatibility, and downstream code
    /// can read the emergent SNR from
    /// `SyntheticEpisode::diagnostic_snr_db` instead.
    pub target_snr_db: f64,
    pub cfar_training_cells: usize,
    pub cfar_guard_cells: usize,
    pub cfar_pfa: f32,
    /// Transmit power at the antenna terminals (W). Default is sourced
    /// from the public-proxy UAE-coastal S-band scenario.
    pub transmit_power_w: f64,
    /// Transmit antenna power gain (dBi).
    pub tx_gain_dbi: f64,
    /// Receive antenna power gain (dBi). For a monostatic radar this
    /// is typically equal to `tx_gain_dbi`.
    pub rx_gain_dbi: f64,
    /// Receiver noise figure (dB).
    pub noise_figure_db: f64,
    /// System noise temperature (K). Defaults to the IEEE reference
    /// `T0 = 290 K` (see `link_budget::REFERENCE_NOISE_TEMPERATURE_K`).
    pub system_temperature_k: f64,
    /// System / plumbing losses (dB), one-way.
    pub system_loss_db: f64,
    /// Signal-processing losses (dB): CFAR, straddle, window.
    pub processing_loss_db: f64,
    /// Radar antenna height above ground level (m).
    pub radar_altitude_agl_m: f64,
    /// One-way atmospheric specific attenuation (dB/km) at carrier.
    /// Compute from `crate::propagation::itu_r_p676_gas_attenuation_db`
    /// divided by range; supplied as a config field so the table
    /// interpolation runs once per scenario.
    pub atmospheric_one_way_db_per_km: f64,
    /// Rain rate along the path (mm/h). Zero disables the rain term.
    pub rain_rate_mm_per_h: f64,
    /// Magnitude of the ground reflection coefficient `|Γ|` for the
    /// two-ray multipath term. Zero disables two-ray multipath.
    pub ground_reflection_coefficient_magnitude: f64,
    /// **Wave 4.5 Lane H2 — polarization-agility primitive.**
    /// Per-pulse transmit polarization sequence. When `None`, uses
    /// [`Polarization::Vv`] for every pulse (bridged). When
    /// `Some(vec)`, the sequence is indexed modulo `pulse_count`.
    /// Length 2 enables the classic alternating VV/HH agility per
    /// Skolnik *Introduction to Radar Systems* 3rd ed., §7.5.3 (clutter
    /// polarization diversity) and §11.6 (target discrimination via
    /// polarization).
    #[serde(default)]
    pub pol_tx_sequence: Option<Vec<Polarization>>,
    /// **Wave 4.5 Lane H2 — polarization-agility primitive.**
    /// Per-pulse receive polarization sequence. Same semantics as
    /// [`Self::pol_tx_sequence`]. Setting `tx` and `rx` to different
    /// sequences enables cross-polarization measurements (HV / VH) —
    /// the classic depolarization signature used to discriminate
    /// rough-surface clutter from smooth target returns (Skolnik
    /// §7.5.3; Ulaby & Long, *Microwave Radar and Radiometric Remote
    /// Sensing*, 2014, §10.2).
    #[serde(default)]
    pub pol_rx_sequence: Option<Vec<Polarization>>,
}

impl Default for RadarSimConfig {
    fn default() -> Self {
        Self {
            sample_rate_hz: 2_000_000.0,
            pulse_width_s: 128e-6,
            bandwidth_hz: 1_000_000.0,
            carrier_hz: 9_600_000_000.0,
            pulse_count: 32,
            pri_s: 900e-6,
            target_snr_db: 18.0,
            cfar_training_cells: 10,
            cfar_guard_cells: 3,
            cfar_pfa: 1e-3,
            // Public-proxy defaults sourced from
            // configs/scenarios/uae-coastal-surveillance-v1.json
            // (illustrative S-band surveillance radar).
            transmit_power_w: 1.0e6,
            tx_gain_dbi: 35.0,
            rx_gain_dbi: 35.0,
            noise_figure_db: 4.0,
            system_temperature_k: REFERENCE_NOISE_TEMPERATURE_K,
            system_loss_db: 4.0,
            processing_loss_db: 2.0,
            radar_altitude_agl_m: 20.0,
            atmospheric_one_way_db_per_km: 0.0,
            rain_rate_mm_per_h: 0.0,
            ground_reflection_coefficient_magnitude: 0.0,
            // Wave 4.5 H2 bridged: `None` keeps every pulse on
            // `Polarization::Vv`, byte-stable with pre-Lane-H2 fixtures.
            pol_tx_sequence: None,
            pol_rx_sequence: None,
        }
    }
}

impl RadarSimConfig {
    pub fn waveform(&self) -> LfmChirp {
        LfmChirp {
            sample_rate_hz: self.sample_rate_hz,
            pulse_width_s: self.pulse_width_s,
            bandwidth_hz: self.bandwidth_hz,
            carrier_hz: 0.0,
            initial_phase_rad: 0.0,
        }
    }

    pub fn cfar_params(&self) -> CfarParams {
        CfarParams::new(
            self.cfar_training_cells,
            self.cfar_guard_cells,
            self.cfar_pfa,
        )
    }

    /// Build a `LinkBudget` from this configuration. The noise
    /// bandwidth is taken from `self.bandwidth_hz` (matched-filter
    /// bandwidth of the LFM chirp), and the coherent integration
    /// length is taken from `self.pulse_count`.
    pub fn link_budget(&self) -> LinkBudget {
        LinkBudget {
            transmit_power_w: self.transmit_power_w,
            tx_gain_dbi: self.tx_gain_dbi,
            rx_gain_dbi: self.rx_gain_dbi,
            carrier_hz: self.carrier_hz,
            noise_figure_db: self.noise_figure_db,
            noise_bandwidth_hz: self.bandwidth_hz,
            system_temperature_k: self.system_temperature_k,
            system_loss_db: self.system_loss_db,
            processing_loss_db: self.processing_loss_db,
            coherent_integration_pulses: self.pulse_count.max(1) as u32,
        }
    }

    /// Build a `PropagationContext` from this configuration and the
    /// initial target state (`profile.state_at(0.0)`). The slant
    /// range, target altitude, and atmospheric/rain settings are
    /// pulled from the supplied profile + config; ground geometry
    /// (antenna height, reflection coefficient) comes from this
    /// config.
    pub fn propagation_context(&self, target_state: &TargetState) -> PropagationContext {
        PropagationContext {
            range_m: target_state.range_m,
            target_altitude_agl_m: target_state.altitude_m,
            radar_altitude_agl_m: self.radar_altitude_agl_m,
            atmospheric_one_way_db_per_km: self.atmospheric_one_way_db_per_km,
            rain_rate_mm_per_h: self.rain_rate_mm_per_h,
            ground_reflection_coefficient_magnitude: self.ground_reflection_coefficient_magnitude,
        }
    }

    /// **Wave 4.5 Lane H2 — polarization-agility primitive.**
    /// Resolve the (tx, rx) polarization pair for pulse index `pulse_idx`.
    ///
    /// Semantics:
    ///   - If [`Self::pol_tx_sequence`] is `None`, tx defaults to
    ///     [`Polarization::Vv`] (bridged).
    ///   - If [`Self::pol_tx_sequence`] is `Some(vec)`, the entry at
    ///     `pulse_idx % vec.len()` is selected (modulo cycling so the
    ///     sequence can be shorter than `pulse_count`).
    ///   - Receive polarization follows the same rule against
    ///     [`Self::pol_rx_sequence`]. If `pol_rx_sequence` is `None`,
    ///     rx mirrors tx (the matched/co-polar receive convention).
    ///
    /// Modern AESA radars switch polarization pulse-to-pulse for
    /// clutter diversity and target discrimination (Skolnik
    /// *Introduction to Radar Systems* 3rd ed., §7.5.3 — clutter
    /// polarization decorrelation; §11.6 — depolarization signatures
    /// for target classification). This helper is the deterministic
    /// hook into that sequencing: per-pulse RCS lookup keyed on
    /// (tx, rx) consumes the result.
    pub fn polarization_for_pulse(&self, pulse_idx: usize) -> (Polarization, Polarization) {
        let tx = self
            .pol_tx_sequence
            .as_ref()
            .and_then(|v| {
                if v.is_empty() {
                    None
                } else {
                    v.get(pulse_idx % v.len()).copied()
                }
            })
            .unwrap_or(Polarization::Vv);
        let rx = self
            .pol_rx_sequence
            .as_ref()
            .and_then(|v| {
                if v.is_empty() {
                    None
                } else {
                    v.get(pulse_idx % v.len()).copied()
                }
            })
            .unwrap_or(tx);
        (tx, rx)
    }
}

/// **Wave 4.5 Lane H2 — first-order polarization scaling.**
/// Returns the linear amplitude multiplier that scales the per-pulse
/// target return for a given `(tx, rx)` polarization pair.
///
/// This is a **first-order proxy** keyed on typical Shahed-class /
/// fixed-wing nose-on RCS observations from the open literature
/// (Skolnik 3rd ed. table 2.1; Knott, Shaeffer & Tuley, *Radar Cross
/// Section* 2nd ed., chap. 14 — small fixed-wing targets). It will be
/// superseded by full per-pulse `Rcs::evaluate(..., pol_tx, pol_rx, …)`
/// lookup when the rcs-aspect dispatch lands in Lane I follow-up;
/// until then, this scaling captures the qualitative polarization
/// signature so detector chains and downstream ML features see a
/// non-trivial polarization channel.
///
/// Reference values (one-way amplitude; dB → linear via 10^(dB/20)):
///   - VV → 1.0 (baseline, the seeded tabulated value).
///   - HH → +1 dB ≈ 1.122 (slightly higher on slender airframes
///     because horizontal polarization couples better with the
///     fuselage-side scatterers when the radar is at low elevation).
///   - HV / VH → -10 dB ≈ 0.316 (typical cross-pol depolarization
///     ratio for a smooth target; rough natural clutter depolarizes
///     less aggressively, which is the whole reason cross-pol is a
///     useful discriminator — Ulaby & Long, *Microwave Radar and
///     Radiometric Remote Sensing*, 2014, §10.2).
///   - Any other variant (Co, Cross, mixed circular L/R) falls back
///     to 1.0 so unseen variants are not silently zeroed.
pub(super) fn polarization_amplitude_scale(tx: Polarization, rx: Polarization) -> f32 {
    match (tx, rx) {
        (Polarization::Vv, Polarization::Vv) => 1.0,
        (Polarization::Hh, Polarization::Hh) => 1.122_018_5, // 10^(+1/20)
        (Polarization::Hv, _)
        | (Polarization::Vh, _)
        | (_, Polarization::Hv)
        | (_, Polarization::Vh) => 0.316_227_77, // 10^(-10/20)
        _ => 1.0,
    }
}

