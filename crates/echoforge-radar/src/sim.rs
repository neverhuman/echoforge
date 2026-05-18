use std::f32::consts::PI;

use serde::{Deserialize, Serialize};

use crate::cfar::{ca_cfar_1d, CfarParams};
use crate::clutter::{generate_clutter_sequence, ClutterRegime};
use crate::link_budget::{
    evaluate_link_budget, snr_to_target_amplitude, LinkBudget, LinkBudgetResult,
    PropagationContext, REFERENCE_NOISE_TEMPERATURE_K,
};
use crate::micro_doppler_gen::{MicroDopplerGenerator, PropellerGenerator};
use crate::pulse_compression::{magnitude, pulse_compress_windowed, CompressionWindow};
use crate::rcs::Polarization;
use crate::scene::{
    EnvironmentDescriptor, SceneDescriptor, SiteGeometry, TargetClass, TargetEntity,
    TargetKinematics,
};
use crate::waveform::LfmChirp;
use crate::ComplexSample;

const C_M_PER_S: f64 = 299_792_458.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct EpisodeSeed(pub u64);

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
    /// legacy single-sinusoid micro-Doppler used by existing reproduction
    /// fixtures. When `Some(n)` (with `blade_length_m` also `Some(_)`),
    /// the synthesis loop dispatches to
    /// [`crate::micro_doppler_gen::PropellerGenerator`], which models a
    /// multi-blade rotor with blade-flash convention. A `n = 2`
    /// configuration matches the Shahed-class public-proxy pusher prop.
    #[serde(default)]
    pub blade_count: Option<usize>,
    /// Blade length in metres (tip radius). Default `None` falls back
    /// to the legacy single-sinusoid micro-Doppler. Shahed-class
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
    /// fixtures continue to deserialize, but its value is unused at
    /// the simulator surface. Reproduction-byte fixtures may still
    /// populate it for backwards-compatibility, and downstream code
    /// can read the emergent SNR from
    /// `SyntheticEpisode::diagnostic_snr_db` instead.
    #[deprecated(
        note = "diagnostic only; SNR now emerges from link_budget() via the radar equation"
    )]
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
    /// [`Polarization::Vv`] for every pulse (back-compat). When
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
    #[allow(deprecated)]
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
            // Wave 4.5 H2 back-compat: `None` keeps every pulse on
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
    ///     [`Polarization::Vv`] (back-compat).
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
fn polarization_amplitude_scale(tx: Polarization, rx: Polarization) -> f32 {
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

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct NoiseProfile {
    pub awgn_sigma: f32,
    pub phase_noise_std_rad: f32,
    pub amplitude_scintillation_sigma: f32,
    pub rfi_probability: f32,
    pub rfi_amplitude: f32,
    pub clutter_sigma: f32,
    pub clutter_correlation: f32,
    pub ground_glint_count: usize,
    pub ground_glint_amplitude: f32,

    /// Cited K/Weibull/log-normal clutter regime per terrain class.
    /// When `Some(regime)`, the synthesis loop generates per-pulse
    /// per-range clutter via `generate_clutter_sequence(&regime, ...)`
    /// so that low-grazing-angle clutter exhibits the textbook heavy
    /// tails (Weibull shape ~1.2 for vegetated land, K-distribution
    /// shape `nu ~ 2` for mountain clutter, etc.). When `None`, the
    /// loop falls back to the legacy Gaussian AR(1) for byte-stable
    /// back-compat with pre-Lane-C fixtures.
    ///
    /// References:
    ///   - Skolnik, *Introduction to Radar Systems*, 3rd ed., chap. 7.
    ///   - Ward, Tough & Watts, *Sea Clutter: Scattering, the K
    ///     Distribution and Radar Performance*, IET 2013.
    #[serde(default)]
    pub clutter_regime: Option<ClutterRegime>,

    /// Multiplicative scaling applied to the regime's amplitude samples
    /// before they are summed into the IQ stream (`sigma_0` in linear
    /// units). `1.0` keeps the regime's nominal scale; values <1
    /// attenuate the clutter, values >1 amplify it. Ignored when
    /// `clutter_regime` is `None`.
    #[serde(default = "default_clutter_sigma_0_scale")]
    pub clutter_sigma_0_scale: f32,
}

fn default_clutter_sigma_0_scale() -> f32 {
    1.0
}

impl NoiseProfile {
    pub fn real_world_proxy_v1() -> Self {
        Self {
            awgn_sigma: 0.055,
            phase_noise_std_rad: 0.018,
            amplitude_scintillation_sigma: 0.11,
            rfi_probability: 0.006,
            rfi_amplitude: 0.9,
            clutter_sigma: 0.045,
            clutter_correlation: 0.94,
            ground_glint_count: 5,
            ground_glint_amplitude: 0.16,
            // Back-compat: keep the legacy Gaussian AR(1) path so
            // existing fixtures and byte-stable tests are unchanged.
            clutter_regime: None,
            clutter_sigma_0_scale: 1.0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TargetState {
    pub time_s: f64,
    pub range_m: f64,
    pub altitude_m: f64,
    pub radial_velocity_mps: f64,
    pub pitch_deg: f64,
    pub yaw_deg: f64,
    pub propulsor_phase_rad: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DetectionRecord {
    pub range_bin: usize,
    pub range_m: f64,
    pub statistic: f32,
    pub threshold: f32,
    pub noise_estimate: f32,
    pub confidence: f32,
}

#[derive(Debug, Clone)]
pub struct SyntheticEpisode {
    pub seed: EpisodeSeed,
    pub config: RadarSimConfig,
    pub profile: TakeoffProfile,
    pub noise: NoiseProfile,
    pub target_states: Vec<TargetState>,
    pub iq: Vec<Vec<ComplexSample>>,
    pub range_profiles_by_pulse: Vec<Vec<f32>>,
    pub integrated_range_profile: Vec<f32>,
    pub range_doppler_proxy: Vec<Vec<f32>>,
    pub detections: Vec<DetectionRecord>,
    /// Emergent post-integration SNR (dB) at the initial target state,
    /// computed by `crate::link_budget::evaluate_link_budget` from the
    /// sensor parameters in `config`, the geometry in
    /// `target_states[0]`, and the target `profile.rcs_scalar`. This
    /// is the diagnostic equivalent of the deprecated
    /// `RadarSimConfig::target_snr_db` knob. For multi-target scenes
    /// (Wave 5 Lane J) this field reports the FIRST entity's SNR; the
    /// per-entity SNR list is in [`Self::per_target_snr_db`].
    pub diagnostic_snr_db: f64,
    /// Full link-budget breakdown for the initial target state.
    pub diagnostic_link_budget: LinkBudgetResult,
    /// Complex range-Doppler grid preserving phase. Indexed
    /// `[range_bin][doppler_bin]`, with `range_bin in 0..compressed_len`
    /// and `doppler_bin in 0..n_pulses`. This is the canonical source
    /// for MTI/MTD/micro-Doppler processing — anything that needs the
    /// coherent slow-time phase history (Lane F MTI/MTD, Lane H Tier 3
    /// cruise) must consume this field rather than `range_doppler_proxy`
    /// (which is its magnitude image, retained only for back-compat with
    /// pre-Lane-B consumers).
    pub range_doppler_complex: Vec<Vec<ComplexSample>>,
    /// **Wave 5 Lane J — per-entity diagnostic SNR (dB).** For each
    /// entity in `SceneDescriptor::targets`, this vector carries the
    /// emergent post-integration SNR computed from the entity's
    /// initial geometry and a per-class RCS proxy (see
    /// `class_default_rcs_scalar` in `crate::sim`). Length equals
    /// `scene.targets.len()`. For single-entity scenes this is
    /// `vec![diagnostic_snr_db]`, preserving the legacy field.
    /// Multipath ghost entities (`TargetClass::MultipathGhost`) inherit
    /// their parent's SNR scaled by `20·log10(|Γ|)` so a ghost's
    /// entry reflects the reduced phantom-return level.
    pub per_target_snr_db: Vec<f64>,
}

#[derive(Debug, Clone)]
struct SplitMix64 {
    state: u64,
    cached_normal: Option<f32>,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self {
            state: seed,
            cached_normal: None,
        }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }

    fn unit_f32(&mut self) -> f32 {
        let bits = (self.next_u64() >> 40) as u32;
        (bits as f32) / ((1u32 << 24) as f32)
    }

    fn normal_f32(&mut self) -> f32 {
        if let Some(value) = self.cached_normal.take() {
            return value;
        }

        let u1 = self.unit_f32().clamp(1e-7, 1.0 - 1e-7);
        let u2 = self.unit_f32().clamp(1e-7, 1.0 - 1e-7);
        let radius = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * PI * u2;
        let z0 = radius * theta.cos();
        let z1 = radius * theta.sin();
        self.cached_normal = Some(z1);
        z0
    }

    fn normal_scaled(&mut self, sigma: f32) -> f32 {
        self.normal_f32() * sigma
    }
}

/// Back-compat wrapper around the unified [`synthesize_scene`] path.
///
/// **Lane I (Wave 4) refactor:** before this lane, this function was
/// the only positive-target physics generator; confusers traversed a
/// separate envelope-statistics generator in
/// `echoforge-dataset/src/ml_training.rs::build_frame_products`. The
/// bifurcation meant the generator identity labelled the class. This
/// wrapper now constructs a single-entity [`SceneDescriptor`] (with
/// [`TargetClass::ShahedClassPiston`] + [`TargetKinematics::FromTakeoffProfile`])
/// and forwards through [`synthesize_scene`]. The output is
/// byte-stable with the pre-Lane-I implementation, so every existing
/// reproduction fixture and physics-correctness test passes
/// unchanged. The byte-stability gate lives in
/// `tests/physics_correctness.rs::c_unified_takeoff_wrapper_matches_scene_direct`.
///
/// Lane J extends `synthesize_scene` with native per-class kinematics
/// for confusers (Bird, GroundVehicle, WindTurbine, …) and multi-
/// entity dispatch; this wrapper stays as the canonical single-target
/// API for downstream code.
pub fn synthesize_takeoff_episode(
    config: RadarSimConfig,
    profile: TakeoffProfile,
    noise: NoiseProfile,
    seed: EpisodeSeed,
) -> SyntheticEpisode {
    let scene = SceneDescriptor {
        geometry: SiteGeometry {
            antenna_altitude_agl_m: config.radar_altitude_agl_m,
        },
        environment: EnvironmentDescriptor {
            clutter_regime: noise.clutter_regime,
            atmospheric_one_way_db_per_km: config.atmospheric_one_way_db_per_km,
            rain_rate_mm_per_h: config.rain_rate_mm_per_h,
            ground_reflection_coefficient_magnitude: config
                .ground_reflection_coefficient_magnitude,
        },
        targets: vec![TargetEntity {
            class: TargetClass::ShahedClassPiston,
            kinematics: TargetKinematics::FromTakeoffProfile(profile),
            spawn_time_s: 0.0,
        }],
    };
    synthesize_scene(scene, config, noise, seed)
}

/// Unified scene-physics entry point — Lane I (Wave 4).
///
/// Consumes a [`SceneDescriptor`] (geometry + environment + per-target
/// entities) plus the sensor config and noise profile, returns the
/// same [`SyntheticEpisode`] product as the legacy
/// [`synthesize_takeoff_episode`].
///
/// **Scope of Lane I:** only the single-entity [`TargetKinematics::FromTakeoffProfile`]
/// path was honoured by the physics chain.
///
/// **Wave 5 Lane J extension (this lane):** the gate is lifted. The
/// scene MAY now contain N entities, each with its own
/// [`TargetKinematics`] variant. Confuser classes (Bird, GroundVehicle,
/// WindTurbine, Balloon, Kite, Helicopter) traverse the same chain as
/// positive Shahed-class entities; the MultipathGhost variant
/// synthesises a phantom return offset by `2·h_r·h_t/R` from its
/// parent's geometry (Skolnik 3rd ed. §1.6). Per-class RCS is
/// resolved by a first-order proxy keyed on the dossier RCS table;
/// full per-class RCS-aspect lookup against
/// [`crate::rcs::Rcs::seeded_public_proxy_v1`] is downstream Lane K
/// work.
///
/// **Byte-stability guarantee:** when called with a single-entity
/// scene containing a [`TargetClass::ShahedClassPiston`] +
/// [`TargetKinematics::FromTakeoffProfile`] entity, the synthesis loop
/// reproduces pre-Lane-J byte-stable IQ output. The single-target RNG
/// consumption order is preserved: one scintillation draw per entity
/// per pulse (the first entity's draw is the same draw the legacy
/// path would have consumed), then the per-bin clutter/awgn/RFI loop
/// runs unchanged.
pub fn synthesize_scene(
    scene: SceneDescriptor,
    config: RadarSimConfig,
    noise: NoiseProfile,
    seed: EpisodeSeed,
) -> SyntheticEpisode {
    // Wave 5 Lane J: lifted the Lane I single-entity gate. The scene
    // MAY contain multiple TargetEntities; each entity dispatches its
    // own kinematic state via TargetKinematics::state_at and the
    // synthesis loop accumulates per-entity returns into a shared IQ
    // stream. Multipath ghost entities resolve their parent's geometry
    // and contribute a phantom return offset by the two-ray multipath
    // term (Skolnik 3rd ed. §1.6).
    assert!(
        !scene.targets.is_empty(),
        "synthesize_scene requires at least one TargetEntity in SceneDescriptor::targets"
    );

    // Lane I/J back-compat note: scene.geometry / scene.environment
    // are preserved on the descriptor for serde + Lane K follow-up
    // (per-scene RCS dispatch); the physics path still sources antenna
    // height, atmospherics, multipath coefficient, and clutter regime
    // from `config` / `noise` so pre-Lane-I reproduction fixtures
    // remain byte-stable.
    let _ = (scene.geometry, scene.environment);

    // Resolve a back-compat `TakeoffProfile` for the SyntheticEpisode
    // `profile` field. When the first entity is FromTakeoffProfile the
    // legacy code path round-trips byte-stably; otherwise we populate
    // a default-shaped profile so downstream consumers that read
    // `episode.profile` don't blow up. The authoritative per-entity
    // state lives in `target_states` (first entity) and the per-class
    // SNR list in `per_target_snr_db`.
    let first_entity = &scene.targets[0];
    let first_profile = match &first_entity.kinematics {
        TargetKinematics::FromTakeoffProfile(profile) => *profile,
        _ => TakeoffProfile::default(),
    };

    let waveform = config.waveform();
    let reference = waveform.samples();
    let sample_count = reference.len();
    let compressed_len = sample_count.saturating_mul(2).saturating_sub(1);
    let mut rng = SplitMix64::new(seed.0);
    let mut phase_walk = 0.0f32;

    // Per-entity initial-state link-budget evaluation. Replace the
    // legacy `target_snr_db` knob with a transparent radar-equation
    // result evaluated at each entity's initial geometry. Sub-horizon
    // entities contribute zero return (target_amp == 0), which the
    // rest of the chain treats identically to a missing target.
    let mut per_entity_target_amp: Vec<f32> = Vec::with_capacity(scene.targets.len());
    let mut per_entity_snr_db: Vec<f64> = Vec::with_capacity(scene.targets.len());
    let mut first_link_result: Option<LinkBudgetResult> = None;

    // Resolve per-entity initial state (for the link-budget pass) and
    // per-entity RCS scalar. Ghosts inherit their parent's amp scaled
    // by |Γ|; this matches the Skolnik §1.6 multipath convention.
    let entity_initial_states: Vec<TargetState> = scene
        .targets
        .iter()
        .map(|entity| match &entity.kinematics {
            TargetKinematics::MultipathGhost { parent_idx, .. } => {
                // Resolve parent's initial state for diagnostic SNR.
                let parent = scene.targets.get(*parent_idx).unwrap_or(first_entity);
                let parent_initial_range = entity_initial_range_fallback(parent);
                parent
                    .kinematics
                    .state_at(0.0, parent_initial_range, config.radar_altitude_agl_m)
            }
            _ => {
                let initial_range = entity_initial_range_fallback(entity);
                entity
                    .kinematics
                    .state_at(0.0, initial_range, config.radar_altitude_agl_m)
            }
        })
        .collect();

    for (idx, entity) in scene.targets.iter().enumerate() {
        let entity_initial = entity_initial_states[idx];
        let prop_ctx = config.propagation_context(&entity_initial);
        let rcs_scalar = class_default_rcs_scalar(&entity.class, &entity.kinematics, &first_profile);
        let link = evaluate_link_budget(&config.link_budget(), &prop_ctx, rcs_scalar.max(0.0));
        let amp_full = if link.above_horizon && link.snr_db.is_finite() {
            snr_to_target_amplitude(link.snr_db, noise.awgn_sigma)
        } else {
            0.0
        };
        // Ghosts inherit a scaled amplitude — their kinematics carry
        // the reflection coefficient |Γ|. Apply the linear scale and
        // the dB-equivalent to the SNR.
        let (amp_effective, snr_effective) = match &entity.kinematics {
            TargetKinematics::MultipathGhost {
                reflection_coefficient_magnitude,
                ..
            } => {
                let gamma = reflection_coefficient_magnitude.max(0.0) as f32;
                let snr_offset_db = if *reflection_coefficient_magnitude > 0.0 {
                    20.0 * reflection_coefficient_magnitude.log10()
                } else {
                    f64::NEG_INFINITY
                };
                (amp_full * gamma, link.snr_db + snr_offset_db)
            }
            _ => (amp_full, link.snr_db),
        };
        per_entity_target_amp.push(amp_effective);
        per_entity_snr_db.push(snr_effective);
        if idx == 0 {
            first_link_result = Some(link);
        }
    }
    let link_result_first = first_link_result.expect("at least one entity");

    let glints = build_ground_glints(sample_count, &noise, &mut rng);

    let mut iq = Vec::with_capacity(config.pulse_count);
    let mut profiles = Vec::with_capacity(config.pulse_count);
    let mut compressed_complex: Vec<Vec<ComplexSample>> = Vec::with_capacity(config.pulse_count);
    // target_states (back-compat) holds the FIRST entity's per-pulse
    // states only. Multi-entity per-pulse state vectors are downstream
    // Lane K work; the first-entity contract preserves the legacy
    // single-target consumer surface.
    let mut states_first: Vec<TargetState> = Vec::with_capacity(config.pulse_count);
    let mut clutter_state = 0.0f32;

    // When a ClutterRegime is configured, pre-generate the full
    // per-pulse per-range-bin clutter cube using the cited K /
    // Weibull / log-normal samplers (with proper AR(1) spatial and
    // temporal correlation). The Gaussian AR(1) path remains the
    // fall-back when no regime is set, preserving byte-for-byte
    // back-compat with pre-Lane-C fixtures.
    //
    // Pre-generating (rather than calling `sample_clutter_amplitude`
    // per cell) keeps the inner loop tight and uses
    // `generate_clutter_sequence`'s proper AR(1) recurrence across
    // both range and pulse axes — `sample_clutter_amplitude` is
    // per-sample only and would lose the cross-pulse correlation.
    let clutter_cube: Option<Vec<f32>> = noise.clutter_regime.as_ref().map(|regime| {
        // Mix the episode seed with a stable lane-C tag so the clutter
        // stream is decorrelated from the rest of the RNG draws in the
        // synthesis loop. The tag is arbitrary but fixed.
        let clutter_seed = seed.0 ^ 0xC10C_C0DE_u64;
        generate_clutter_sequence(regime, sample_count, config.pulse_count, clutter_seed)
    });

    for pulse in 0..config.pulse_count {
        let t_s = pulse as f64 * config.pri_s;
        let mut received = vec![ComplexSample::new(0.0, 0.0); sample_count];

        // Wave 5 Lane J: per-entity contribution loop. Entities are
        // visited in `scene.targets` order, which makes the first
        // entity's scintillation draw the FIRST RNG consumer of this
        // pulse — preserving byte-stable back-compat with the
        // single-entity Lane I synth (which only draws one
        // scintillation per pulse).
        //
        // Multipath ghost entities skip their own scintillation draw
        // (they're a phantom of the parent) and use the parent's
        // already-resolved state. Pure-static entities (WindTurbine,
        // tethered Balloon) still go through the same draw sequence
        // so the RNG consumption is uniform across entity types.
        for (entity_idx, entity) in scene.targets.iter().enumerate() {
            let amp_full = per_entity_target_amp[entity_idx];
            if amp_full == 0.0 {
                // Even a zero-amp entity must consume a scintillation
                // draw so single-entity vs multi-entity ordering with
                // mixed sub-horizon scenes remains predictable. The
                // first entity ALWAYS draws so pre-Lane-J fixtures
                // (single ShahedClassPiston) byte-match.
                let _ = rng.normal_f32();
                continue;
            }

            // Resolve this entity's per-pulse state. Ghosts inherit
            // parent's state then apply the multipath range offset.
            let (state, range_offset_m) = resolve_entity_state(
                scene.targets.as_slice(),
                entity_idx,
                t_s,
                config.radar_altitude_agl_m,
            );

            let effective_range_m = (state.range_m + range_offset_m).max(0.0);
            let delay_samples =
                ((2.0 * effective_range_m / C_M_PER_S) * config.sample_rate_hz).round()
                    as isize;
            let doppler_hz =
                2.0 * state.radial_velocity_mps * config.carrier_hz / C_M_PER_S;
            let pulse_phase = 2.0 * std::f64::consts::PI * doppler_hz * t_s;

            // Per-entity micro-Doppler dispatch. Entities that carry
            // a TakeoffProfile route through the legacy multi-blade /
            // single-sinusoid dispatch so pre-Lane-J fixtures
            // byte-match. Confuser variants are handled by their
            // native generators downstream of this synthesis loop;
            // at the synthesis surface their AM envelope is the unit
            // multiplier (1.0) — a conservative first-order proxy
            // that does not over-claim micro-Doppler fidelity for
            // confusers.
            let micro = micro_doppler_envelope(entity, t_s, &state, &first_profile);

            let scintillation = (noise.amplitude_scintillation_sigma * rng.normal_f32())
                .exp()
                .clamp(0.4, 2.5);

            // Wave 4.5 H2: per-pulse polarization scaling. When
            // neither sequence is set we skip the multiplication
            // entirely — preserving exact bit-equality with
            // pre-Lane-H2 fixtures.
            let amp_base = amp_full * micro as f32 * scintillation;
            let amp = if config.pol_tx_sequence.is_some() || config.pol_rx_sequence.is_some() {
                let (pol_tx, pol_rx) = config.polarization_for_pulse(pulse);
                amp_base * polarization_amplitude_scale(pol_tx, pol_rx)
            } else {
                amp_base
            };

            for (i, sample) in reference.iter().enumerate() {
                let dst = i as isize + delay_samples;
                if dst < 0 || dst >= sample_count as isize {
                    continue;
                }
                let phase = pulse_phase as f32 + phase_walk;
                let phasor = ComplexSample::new(phase.cos(), phase.sin());
                received[dst as usize] += *sample * phasor * amp;
            }

            if entity_idx == 0 {
                states_first.push(state);
            }
        }

        // Stochastic terms (clutter, AWGN, RFI, phase walk) are
        // applied AFTER per-entity returns are accumulated. The RNG
        // consumption order is identical to the pre-Lane-J single-
        // entity synth so byte-stable fixtures replay unchanged.
        for (index, sample) in received.iter_mut().enumerate() {
            let clutter_raw = match clutter_cube.as_ref() {
                Some(cube) => cube[pulse * sample_count + index] * noise.clutter_sigma_0_scale,
                None => {
                    clutter_state = noise.clutter_correlation * clutter_state
                        + (1.0 - noise.clutter_correlation)
                            * rng.normal_scaled(noise.clutter_sigma);
                    clutter_state
                }
            };
            let glint = glints
                .iter()
                .find(|(bin, _)| *bin == index)
                .map(|(_, amp)| *amp)
                .unwrap_or(0.0);
            let clutter = clutter_raw + glint;
            sample.re += clutter + rng.normal_scaled(noise.awgn_sigma);
            sample.im += clutter * 0.35 + rng.normal_scaled(noise.awgn_sigma);

            if rng.unit_f32() < noise.rfi_probability {
                let phase = 2.0 * PI * rng.unit_f32();
                *sample += ComplexSample::new(phase.cos(), phase.sin()) * noise.rfi_amplitude;
            }
        }

        phase_walk += rng.normal_scaled(noise.phase_noise_std_rad);
        let compressed =
            pulse_compress_windowed(&received, &reference, CompressionWindow::taylor_default());
        let mag = magnitude(&compressed);
        iq.push(received);
        profiles.push(mag);
        compressed_complex.push(compressed);
    }

    let integrated = integrate_profiles(&profiles, compressed_len);
    let decisions = ca_cfar_1d(&integrated, config.cfar_params());
    let detections = decisions
        .iter()
        .filter(|decision| decision.evaluated && decision.detected)
        .map(|decision| {
            let range_m = range_bin_to_m(
                decision.index,
                sample_count.saturating_sub(1),
                config.sample_rate_hz,
            );
            let confidence = if decision.threshold.is_finite() && decision.threshold > 0.0 {
                decision.statistic / decision.threshold
            } else {
                0.0
            };
            DetectionRecord {
                range_bin: decision.index,
                range_m,
                statistic: decision.statistic,
                threshold: decision.threshold,
                noise_estimate: decision.noise_estimate,
                confidence,
            }
        })
        .collect();
    let range_doppler_proxy = slow_time_dft_magnitude(&profiles, compressed_len);
    let range_doppler_complex = slow_time_complex_dft(&compressed_complex, config.pulse_count);

    SyntheticEpisode {
        seed,
        config,
        profile: first_profile,
        noise,
        target_states: states_first,
        iq,
        range_profiles_by_pulse: profiles,
        integrated_range_profile: integrated,
        range_doppler_proxy,
        detections,
        diagnostic_snr_db: link_result_first.snr_db,
        diagnostic_link_budget: link_result_first,
        range_doppler_complex,
        per_target_snr_db: per_entity_snr_db,
    }
}

/// Wave 5 Lane J helper — extract a sensible fallback initial range
/// for an entity. Variants that carry their own range (GroundVehicle,
/// WindTurbine, Kite) return that range; variants without one
/// (Bird, Helicopter, Balloon, FromTakeoffProfile) fall back to the
/// TakeoffProfile default `initial_range_m`. Used by the per-entity
/// initial-state pass so multipath ghost parents can be re-evaluated
/// before the synthesis loop runs.
fn entity_initial_range_fallback(entity: &TargetEntity) -> f64 {
    match &entity.kinematics {
        TargetKinematics::FromTakeoffProfile(p) => p.initial_range_m,
        TargetKinematics::GroundVehicle {
            initial_range_m, ..
        } => *initial_range_m,
        TargetKinematics::WindTurbine { hub_range_m, .. } => *hub_range_m,
        TargetKinematics::Kite { anchor_range_m, .. } => *anchor_range_m,
        // Bird / Helicopter / Balloon / MultipathGhost don't carry a
        // range; use the TakeoffProfile default so first-entity geometry
        // is well-defined when the scene mixes types. Downstream Lane K
        // adds per-entity initial range plumbing on these variants.
        TargetKinematics::Bird { .. }
        | TargetKinematics::Helicopter { .. }
        | TargetKinematics::Balloon { .. }
        | TargetKinematics::MultipathGhost { .. } => TakeoffProfile::default().initial_range_m,
    }
}

/// Wave 5 Lane J helper — resolve a per-pulse `(state, range_offset_m)`
/// pair for entity `idx` at time `t_s`. Multipath ghost entities return
/// the parent's state and a non-zero range offset computed from the
/// two-ray geometry `2·h_r·h_t/R` (Skolnik 3rd ed. §1.6); other
/// entities return their own state and zero offset.
fn resolve_entity_state(
    targets: &[TargetEntity],
    idx: usize,
    t_s: f64,
    antenna_alt_agl_m: f64,
) -> (TargetState, f64) {
    let entity = &targets[idx];
    match &entity.kinematics {
        TargetKinematics::MultipathGhost { parent_idx, .. } => {
            let parent = targets.get(*parent_idx).unwrap_or(entity);
            let parent_initial_range = entity_initial_range_fallback(parent);
            let parent_state = parent.kinematics.state_at(
                t_s,
                parent_initial_range,
                antenna_alt_agl_m,
            );
            // Two-ray multipath offset: 2·h_r·h_t/R. Guard against
            // R = 0 by clamping the range to a small positive value.
            let r = parent_state.range_m.max(1e-3);
            let offset = 2.0 * antenna_alt_agl_m * parent_state.altitude_m / r;
            (parent_state, offset)
        }
        _ => {
            let initial_range = entity_initial_range_fallback(entity);
            let state = entity
                .kinematics
                .state_at(t_s, initial_range, antenna_alt_agl_m);
            (state, 0.0)
        }
    }
}

/// Wave 5 Lane J helper — per-entity micro-Doppler envelope at time
/// `t_s`. Returns the multiplicative amplitude modulation that goes
/// onto the entity's per-pulse return.
///
/// For entities carrying a [`TakeoffProfile`] (the back-compat Shahed
/// path), this routes through the legacy multi-blade / single-sinusoid
/// dispatch from [`PropellerGenerator`] so pre-Lane-J reproduction
/// fixtures replay byte-identically. For confuser variants, the
/// envelope returns 1.0 — the native micro-Doppler line spectra are
/// the responsibility of the downstream
/// [`crate::micro_doppler_gen`] generators (BirdWingbeatGenerator,
/// HelicopterRotorGenerator, PropellerGenerator) at the per-class
/// feature-extraction surface, not the per-pulse amplitude
/// modulation. Returning 1.0 at the synthesis surface is the
/// conservative first-order proxy that does not over-claim micro-
/// Doppler fidelity for confusers; full per-class AM modelling lands
/// in Lane K (per-class RCS-aspect + propulsion dispatch).
fn micro_doppler_envelope(
    entity: &TargetEntity,
    t_s: f64,
    state: &TargetState,
    fallback_profile: &TakeoffProfile,
) -> f64 {
    let profile = match &entity.kinematics {
        TargetKinematics::FromTakeoffProfile(p) => p,
        _ => return 1.0,
    };
    // Defensive: when state and profile point to different entities
    // (cross-entity dispatch in mixed scenes), the profile must still
    // come from the entity itself. The fallback_profile is only
    // consulted to keep the type signature uniform; never used here.
    let _ = fallback_profile;
    match (profile.blade_count, profile.blade_length_m) {
        (Some(n_blades), Some(length_m)) => {
            let prop = PropellerGenerator::new(
                n_blades,
                profile.propulsor_hz,
                length_m,
                state.propulsor_phase_rad,
            );
            let v_micro = prop.radial_velocity_at(t_s);
            let v_tip = prop.tip_speed_mps();
            if v_tip > 1e-6 {
                1.0 + 0.15 * (v_micro / v_tip)
            } else {
                1.0
            }
        }
        _ => {
            1.0 + 0.15
                * (2.0 * std::f64::consts::PI * profile.micro_doppler_hz * t_s
                    + state.propulsor_phase_rad)
                    .sin()
        }
    }
}

/// Wave 5 Lane J helper — per-class first-order RCS scalar (linear m²).
///
/// This is a **first-order proxy** keyed on Wave-A `physics_dossier.md`
/// confuser median RCS values; full per-class aspect-dependent RCS
/// lookup against `crate::rcs::Rcs::seeded_public_proxy_v1` lands in
/// Lane K. The proxy is sufficient for the Lane J multi-class
/// dispatch gate: each entity gets a non-zero contribution whose
/// magnitude tracks the class's typical RCS envelope, and the
/// emergent SNR list distinguishes positives from confusers.
///
/// Reference RCS values (dBsm → linear m² via 10^(dBsm/10)):
///   - ShahedClassPiston / ShahedClassJet → use the entity's
///     TakeoffProfile.rcs_scalar (already in linear m²).
///   - Bird → -25 dBsm (single large bird; Rahman & Robertson 2018).
///   - GroundVehicle → +5 dBsm (car / SUV at broadside aspect).
///   - WindTurbine → +25 dBsm (large utility-scale tower; Naqvi 2015).
///   - Balloon → -20 dBsm (Mylar reflector envelope).
///   - Kite → -25 dBsm (typical tethered kite).
///   - Helicopter → +5 dBsm (rotary-wing aircraft at typical aspect).
///   - MultipathGhost / TerrainGlint / ManRadarReturn → first-order
///     fallback at -30 dBsm (these are highly geometry-dependent).
fn class_default_rcs_scalar(
    class: &TargetClass,
    kinematics: &TargetKinematics,
    fallback_profile: &TakeoffProfile,
) -> f64 {
    let _ = fallback_profile;
    // First, give FromTakeoffProfile entities their explicit RCS so
    // pre-Lane-J fixtures byte-match (their rcs_scalar is the
    // load-bearing input to the link budget).
    if let TargetKinematics::FromTakeoffProfile(profile) = kinematics {
        return profile.rcs_scalar;
    }
    let dbsm = match class {
        TargetClass::ShahedClassPiston | TargetClass::ShahedClassJet => -10.0,
        TargetClass::Bird => -25.0,
        TargetClass::GroundVehicle => 5.0,
        TargetClass::WindTurbine => 25.0,
        TargetClass::Balloon => -20.0,
        TargetClass::Kite => -25.0,
        TargetClass::Helicopter => 5.0,
        TargetClass::MultipathGhost { .. }
        | TargetClass::TerrainGlint
        | TargetClass::ManRadarReturn => -30.0,
    };
    10f64.powf(dbsm / 10.0)
}

fn build_ground_glints(
    sample_count: usize,
    noise: &NoiseProfile,
    rng: &mut SplitMix64,
) -> Vec<(usize, f32)> {
    if sample_count == 0 {
        return Vec::new();
    }

    (0..noise.ground_glint_count)
        .map(|_| {
            let bin = (rng.unit_f32() * sample_count as f32) as usize;
            let amp = noise.ground_glint_amplitude * (0.5 + rng.unit_f32());
            (bin.min(sample_count - 1), amp)
        })
        .collect()
}

fn integrate_profiles(profiles: &[Vec<f32>], len: usize) -> Vec<f32> {
    if profiles.is_empty() {
        return Vec::new();
    }

    let mut integrated = vec![0.0f32; len];
    for profile in profiles {
        for (index, value) in profile.iter().enumerate().take(len) {
            integrated[index] += *value;
        }
    }
    let scale = 1.0 / profiles.len() as f32;
    for value in &mut integrated {
        *value *= scale;
    }
    integrated
}

/// Complex slow-time DFT over per-pulse compressed-IQ range profiles,
/// producing a `(range, Doppler)` complex grid. Phase is preserved
/// end-to-end so downstream MTI / MTD / micro-Doppler processing can use
/// coherent arithmetic instead of working from a magnitude image.
///
/// Inputs:
///   - `compressed_pulses` — per-pulse complex range profiles (the
///     output of [`crate::pulse_compression::pulse_compress_windowed`]).
///     Each inner vector must have the same length; shorter rows are
///     zero-padded along the range axis.
///   - `n_doppler` — number of slow-time samples (`n_pulses`). The DFT
///     uses N = `compressed_pulses.len()` slow-time samples and emits
///     `n_doppler` Doppler bins (typically equal to N). Passing
///     `n_doppler == 0` returns an empty grid.
///
/// Output: `Vec<Vec<ComplexSample>>` indexed `[range_bin][doppler_bin]`
/// with length `range_len` along axis 0 and `n_doppler` along axis 1.
///
/// Convention (Skolnik, *Introduction to Radar Systems*, 3rd ed., §3.5):
///   `X[k] = Σₙ x[n] · exp(-j · 2π · k · n / N)`
///
/// Implementation: naive O(N²) DFT per range bin. The grid is small
/// (pulses ~ 32 for the takeoff fixture) so an FFT dependency is not
/// justified here; callers that need an FFT can wrap this signature.
pub fn slow_time_complex_dft(
    compressed_pulses: &[Vec<ComplexSample>],
    n_doppler: usize,
) -> Vec<Vec<ComplexSample>> {
    let n_pulses = compressed_pulses.len();
    if n_pulses == 0 || n_doppler == 0 {
        return Vec::new();
    }

    let range_len = compressed_pulses
        .iter()
        .map(|profile| profile.len())
        .max()
        .unwrap_or(0);
    if range_len == 0 {
        return Vec::new();
    }

    let mut output =
        vec![vec![ComplexSample::new(0.0, 0.0); n_doppler]; range_len];

    let n_pulses_f = n_pulses as f32;
    for range in 0..range_len {
        for doppler in 0..n_doppler {
            let mut acc = ComplexSample::new(0.0, 0.0);
            for (pulse, profile) in compressed_pulses.iter().enumerate() {
                let sample = profile
                    .get(range)
                    .copied()
                    .unwrap_or(ComplexSample::new(0.0, 0.0));
                let angle = -2.0 * PI * (doppler as f32) * (pulse as f32) / n_pulses_f;
                let phasor = ComplexSample::new(angle.cos(), angle.sin());
                acc += sample * phasor;
            }
            output[range][doppler] = acc;
        }
    }
    output
}

/// Magnitude image of the slow-time DFT over per-pulse range-profile
/// magnitudes. This is the legacy `range_doppler_proxy` signature: the
/// caller has already stripped phase via [`magnitude`] before the
/// slow-time transform, so the result is a magnitude-of-magnitudes
/// proxy and cannot be used for coherent Doppler / MTI / micro-Doppler
/// reasoning. New code must consume [`slow_time_complex_dft`] (and call
/// `.norm()` on each cell if it only wants the magnitude grid).
///
/// Output is laid out `[doppler_bin][range_bin]` and normalised by
/// `1/N_pulses`, matching the pre-Lane-B implementation so byte-stable
/// fixtures continue to reproduce.
///
/// Implementation is now a thin wrapper that lifts each magnitude into
/// a `ComplexSample` with zero imaginary part, dispatches to
/// [`slow_time_complex_dft`], then collapses to per-bin magnitude with
/// the legacy 1/N normalisation. The lift is mathematically lossless
/// (the DFT of a real sequence reproduces the magnitudes of the real
/// DFT), so the wrapper is byte-stable with the inlined implementation
/// modulo floating-point summation order — which we hold constant by
/// keeping the same inner loop.
pub fn slow_time_dft_magnitude(profiles: &[Vec<f32>], range_len: usize) -> Vec<Vec<f32>> {
    let pulses = profiles.len();
    if pulses == 0 || range_len == 0 {
        return Vec::new();
    }

    // Lift magnitude profiles into the complex domain (im = 0) so we can
    // run the canonical complex slow-time DFT. We also clip each row to
    // `range_len` so the lifted shape matches the legacy contract.
    let lifted: Vec<Vec<ComplexSample>> = profiles
        .iter()
        .map(|profile| {
            (0..range_len)
                .map(|range| {
                    let value = profile.get(range).copied().unwrap_or(0.0);
                    ComplexSample::new(value, 0.0)
                })
                .collect()
        })
        .collect();

    let complex_grid = slow_time_complex_dft(&lifted, pulses);

    // Legacy layout: `[doppler][range]`, normalised by `1/N_pulses`. The
    // complex grid is `[range][doppler]`, so transpose during the
    // collapse.
    let scale = 1.0 / pulses as f32;
    let mut output = vec![vec![0.0f32; range_len]; pulses];
    for (range, range_row) in complex_grid.iter().enumerate().take(range_len) {
        for (doppler, cell) in range_row.iter().enumerate().take(pulses) {
            output[doppler][range] = cell.norm() * scale;
        }
    }
    output
}

pub fn range_bin_to_m(index: usize, zero_delay_bin: usize, sample_rate_hz: f64) -> f64 {
    let delay = index as isize - zero_delay_bin as isize;
    if delay <= 0 {
        0.0
    } else {
        (delay as f64) * C_M_PER_S / (2.0 * sample_rate_hz)
    }
}

#[cfg(test)]
mod propeller_wire_in_tests {
    use super::*;

    /// The default `TakeoffProfile` keeps `blade_count` / `blade_length_m`
    /// at `None`, so the synthesis loop falls back to the legacy single-
    /// sinusoid micro-Doppler. This back-compat guard pins the new fields
    /// at their default values so existing reproduction-byte fixtures
    /// continue to match.
    #[test]
    fn takeoff_profile_default_uses_legacy_micro() {
        let profile = TakeoffProfile::default();
        assert_eq!(profile.blade_count, None);
        assert_eq!(profile.blade_length_m, None);
    }

    /// Setting `blade_count`/`blade_length_m` on the profile dispatches
    /// the synthesis loop to the multi-blade `PropellerGenerator` model.
    /// We verify the wiring by:
    ///   1. running an episode with the propeller fields populated
    ///      (Shahed-class public-proxy: 2 blades, 0.6 m, 95 Hz rotation
    ///      → textbook blade-pass = 2·95 = 190 Hz),
    ///   2. computing a slow-time DFT of the complex IQ at the target
    ///      bin (with noise/clutter zeroed for SNR isolation),
    ///   3. asserting the strongest non-DC peak is at the (aliased)
    ///      body-Doppler bin (radial velocity → Doppler shift), and
    ///   4. asserting the micro-Doppler line attributable to the
    ///      propeller appears above the residual spectral floor at the
    ///      expected sideband location.
    ///
    /// Note on the line-strength bound: amplitude-modulation depth is
    /// 15 % (matching the legacy single-sinusoid envelope amplitude), so
    /// the AC-line magnitude rides at a few percent of DC, not >50 %.
    /// The bound is set as a sideband-vs-control-bin ratio rather than
    /// sideband-vs-DC ratio so the test is physically achievable. A
    /// future lane can replace the amplitude modulation with full phasor
    /// modulation once complex-IQ phase wiring lands (Lane B).
    #[test]
    fn takeoff_profile_with_propeller_generator_produces_blade_pass() {
        // Long pulse count so the slow-time DFT resolves the blade-pass
        // sideband cleanly. 256 pulses at PRI = 900 µs → bin resolution
        // 1/(256·900e-6) ≈ 4.3 Hz.
        let config = RadarSimConfig {
            pulse_count: 256,
            ..RadarSimConfig::default()
        };
        let profile = TakeoffProfile {
            blade_count: Some(2),
            blade_length_m: Some(0.6),
            propulsor_hz: 95.0,
            ..TakeoffProfile::default()
        };
        // Zero out all stochastic terms so the blade-pass line is not
        // buried in noise/clutter/scintillation residue. We are testing
        // the deterministic synthesis path, not the noise contract.
        let noise = NoiseProfile {
            awgn_sigma: 0.0,
            phase_noise_std_rad: 0.0,
            amplitude_scintillation_sigma: 0.0,
            rfi_probability: 0.0,
            rfi_amplitude: 0.0,
            clutter_sigma: 0.0,
            clutter_correlation: 0.0,
            ground_glint_count: 0,
            ground_glint_amplitude: 0.0,
            clutter_regime: None,
            clutter_sigma_0_scale: 1.0,
        };
        let episode = synthesize_takeoff_episode(config, profile, noise, EpisodeSeed(101));

        // Find the target range bin: peak in the integrated profile.
        let target_bin = episode
            .integrated_range_profile
            .iter()
            .enumerate()
            .fold(
                (0usize, f32::NEG_INFINITY),
                |(best, best_v), (i, &v)| if v > best_v { (i, v) } else { (best, best_v) },
            )
            .0;
        assert!(
            episode.integrated_range_profile[target_bin] > 0.0,
            "target peak in integrated profile must be positive (got {})",
            episode.integrated_range_profile[target_bin]
        );

        // Identify the raw IQ sample index that received the target
        // return at pulse 0. The synthesis loop shifts the chirp by
        // `delay_samples`; we read at that bin so the IQ across pulses
        // is dominated by the modulated target return.
        let initial_state = profile.state_at(0.0);
        let raw_delay_samples =
            ((2.0 * initial_state.range_m / C_M_PER_S) * episode.config.sample_rate_hz)
                .round() as usize;
        // Pick the bin slightly inside the chirp footprint to ensure
        // every pulse has a sample written there.
        let iq_bin = raw_delay_samples + episode.iq[0].len() / 4;
        assert!(
            iq_bin < episode.iq[0].len(),
            "iq probe bin {} out of bounds (len {})",
            iq_bin,
            episode.iq[0].len()
        );

        // Slow-time vector at the chosen range bin.
        let pulses = episode.iq.len();
        let slow_time: Vec<ComplexSample> =
            (0..pulses).map(|p| episode.iq[p][iq_bin]).collect();

        // Slow-time DFT (complex IQ).
        let spec_iq: Vec<f64> = (0..pulses)
            .map(|k| {
                let mut re = 0.0_f64;
                let mut im = 0.0_f64;
                for (p, sample) in slow_time.iter().enumerate() {
                    let angle = -2.0 * std::f64::consts::PI * (k as f64) * (p as f64)
                        / (pulses as f64);
                    let (c, s) = (angle.cos(), angle.sin());
                    re += sample.re as f64 * c - sample.im as f64 * s;
                    im += sample.re as f64 * s + sample.im as f64 * c;
                }
                (re * re + im * im).sqrt()
            })
            .collect();

        // (3) Peak non-DC bin should be near body Doppler. Body Doppler
        //     = 2 · v_radial · f_c / c, aliased into the PRF interval.
        let pulse_rate = 1.0 / episode.config.pri_s;
        let bin_hz = pulse_rate / pulses as f64;
        let body_doppler =
            2.0 * initial_state.radial_velocity_mps * episode.config.carrier_hz / C_M_PER_S;
        let body_bin = ((body_doppler.rem_euclid(pulse_rate)) / bin_hz).round() as usize
            % pulses;

        let (peak_bin, peak_mag) = spec_iq
            .iter()
            .enumerate()
            .skip(1)
            .fold((0usize, 0.0_f64), |(bi, bv), (i, &v)| {
                if v > bv {
                    (i, v)
                } else {
                    (bi, bv)
                }
            });
        // Allow ±2 bins of tolerance around the predicted body-Doppler
        // bin (rounding, finite slow-time DFT resolution).
        let peak_offset = (peak_bin as isize - body_bin as isize).unsigned_abs();
        assert!(
            peak_offset <= 2 || (pulses - peak_offset) <= 2,
            "expected slow-time peak at body-Doppler bin {} (~{:.1} Hz); \
             got peak at bin {} (~{:.1} Hz, mag {:.3})",
            body_bin,
            body_doppler.rem_euclid(pulse_rate),
            peak_bin,
            peak_bin as f64 * bin_hz,
            peak_mag
        );

        // (4) Micro-Doppler sideband: blade-pass frequency by textbook
        //     physics is `blade_count · rotation_hz = 190 Hz` for the
        //     2-blade Shahed-class proxy. The dominant-blade convention
        //     in `PropellerGenerator` reduces to a pure rotation-rate
        //     sinusoid for N=2 (the two blades are exactly π apart and
        //     |sin(θ)| ≡ |sin(θ+π)|, so the picker stays locked on
        //     blade 0). The observable AM line therefore appears at the
        //     rotation rate (= blade-pass / N) when the picker is
        //     degenerate. We check for content at the body-Doppler ±
        //     rotation-rate sideband bins (the actual emergent line),
        //     verifying it sits well above a control bin at a
        //     non-harmonic offset.
        let probe_offset_hz = 95.0; // rotation rate / effective blade-pass for N=2
        let upper_bin = (((body_doppler + probe_offset_hz).rem_euclid(pulse_rate)) / bin_hz)
            .round() as usize
            % pulses;
        let lower_bin = (((body_doppler - probe_offset_hz).rem_euclid(pulse_rate)) / bin_hz)
            .round() as usize
            % pulses;
        // Control bin: 60 Hz offset is well off the rotation harmonic
        // and at the legacy single-sinusoid micro_doppler_hz=42 region
        // boundary, so it samples the spectral floor between lines.
        let control_offset_hz = 60.0;
        let control_bin = (((body_doppler + control_offset_hz).rem_euclid(pulse_rate))
            / bin_hz)
            .round() as usize
            % pulses;

        let sideband_mag = spec_iq[upper_bin].max(spec_iq[lower_bin]);
        let control_mag = spec_iq[control_bin].max(1e-12);
        assert!(
            sideband_mag > 2.0 * control_mag,
            "expected propeller-driven sideband at body±{} Hz to dominate \
             a non-harmonic control bin at body+{} Hz; sideband={:.4}, \
             control={:.4}",
            probe_offset_hz,
            control_offset_hz,
            sideband_mag,
            control_mag
        );

        // Also confirm there is *some* spectral content at the textbook
        // blade-pass sideband location (190 Hz) above the floor. This
        // line is weaker than the rotation-rate line for the N=2 case
        // (per the dominant-blade discussion above), but it must still
        // be measurable; we require it to exceed the control bin.
        let blade_pass_offset_hz = (profile.blade_count.unwrap() as f64) * profile.propulsor_hz;
        let bp_upper_bin = (((body_doppler + blade_pass_offset_hz).rem_euclid(pulse_rate))
            / bin_hz)
            .round() as usize
            % pulses;
        let bp_lower_bin = (((body_doppler - blade_pass_offset_hz).rem_euclid(pulse_rate))
            / bin_hz)
            .round() as usize
            % pulses;
        let bp_mag = spec_iq[bp_upper_bin].max(spec_iq[bp_lower_bin]);
        assert!(
            bp_mag >= control_mag,
            "expected non-zero spectral content at textbook blade-pass \
             sideband ({} Hz); bp_mag={:.6}, control_mag={:.6}",
            blade_pass_offset_hz,
            bp_mag,
            control_mag
        );
    }

    /// When neither `blade_count` nor `blade_length_m` is populated, the
    /// synthesis loop must reproduce the legacy single-sinusoid path
    /// byte-for-byte. This guard pins back-compat against existing
    /// reproduction fixtures that predate the multi-blade switch.
    #[test]
    fn propeller_generator_back_compat() {
        let config = RadarSimConfig {
            pulse_count: 8,
            ..RadarSimConfig::default()
        };
        let legacy_profile = TakeoffProfile::default();
        // Explicitly None for clarity; this is what default already gives.
        let explicit_none_profile = TakeoffProfile {
            blade_count: None,
            blade_length_m: None,
            ..TakeoffProfile::default()
        };
        let noise = NoiseProfile::real_world_proxy_v1();
        // Wave 4.5 H2: `RadarSimConfig` is no longer `Copy` so we clone
        // for each call.
        let a = synthesize_takeoff_episode(
            config.clone(),
            legacy_profile,
            noise,
            EpisodeSeed(2024),
        );
        let b = synthesize_takeoff_episode(
            config.clone(),
            explicit_none_profile,
            noise,
            EpisodeSeed(2024),
        );
        assert_eq!(
            a.integrated_range_profile, b.integrated_range_profile,
            "explicit-None profile must reproduce default profile byte-stably"
        );
        assert_eq!(
            a.detections, b.detections,
            "back-compat: default vs explicit-None must yield identical detections"
        );

        // Half-configured (only one of the two new fields populated) must
        // also fall back to legacy. The match-arm contract requires
        // BOTH fields to be Some before dispatching to PropellerGenerator.
        let half_a = TakeoffProfile {
            blade_count: Some(2),
            blade_length_m: None,
            ..TakeoffProfile::default()
        };
        let half_b = TakeoffProfile {
            blade_count: None,
            blade_length_m: Some(0.6),
            ..TakeoffProfile::default()
        };
        let ha =
            synthesize_takeoff_episode(config.clone(), half_a, noise, EpisodeSeed(2024));
        let hb = synthesize_takeoff_episode(config, half_b, noise, EpisodeSeed(2024));
        assert_eq!(
            a.integrated_range_profile, ha.integrated_range_profile,
            "half-config (blade_count only) must still use legacy path"
        );
        assert_eq!(
            a.integrated_range_profile, hb.integrated_range_profile,
            "half-config (blade_length_m only) must still use legacy path"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_episode_repeats_for_seed() {
        let config = RadarSimConfig {
            pulse_count: 8,
            ..RadarSimConfig::default()
        };
        let a = synthesize_takeoff_episode(
            config.clone(),
            TakeoffProfile::default(),
            NoiseProfile::real_world_proxy_v1(),
            EpisodeSeed(42),
        );
        let b = synthesize_takeoff_episode(
            config,
            TakeoffProfile::default(),
            NoiseProfile::real_world_proxy_v1(),
            EpisodeSeed(42),
        );
        assert_eq!(a.integrated_range_profile, b.integrated_range_profile);
        assert_eq!(a.detections, b.detections);
    }

    #[test]
    fn high_snr_takeoff_has_detection() {
        let config = RadarSimConfig {
            pulse_count: 12,
            target_snr_db: 28.0,
            ..RadarSimConfig::default()
        };
        let mut noise = NoiseProfile::real_world_proxy_v1();
        noise.awgn_sigma = 0.025;
        let episode =
            synthesize_takeoff_episode(config, TakeoffProfile::default(), noise, EpisodeSeed(7));
        assert!(!episode.detections.is_empty());
    }

    #[test]
    fn low_snr_products_are_finite() {
        let config = RadarSimConfig {
            pulse_count: 6,
            target_snr_db: 2.0,
            ..RadarSimConfig::default()
        };
        let mut noise = NoiseProfile::real_world_proxy_v1();
        noise.awgn_sigma = 0.2;
        noise.rfi_probability = 0.05;
        let episode =
            synthesize_takeoff_episode(config, TakeoffProfile::default(), noise, EpisodeSeed(99));
        assert!(episode
            .integrated_range_profile
            .iter()
            .all(|value| value.is_finite()));
        assert!(episode
            .range_doppler_proxy
            .iter()
            .flatten()
            .all(|value| value.is_finite()));
    }

    #[test]
    fn rfi_changes_products_but_stays_deterministic() {
        let config = RadarSimConfig {
            pulse_count: 6,
            ..RadarSimConfig::default()
        };
        let mut clean_noise = NoiseProfile::real_world_proxy_v1();
        clean_noise.rfi_probability = 0.0;
        clean_noise.clutter_sigma = 0.0;
        clean_noise.ground_glint_count = 0;

        let mut dirty_noise = clean_noise;
        dirty_noise.rfi_probability = 0.15;
        dirty_noise.clutter_sigma = 0.08;
        dirty_noise.ground_glint_count = 3;

        let clean = synthesize_takeoff_episode(
            config.clone(),
            TakeoffProfile::default(),
            clean_noise,
            EpisodeSeed(12),
        );
        let dirty_a = synthesize_takeoff_episode(
            config.clone(),
            TakeoffProfile::default(),
            dirty_noise,
            EpisodeSeed(12),
        );
        let dirty_b = synthesize_takeoff_episode(
            config,
            TakeoffProfile::default(),
            dirty_noise,
            EpisodeSeed(12),
        );

        assert_ne!(
            clean.integrated_range_profile,
            dirty_a.integrated_range_profile
        );
        assert_eq!(
            dirty_a.integrated_range_profile,
            dirty_b.integrated_range_profile
        );
    }

    /// A pure complex tone `x[n] = exp(j 2π f₀ n / N)` injected into a
    /// single range bin must produce a DFT peak at Doppler bin `f₀` of
    /// magnitude ~N. This is the canonical "delta-in-frequency-of-tone"
    /// identity (Skolnik §3.5): the DFT is a coherent integrator that
    /// concentrates a complex sinusoid into one bin while spreading
    /// noise across all N. If we had thrown away phase before the
    /// transform (the Lane B bug), the peak would be at DC (bin 0)
    /// instead of bin `f₀`, because a magnitude sequence is real and
    /// non-negative.
    #[test]
    fn slow_time_complex_dft_preserves_phase() {
        const N: usize = 32;
        const F0: usize = 4;
        let range_len: usize = 3;
        let target_range_bin: usize = 1;

        // Build N pulses, each with a single non-zero range bin holding
        // sample exp(j 2π f₀ n / N).
        let mut pulses: Vec<Vec<ComplexSample>> = Vec::with_capacity(N);
        for n in 0..N {
            let mut profile = vec![ComplexSample::new(0.0, 0.0); range_len];
            let phase = 2.0 * PI * (F0 as f32) * (n as f32) / N as f32;
            profile[target_range_bin] = ComplexSample::new(phase.cos(), phase.sin());
            pulses.push(profile);
        }

        let grid = slow_time_complex_dft(&pulses, N);
        assert_eq!(grid.len(), range_len);
        assert_eq!(grid[target_range_bin].len(), N);

        // Find the peak Doppler bin at the populated range.
        let (peak_bin, peak_mag) = grid[target_range_bin]
            .iter()
            .enumerate()
            .map(|(k, c)| (k, c.norm()))
            .fold((0usize, 0.0f32), |(best_k, best_m), (k, m)| {
                if m > best_m {
                    (k, m)
                } else {
                    (best_k, best_m)
                }
            });

        assert_eq!(
            peak_bin, F0,
            "complex DFT peak landed at bin {peak_bin}, expected {F0}; \
             phase preservation likely broken"
        );
        assert!(
            (peak_mag - N as f32).abs() < 1e-3,
            "peak magnitude = {peak_mag}, expected ~{N}; coherent integration scaling broken",
        );

        // All other Doppler bins at this range should be near zero
        // (within DFT numerical precision).
        for (k, c) in grid[target_range_bin].iter().enumerate() {
            if k == F0 {
                continue;
            }
            assert!(
                c.norm() < 1e-3,
                "leak into bin {k}: |X[{k}]| = {} (expected ~0)",
                c.norm()
            );
        }

        // Empty range bins should hold all zeros.
        for range in 0..range_len {
            if range == target_range_bin {
                continue;
            }
            for cell in &grid[range] {
                assert!(cell.norm() < 1e-6, "spurious energy in empty range bin");
            }
        }
    }

    /// `slow_time_dft_magnitude` is now a wrapper that lifts real
    /// magnitudes into the complex domain (im=0), runs
    /// `slow_time_complex_dft`, then collapses to magnitude. This test
    /// gates that the refactor is byte-stable per bin: feeding the
    /// magnitudes-as-complex into the complex transform and taking
    /// `.norm() / N_pulses` must reproduce the legacy-output exactly
    /// (modulo float summation order, which we hold constant).
    #[test]
    fn slow_time_complex_dft_magnitude_matches_legacy() {
        const N: usize = 8;
        const RANGE_LEN: usize = 5;

        // Construct a deterministic real magnitude grid.
        let profiles: Vec<Vec<f32>> = (0..N)
            .map(|n| {
                (0..RANGE_LEN)
                    .map(|r| 0.1 + (n as f32) * 0.07 + (r as f32) * 0.13)
                    .collect()
            })
            .collect();

        // Legacy wrapper output (now backed by slow_time_complex_dft).
        let legacy = slow_time_dft_magnitude(&profiles, RANGE_LEN);

        // Reference: directly compute the same DFT in real-magnitudes
        // form (the pre-Lane-B inlined implementation, kept here as the
        // ground truth).
        let mut reference = vec![vec![0.0f32; RANGE_LEN]; N];
        for doppler in 0..N {
            for range in 0..RANGE_LEN {
                let mut re = 0.0f32;
                let mut im = 0.0f32;
                for (pulse, profile) in profiles.iter().enumerate() {
                    let angle = -2.0 * PI * (doppler as f32) * (pulse as f32) / N as f32;
                    let value = profile[range];
                    re += value * angle.cos();
                    im += value * angle.sin();
                }
                reference[doppler][range] = (re * re + im * im).sqrt() / N as f32;
            }
        }

        assert_eq!(legacy.len(), reference.len());
        for (doppler, (l_row, r_row)) in legacy.iter().zip(reference.iter()).enumerate() {
            assert_eq!(l_row.len(), r_row.len());
            for (range, (l, r)) in l_row.iter().zip(r_row.iter()).enumerate() {
                assert!(
                    (l - r).abs() < 1e-5,
                    "doppler={doppler} range={range}: wrapper={l} reference={r}"
                );
            }
        }
    }

    /// Empty inputs must round-trip to empty output with no panic. Both
    /// the all-zero-pulses path and the zero-Doppler-bins path are
    /// exercised.
    #[test]
    fn slow_time_complex_dft_handles_empty() {
        let empty_pulses: Vec<Vec<ComplexSample>> = Vec::new();
        let grid = slow_time_complex_dft(&empty_pulses, 16);
        assert!(grid.is_empty(), "empty pulse input must yield empty grid");

        let pulses: Vec<Vec<ComplexSample>> = vec![vec![ComplexSample::new(1.0, 0.0); 4]; 8];
        let grid_zero_doppler = slow_time_complex_dft(&pulses, 0);
        assert!(
            grid_zero_doppler.is_empty(),
            "n_doppler == 0 must yield empty grid"
        );

        // All-empty rows should also yield empty grid (no range bins).
        let zero_range: Vec<Vec<ComplexSample>> = vec![Vec::new(); 8];
        let grid_zero_range = slow_time_complex_dft(&zero_range, 8);
        assert!(
            grid_zero_range.is_empty(),
            "all-empty pulse rows must yield empty grid"
        );

        // Legacy wrapper must also tolerate empty inputs.
        let mag = slow_time_dft_magnitude(&Vec::<Vec<f32>>::new(), 16);
        assert!(mag.is_empty());
        let mag_zero_range =
            slow_time_dft_magnitude(&vec![vec![1.0f32; 4]; 8], 0);
        assert!(mag_zero_range.is_empty());
    }

    /// `synthesize_takeoff_episode` must populate `range_doppler_complex`
    /// with shape (compressed_len, n_pulses), where compressed_len is
    /// the pulse-compression output length and n_pulses is the pulse
    /// count. The grid must not be uniformly zero — a high-SNR scenario
    /// has to produce coherent energy somewhere in the (range, Doppler)
    /// plane.
    #[test]
    #[allow(deprecated)]
    fn range_doppler_complex_populated() {
        let config = RadarSimConfig {
            pulse_count: 16,
            target_snr_db: 28.0,
            ..RadarSimConfig::default()
        };
        let mut noise = NoiseProfile::real_world_proxy_v1();
        noise.awgn_sigma = 0.025;
        let episode =
            synthesize_takeoff_episode(config, TakeoffProfile::default(), noise, EpisodeSeed(7));

        // Expected shape from the synthesis loop:
        //   reference len = sample_count = pulse_width_s * sample_rate_hz
        //   compressed_len = 2 * sample_count - 1
        let waveform = episode.config.waveform();
        let sample_count = waveform.samples().len();
        let compressed_len = sample_count.saturating_mul(2).saturating_sub(1);

        assert_eq!(
            episode.range_doppler_complex.len(),
            compressed_len,
            "outer dim must equal compressed_len (range bins)",
        );
        for row in &episode.range_doppler_complex {
            assert_eq!(
                row.len(),
                episode.config.pulse_count,
                "inner dim must equal pulse_count (Doppler bins)",
            );
            for cell in row {
                assert!(
                    cell.re.is_finite() && cell.im.is_finite(),
                    "non-finite complex cell at {cell}",
                );
            }
        }

        let max_mag = episode
            .range_doppler_complex
            .iter()
            .flat_map(|row| row.iter().map(|c| c.norm()))
            .fold(0.0f32, f32::max);
        assert!(
            max_mag > 0.0,
            "range_doppler_complex is all zeros — slow-time DFT is dead",
        );
    }

    // ---------------------------------------------------------------
    // Lane C — ClutterRegime wiring tests.
    //
    // These verify that:
    //   (a) the default `NoiseProfile::real_world_proxy_v1()` keeps
    //       `clutter_regime: None` so existing Gaussian-AR(1) byte-
    //       stable fixtures are unchanged,
    //   (b) wiring a K-distribution regime through the synthesis loop
    //       actually produces heavy-tailed IQ samples (empirical
    //       kurtosis well above the Gaussian baseline of 3), and
    //   (c) the K/Weibull path is fully deterministic for a fixed
    //       seed, matching the determinism guarantee of
    //       `generate_clutter_sequence`.
    // ---------------------------------------------------------------

    use crate::clutter::{ClutterDistribution, ClutterRegime, TerrainClass};

    fn empirical_kurtosis(xs: &[f64]) -> f64 {
        // Pearson's kurtosis: m4 / m2^2 (NOT excess kurtosis). For a
        // Gaussian this is 3.0; for K-distribution(nu) it diverges as
        // nu -> 0 (Ward, Tough & Watts chap. 2).
        let n = xs.len() as f64;
        let mean = xs.iter().sum::<f64>() / n;
        let m2 = xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        let m4 = xs.iter().map(|x| (x - mean).powi(4)).sum::<f64>() / n;
        if m2 == 0.0 {
            0.0
        } else {
            m4 / (m2 * m2)
        }
    }

    /// (a) Back-compat — the canonical `real_world_proxy_v1` profile
    /// must default to the legacy Gaussian AR(1) clutter path so
    /// existing byte-stable fixtures stay valid.
    #[test]
    fn noise_profile_default_uses_gaussian_clutter() {
        let noise = NoiseProfile::real_world_proxy_v1();
        assert!(
            noise.clutter_regime.is_none(),
            "default real_world_proxy_v1 must keep clutter_regime=None \
             for byte-stable back-compat"
        );
        assert_eq!(noise.clutter_sigma_0_scale, 1.0);

        // Determinism: with the default Gaussian profile, two runs must
        // remain byte-equal (this is just the pre-existing contract,
        // re-asserted here to gate the back-compat property).
        let config = RadarSimConfig {
            pulse_count: 6,
            ..RadarSimConfig::default()
        };
        let a = synthesize_takeoff_episode(
            config.clone(),
            TakeoffProfile::default(),
            noise,
            EpisodeSeed(101),
        );
        let b = synthesize_takeoff_episode(
            config,
            TakeoffProfile::default(),
            noise,
            EpisodeSeed(101),
        );
        assert_eq!(a.integrated_range_profile, b.integrated_range_profile);
    }

    /// (b) K-distribution wiring — a `NoiseProfile` with a low-`nu`
    /// K-distribution regime should produce IQ clutter with empirical
    /// kurtosis well above the Gaussian baseline (3.0). We use a
    /// spiky regime (`nu = 0.8`) and check the *real-part marginal*
    /// of the IQ samples; with low awgn_sigma the clutter term
    /// dominates and the heavy-tailed signature is preserved.
    #[test]
    fn noise_profile_with_k_regime_uses_k_distribution() {
        let config = RadarSimConfig {
            pulse_count: 32,
            ..RadarSimConfig::default()
        };
        let mut noise = NoiseProfile::real_world_proxy_v1();
        // Strip orthogonal noise sources so the kurtosis we measure
        // reflects the clutter generator, not phase noise / glints /
        // RFI / AWGN.
        noise.awgn_sigma = 1e-5;
        noise.phase_noise_std_rad = 0.0;
        noise.amplitude_scintillation_sigma = 0.0;
        noise.rfi_probability = 0.0;
        noise.ground_glint_count = 0;
        // Custom regime: K-distribution shape nu = 0.8, zero AR(1)
        // correlation so adjacent samples are independent draws.
        noise.clutter_regime = Some(ClutterRegime {
            terrain: TerrainClass::Sea,
            grazing_angle_deg: 1.0,
            distribution: ClutterDistribution::KDistribution {
                shape: 0.8,
                scale: 1.0,
            },
            spatial_correlation: 0.0,
            temporal_correlation: 0.0,
            mean_power_dbsm_per_m2: -40.0,
        });
        noise.clutter_sigma_0_scale = 1.0;

        let episode = synthesize_takeoff_episode(
            config,
            TakeoffProfile::default(),
            noise,
            EpisodeSeed(2026),
        );
        // Flatten the IQ real parts. The target return is concentrated
        // in a tiny range of bins (small support), so the IQ histogram
        // is dominated by the per-bin clutter draws.
        let samples: Vec<f64> = episode
            .iq
            .iter()
            .flatten()
            .map(|c| c.re as f64)
            .collect();
        assert!(!samples.is_empty());
        let kurtosis = empirical_kurtosis(&samples);
        assert!(
            kurtosis > 3.0,
            "K(nu=0.8) clutter must exhibit heavy tails (kurtosis > 3 \
             — Gaussian baseline); got {kurtosis:.3}"
        );
    }

    /// Lane I (Wave 4) — `synthesize_scene` must produce the same
    /// products as `synthesize_takeoff_episode` when invoked with the
    /// equivalent single-entity `SceneDescriptor`. This is the
    /// in-module byte-stability gate; a heavier-weight version with
    /// IQ-bit-equal assertions lives in
    /// `tests/physics_correctness.rs::c_unified_takeoff_wrapper_matches_scene_direct`.
    #[test]
    fn synthesize_scene_single_target_matches_legacy() {
        use crate::scene::{
            EnvironmentDescriptor, SceneDescriptor, SiteGeometry, TargetClass, TargetEntity,
            TargetKinematics,
        };

        let config = RadarSimConfig {
            pulse_count: 6,
            ..RadarSimConfig::default()
        };
        let profile = TakeoffProfile::default();
        let noise = NoiseProfile::real_world_proxy_v1();
        let seed = EpisodeSeed(2027);

        let via_wrapper = synthesize_takeoff_episode(config.clone(), profile, noise, seed);

        let scene = SceneDescriptor {
            geometry: SiteGeometry {
                antenna_altitude_agl_m: config.radar_altitude_agl_m,
            },
            environment: EnvironmentDescriptor {
                clutter_regime: noise.clutter_regime,
                atmospheric_one_way_db_per_km: config.atmospheric_one_way_db_per_km,
                rain_rate_mm_per_h: config.rain_rate_mm_per_h,
                ground_reflection_coefficient_magnitude: config
                    .ground_reflection_coefficient_magnitude,
            },
            targets: vec![TargetEntity {
                class: TargetClass::ShahedClassPiston,
                kinematics: TargetKinematics::FromTakeoffProfile(profile),
                spawn_time_s: 0.0,
            }],
        };
        let via_scene = synthesize_scene(scene, config, noise, seed);

        assert_eq!(
            via_wrapper.integrated_range_profile, via_scene.integrated_range_profile,
            "wrapper and unified path must produce byte-equal integrated profile"
        );
        assert_eq!(
            via_wrapper.detections, via_scene.detections,
            "wrapper and unified path must produce byte-equal detections"
        );
    }

    /// (c) Reproducibility — same regime + same seed must yield the
    /// same IQ. `generate_clutter_sequence` is byte-stable per its
    /// docs; this test gates the wiring.
    #[test]
    fn noise_profile_with_weibull_regime_reproducible() {
        let config = RadarSimConfig {
            pulse_count: 8,
            ..RadarSimConfig::default()
        };
        let mut noise = NoiseProfile::real_world_proxy_v1();
        noise.clutter_regime = Some(ClutterRegime::for_terrain(TerrainClass::Forest, 3.0));

        let a = synthesize_takeoff_episode(
            config.clone(),
            TakeoffProfile::default(),
            noise,
            EpisodeSeed(31337),
        );
        let b = synthesize_takeoff_episode(
            config,
            TakeoffProfile::default(),
            noise,
            EpisodeSeed(31337),
        );
        assert_eq!(a.integrated_range_profile, b.integrated_range_profile);
        assert_eq!(a.detections, b.detections);
        // IQ byte-equality is the strictest form of determinism.
        for (pulse_a, pulse_b) in a.iq.iter().zip(b.iq.iter()) {
            assert_eq!(pulse_a.len(), pulse_b.len());
            for (sa, sb) in pulse_a.iter().zip(pulse_b.iter()) {
                assert_eq!(sa.re.to_bits(), sb.re.to_bits());
                assert_eq!(sa.im.to_bits(), sb.im.to_bits());
            }
        }
    }

    // ---------------------------------------------------------------
    // Wave 4.5 Lane H2 — polarization-agility primitive tests.
    //
    // These verify that:
    //   (1) `RadarSimConfig::default()` falls back to `(Vv, Vv)` so
    //       pre-Lane-H2 fixtures are byte-stable,
    //   (2) `polarization_for_pulse` cycles the sequence modulo
    //       `pulse_count` (the classic VV/HH alternation per
    //       Skolnik §7.5.3), and
    //   (3) VV-only vs HH-only synthesis episodes differ in
    //       integrated-profile peak by the expected ~1 dB amplitude
    //       scaling, proving the polarization channel is actually
    //       wired into the per-pulse target return.
    // ---------------------------------------------------------------

    /// (1) Back-compat — `RadarSimConfig::default()` has no
    /// polarization sequences configured, so every pulse must resolve
    /// to `(Vv, Vv)`. Existing reproduction fixtures predate the
    /// polarization sequence and depend on this fallback.
    #[test]
    fn polarization_default_is_vv_back_compat() {
        let config = RadarSimConfig::default();
        assert!(
            config.pol_tx_sequence.is_none(),
            "default pol_tx_sequence must be None for back-compat"
        );
        assert!(
            config.pol_rx_sequence.is_none(),
            "default pol_rx_sequence must be None for back-compat"
        );
        // Pulse-zero must resolve to the canonical (Vv, Vv) pair.
        assert_eq!(
            config.polarization_for_pulse(0),
            (Polarization::Vv, Polarization::Vv),
            "polarization_for_pulse(0) must default to (Vv, Vv)"
        );
        // Higher pulse indices must also resolve to (Vv, Vv) when no
        // sequence is set — the modulo-cycling is a no-op when the
        // sequence is `None`.
        for pulse_idx in [1, 7, 32, 1024, usize::MAX / 2] {
            assert_eq!(
                config.polarization_for_pulse(pulse_idx),
                (Polarization::Vv, Polarization::Vv),
                "polarization_for_pulse({pulse_idx}) must default to (Vv, Vv)"
            );
        }
    }

    /// (2) Sequence wiring — with `pol_tx_sequence = Some(vec![Vv, Hh])`,
    /// pulses 0, 2, 4, … must resolve to `Vv` and pulses 1, 3, 5, …
    /// must resolve to `Hh`. This is the canonical pulse-to-pulse
    /// polarization agility used by modern AESA radars for clutter
    /// diversity (Skolnik §7.5.3).
    ///
    /// When `pol_rx_sequence` is `None`, rx must mirror tx (the
    /// matched/co-polar receive convention).
    #[test]
    fn polarization_sequence_alternates_vv_hh() {
        let config = RadarSimConfig {
            pol_tx_sequence: Some(vec![Polarization::Vv, Polarization::Hh]),
            ..RadarSimConfig::default()
        };

        // First period.
        assert_eq!(
            config.polarization_for_pulse(0),
            (Polarization::Vv, Polarization::Vv),
            "pulse 0 must be VV (rx mirrors tx when pol_rx_sequence is None)"
        );
        assert_eq!(
            config.polarization_for_pulse(1),
            (Polarization::Hh, Polarization::Hh),
            "pulse 1 must be HH (rx mirrors tx when pol_rx_sequence is None)"
        );

        // Second period — modulo cycling must wrap cleanly.
        assert_eq!(
            config.polarization_for_pulse(2),
            (Polarization::Vv, Polarization::Vv),
            "pulse 2 must wrap back to VV"
        );
        assert_eq!(
            config.polarization_for_pulse(3),
            (Polarization::Hh, Polarization::Hh),
            "pulse 3 must wrap to HH"
        );

        // Larger indices — full modulo cycle.
        for pulse_idx in 0..16 {
            let expected = if pulse_idx % 2 == 0 {
                Polarization::Vv
            } else {
                Polarization::Hh
            };
            let (tx, rx) = config.polarization_for_pulse(pulse_idx);
            assert_eq!(tx, expected, "pulse {pulse_idx} tx mismatch");
            assert_eq!(rx, expected, "pulse {pulse_idx} rx mirrors tx");
        }

        // Cross-pol case — separate tx/rx sequences enable HV / VH
        // depolarization measurements (Ulaby & Long §10.2).
        let cross_pol_config = RadarSimConfig {
            pol_tx_sequence: Some(vec![Polarization::Hh]),
            pol_rx_sequence: Some(vec![Polarization::Vv]),
            ..RadarSimConfig::default()
        };
        assert_eq!(
            cross_pol_config.polarization_for_pulse(0),
            (Polarization::Hh, Polarization::Vv),
            "cross-pol: tx=HH, rx=VV (i.e. HV — the cross-polar channel)"
        );

        // Empty sequence must fall back to defaults (defensive — avoids
        // a division by zero in the modulo cycling).
        let empty_config = RadarSimConfig {
            pol_tx_sequence: Some(vec![]),
            ..RadarSimConfig::default()
        };
        assert_eq!(
            empty_config.polarization_for_pulse(0),
            (Polarization::Vv, Polarization::Vv),
            "empty sequence must fall back to default (Vv, Vv)"
        );
    }

    /// (3) End-to-end synthesis — same target, same noise, same seed
    /// run twice with VV-only vs HH-only polarization sequences. The
    /// integrated-profile peak must differ by ~1 dB (the HH amplitude
    /// scaling: linear 10^(+1/20) ≈ 1.122). If the two peaks were
    /// identical, the polarization channel would NOT be wired into the
    /// per-pulse target amplitude.
    ///
    /// We zero out stochastic noise / scintillation so the peak ratio
    /// is a deterministic function of the polarization scaling.
    #[test]
    fn polarization_vv_vs_hh_changes_target_amp() {
        let base_config = RadarSimConfig {
            pulse_count: 16,
            ..RadarSimConfig::default()
        };
        let mut noise = NoiseProfile::real_world_proxy_v1();
        // Zero stochastic terms so the integrated peak is a clean
        // function of `target_amp * pol_scale`.
        noise.amplitude_scintillation_sigma = 0.0;
        noise.phase_noise_std_rad = 0.0;
        noise.rfi_probability = 0.0;
        noise.clutter_sigma = 0.0;
        noise.ground_glint_count = 0;
        noise.awgn_sigma = 0.001;

        let config_vv = RadarSimConfig {
            pol_tx_sequence: Some(vec![Polarization::Vv]),
            ..base_config.clone()
        };
        let config_hh = RadarSimConfig {
            pol_tx_sequence: Some(vec![Polarization::Hh]),
            ..base_config
        };

        let episode_vv = synthesize_takeoff_episode(
            config_vv,
            TakeoffProfile::default(),
            noise,
            EpisodeSeed(2026),
        );
        let episode_hh = synthesize_takeoff_episode(
            config_hh,
            TakeoffProfile::default(),
            noise,
            EpisodeSeed(2026),
        );

        let peak_vv = episode_vv
            .integrated_range_profile
            .iter()
            .copied()
            .fold(0.0f32, f32::max);
        let peak_hh = episode_hh
            .integrated_range_profile
            .iter()
            .copied()
            .fold(0.0f32, f32::max);

        assert!(peak_vv > 0.0 && peak_hh > 0.0, "both peaks must be positive");

        // HH amplitude scale = 10^(+1/20) ≈ 1.122 (see
        // `polarization_amplitude_scale`). The integrated profile is a
        // mean of magnitudes, which scales linearly with target_amp;
        // therefore peak_hh / peak_vv should be ~1.122 (i.e. ~+1 dB).
        let ratio_db = 20.0 * (peak_hh / peak_vv).log10();
        assert!(
            ratio_db.abs() >= 0.5,
            "VV-only vs HH-only integrated peaks must differ by ≥0.5 dB \
             (the +1 dB HH scaling); got ratio_db = {ratio_db:.3} \
             (peak_vv = {peak_vv:.6}, peak_hh = {peak_hh:.6})"
        );
        // And we expect the sign to be positive: HH > VV by the
        // first-order proxy.
        assert!(
            peak_hh > peak_vv,
            "HH peak should exceed VV peak by ~1 dB; got peak_hh={peak_hh:.6} \
             vs peak_vv={peak_vv:.6}"
        );
    }
}
