use std::f32::consts::PI;

use serde::{Deserialize, Serialize};

use crate::link_budget::LinkBudgetResult;
use crate::ComplexSample;

use super::config::{NoiseProfile, RadarSimConfig, TakeoffProfile};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct EpisodeSeed(pub u64);

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TargetState {
    pub time_s: f64,
    pub range_m: f64,
    pub altitude_m: f64,
    pub radial_velocity_mps: f64,
    pub pitch_deg: f64,
    pub yaw_deg: f64,
    #[serde(default)]
    pub course_deg: f64,
    pub propulsor_phase_rad: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SourceDiagnostics {
    pub pulse_index: usize,
    pub entity_index: usize,
    pub class_name: String,
    pub active: bool,
    pub time_s: f64,
    pub range_m: f64,
    pub altitude_m: f64,
    pub aspect_deg: f64,
    pub elevation_deg: f64,
    pub rcs_dbsm: f64,
    pub rcs_m2: f64,
    pub received_power_w: f64,
    pub thermal_noise_power_w: f64,
    pub clutter_power_w: f64,
    pub interference_power_w: f64,
    pub propagation_loss_db: f64,
    pub processing_loss_db: f64,
    pub sinr_db: f64,
    #[serde(default)]
    pub masked_by_receive_window: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PulseDiagnostics {
    pub pulse_index: usize,
    pub time_s: f64,
    pub source_diagnostics: Vec<SourceDiagnostics>,
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
    /// is the diagnostic equivalent of the superseded
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
    /// (which is its magnitude image, retained only for bridged with
    /// pre-Lane-B consumers).
    pub range_doppler_complex: Vec<Vec<ComplexSample>>,
    /// **Wave 5 Lane J — per-entity diagnostic SNR (dB).** For each
    /// entity in `SceneDescriptor::targets`, this vector carries the
    /// emergent post-integration SNR computed from the entity's
    /// initial geometry and a per-class RCS proxy (see
    /// `class_default_rcs_scalar` in `crate::sim`). Length equals
    /// `scene.targets.len()`. For single-entity scenes this is
    /// `vec![diagnostic_snr_db]`, preserving the prior field.
    /// Multipath ghost entities (`TargetClass::MultipathGhost`) inherit
    /// their parent's SNR scaled by `20·log10(|Γ|)` so a ghost's
    /// entry reflects the reduced phantom-return level.
    pub per_target_snr_db: Vec<f64>,
    /// Per-pulse, per-entity diagnostics emitted by the live scene
    /// link-budget path. These are public-proxy diagnostics only and
    /// are intended for QA / validation products, not as exact truth.
    pub pulse_diagnostics: Vec<PulseDiagnostics>,
}

/// Deterministic PRNG (SplitMix64 + Box-Muller Gaussian).
/// Used exclusively inside the synthesis functions to produce
/// reproducible stochastic IQ draws for a given seed.
#[derive(Debug, Clone)]
pub(super) struct SplitMix64 {
    pub(super) state: u64,
    pub(super) cached_normal: Option<f32>,
}

impl SplitMix64 {
    pub(super) fn new(seed: u64) -> Self {
        Self {
            state: seed,
            cached_normal: None,
        }
    }

    pub(super) fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }

    pub(super) fn unit_f32(&mut self) -> f32 {
        let bits = (self.next_u64() >> 40) as u32;
        (bits as f32) / ((1u32 << 24) as f32)
    }

    pub(super) fn normal_f32(&mut self) -> f32 {
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

    pub(super) fn normal_scaled(&mut self, sigma: f32) -> f32 {
        self.normal_f32() * sigma
    }
}
