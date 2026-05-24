//! NoiseProfile — extracted from config.rs for LOC compliance.

use serde::{Deserialize, Serialize};

use crate::clutter::ClutterRegime;

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
    /// loop falls back to the prior Gaussian AR(1) for byte-stable
    /// bridged with pre-Lane-C fixtures.
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
            // bridged: keep the prior Gaussian AR(1) path so
            // existing fixtures and byte-stable tests are unchanged.
            clutter_regime: None,
            clutter_sigma_0_scale: 1.0,
        }
    }
}
